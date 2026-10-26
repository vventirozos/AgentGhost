// Eva's World hub (runs on nova, always on under launchd).
//
// One process that:
//   - starts / pauses / stops the world (Paper server + Mindcraft bot);
//   - listens to Mindcraft's control server and the two log files and
//     records every event, append-only, per session;
//   - serves the monitor page and the API Eva's mind (on the Mac) polls;
//   - proxies the 3D viewer, which listens on localhost only.
// Everything except /api/login and the login page needs the hub key.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFile, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { io as ioClient } from 'socket.io-client';

import { rcon } from './lib/rcon.js';
import { parseServerLine, MindcraftParser } from './lib/parse.js';
import { buildDiary } from './lib/diary.js';
import { Attention } from './lib/attention.js';
import { planDirectives } from './lib/directives.js';
import { Store } from './lib/store.js';
import { Proc } from './lib/procs.js';
import { Tail } from './lib/tail.js';
import { proxyRequest, proxyUpgrade } from './lib/proxy.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, '..');                       // ~/Data/AI/World/app
const WORLD = path.resolve(APP, '..');                      // ~/Data/AI/World
const CFG = {
  bind: '100.83.184.117', port: 8090, botName: 'Eva', mindName: 'Mind',
  gemmaUrl: 'http://127.0.0.1:8088', java: '/opt/homebrew/opt/openjdk@21/bin/java',
  node: '/opt/homebrew/opt/node@20/bin/node', serverHeap: '1536m', viewerPort: 3000,
  mindserverPort: 8080, rconPort: 25575,
  ...JSON.parse(fs.existsSync(path.join(APP, 'config', 'world.json')) ? fs.readFileSync(path.join(APP, 'config', 'world.json'), 'utf8') : '{}'),
};
const RUN = path.join(WORLD, 'run');
fs.mkdirSync(RUN, { recursive: true });
const KEY = fs.readFileSync(path.join(WORLD, '.hub_key'), 'utf8').trim();
const RCON_PW = () => fs.readFileSync(path.join(WORLD, 'server', '.rcon_password'), 'utf8').trim();
const STATE_FILE = path.join(RUN, 'state.json');
// RCON binds to server-ip (the Tailscale address), not to localhost.
const RC = (cmd, extra = {}) => rcon(cmd, { host: CFG.bind, port: CFG.rconPort, password: RCON_PW(), ...extra });
const SNAPSHOT_MS = 30000;

const log = (...a) => console.log(new Date().toISOString(), ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── persistent world state ──────────────────────────────────────────────
const store = new Store(path.join(WORLD, 'data'));
let W = { phase: 'stopped', session: null, wakeAt: null, lastLookTs: null, currentGoal: null, error: null };
try { W = { ...W, ...JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')) }; } catch { /* first run */ }
const saveState = () => fs.writeFileSync(STATE_FILE, JSON.stringify(W, null, 2));
const attention = new Attention();
let latestState = null;
let lastSnapshot = 0;
let mind = { status: 'never seen', detail: '', ts: null };
const gemma = { waitStart: null, latencies: [] };
let memFreePct = null;

const server = new Proc({
  name: 'server', cmd: CFG.java, match: 'paper-',
  args: [`-Xms512m`, `-Xmx${CFG.serverHeap}`, '-jar', fs.readdirSync(path.join(WORLD, 'server')).find((f) => /^paper-.*\.jar$/.test(f)) || 'paper.jar', '--nogui'],
  cwd: path.join(WORLD, 'server'), logFile: path.join(RUN, 'server.log'), pidFile: path.join(RUN, 'server.pid'),
});
const mindcraft = new Proc({
  name: 'mindcraft', cmd: CFG.node, match: 'main.js', args: ['main.js'],
  cwd: path.join(WORLD, 'mindcraft'), logFile: path.join(RUN, 'mindcraft.log'), pidFile: path.join(RUN, 'mindcraft.pid'),
});

function record(kind, fields) {
  const e = store.append(kind, fields);
  attention.onEvent(e);
  return e;
}

// ── log tails ───────────────────────────────────────────────────────────
let serverReadyAt = 0;
new Tail(server.logFile, (line) => {
  const e = parseServerLine(line, CFG.botName);
  if (!e) return;
  if (e.kind === 'server_ready') { serverReadyAt = Date.now(); return; }
  if (W.session) record(e.kind, e);
}).start();

const mcParser = new MindcraftParser(CFG.botName);
new Tail(mindcraft.logFile, (line) => {
  for (const e of mcParser.feed(line)) onMindcraftEvent(e);
}).start();

function onMindcraftEvent(e) {
  if (e.kind === 'model_wait') { gemma.waitStart = Date.now(); return; }
  if (e.kind === 'model_done') {
    if (gemma.waitStart) { gemma.latencies.push(Date.now() - gemma.waitStart); if (gemma.latencies.length > 30) gemma.latencies.shift(); }
    gemma.waitStart = null; return;
  }
  if (e.kind === 'goal_started') return;
  if (W.session) record(e.kind, e);
}

// ── Mindcraft control server ────────────────────────────────────────────
let ms = null;
function connectMindserver() {
  if (ms) return;
  // Mindcraft listens on 'localhost', which Node resolves to ::1 here: dial
  // the same name, not 127.0.0.1 (that connection is refused).
  ms = ioClient(`http://localhost:${CFG.mindserverPort}`, { reconnection: true, reconnectionDelay: 2000 });
  ms.on('connect', () => { log('mindserver connected'); ms.emit('listen-to-agents'); });
  ms.on('state-update', (states) => {
    const s = states?.[CFG.botName];
    if (!s || s.error) return;
    latestState = s;
    const now = Date.now();
    latestStateTs = now;
    attention.onState(s, now);
    if (W.session && now - lastSnapshot > SNAPSHOT_MS) {
      lastSnapshot = now;
      const g = s.gameplay || {};
      store.append('snapshot', { pos: g.position, health: g.health, hunger: g.hunger, time: g.timeOfDay, biome: g.biome,
        action: s.action?.current, inventory: s.inventory?.counts });
    }
  });
  ms.on('bot-output', (agent, message) => {
    if (agent === CFG.botName && W.session) record('bot_output', { text: String(message) });
  });
}
function disconnectMindserver() { if (ms) { ms.close(); ms = null; } latestState = null; }
// Mindcraft sends state once a second, only for a bot that is in the game;
// silence means the body is gone, whatever the last state said.
let latestStateTs = 0;
function bodyInGame() { return !!latestState?.gameplay && Date.now() - latestStateTs < 10000; }

// ── body watchdog ───────────────────────────────────────────────────────
// The body can drop out (first session: kicked for an invalid move packet;
// Mindcraft restarted it once and then gave up for good). While the world is
// running, a body missing for LOST_GRACE_MS is restarted with backoff:
// start-agent first, the whole Mindcraft process after PROCESS_AFTER failed
// tries. Every loss and return is recorded and wakes Eva.
const LOST_GRACE_MS = 20000;
const BACKOFF_MS = [30000, 60000, 180000, 600000];
const PROCESS_AFTER = 3;
const STABLE_MS = 10 * 60000;
const dog = { lostSince: null, attempts: 0, nextAt: 0, backSince: null };

async function watchdog() {
  const now = Date.now();
  if (W.phase !== 'running') { dog.lostSince = null; return; }
  // busy (a control, a decision, or this watchdog's own restart): skip the
  // tick but keep the clock — clearing it would reset the backoff to the grace
  if (busy) return;
  if (bodyInGame()) {
    if (dog.lostSince) {
      record('body_back', { text: `your body is back in the world after ${Math.round((now - dog.lostSince) / 1000)} s`, attempts: dog.attempts });
      dog.lostSince = null; dog.backSince = now;
    }
    if (dog.backSince && now - dog.backSince > STABLE_MS) { dog.attempts = 0; dog.backSince = null; }
    return;
  }
  if (!dog.lostSince) { dog.lostSince = now; dog.nextAt = now + LOST_GRACE_MS; return; }
  if (now < dog.nextAt) return;
  if (dog.attempts === 0) record('body_lost', { text: 'your body dropped out of the world' });
  dog.attempts += 1;
  dog.nextAt = now + BACKOFF_MS[Math.min(dog.attempts - 1, BACKOFF_MS.length - 1)];
  // Holds the control lock: a Stop or Pause pressed mid-restart waits for it
  // instead of finishing first and having Mindcraft started behind it.
  busy = true;
  try {
    if (dog.attempts > PROCESS_AFTER || !mindcraft.alive() || !ms?.connected) {
      record('body_restart', { text: `restarting Mindcraft (attempt ${dog.attempts})`, attempt: dog.attempts });
      disconnectMindserver();
      await mindcraft.stop(10000);
      execFileSync('/bin/bash', [path.join(APP, 'scripts', 'apply_patches.sh')], { stdio: 'pipe' });
      mindcraft.start();
      await sleep(3000);
      connectMindserver();
    } else {
      record('body_restart', { text: `restarting the body (attempt ${dog.attempts})`, attempt: dog.attempts });
      ms.emit('start-agent', CFG.botName);
    }
  } catch (e) {
    record('error', { text: `body restart failed: ${e.message}` });
  } finally {
    busy = false;
  }
}

function resetDog() { Object.assign(dog, { lostSince: null, attempts: 0, nextAt: 0, backSince: null }); }
setInterval(() => { watchdog().catch((e) => log('watchdog', e.message)); }, 5000);

function sendToBody(message) {
  if (!ms?.connected) throw new Error('mindserver not connected');
  ms.emit('send-message', CFG.botName, { from: CFG.mindName, message });
}

// ── health probes ───────────────────────────────────────────────────────
async function probeGemma() {
  const t0 = Date.now();
  const r = await fetch(`${CFG.gemmaUrl}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, signal: AbortSignal.timeout(90000),
    body: JSON.stringify({ messages: [{ role: 'user', content: 'Reply with the single word: ready' }], max_tokens: 8,
      chat_template_kwargs: { enable_thinking: false }, cache_prompt: false }),
  });
  if (!r.ok) throw new Error(`gemma HTTP ${r.status}`);
  const j = await r.json();
  if (!j.choices?.[0]?.message?.content?.trim()) throw new Error('gemma returned an empty reply');
  return Date.now() - t0;
}

function pollMemory() {
  execFile('memory_pressure', [], { timeout: 10000 }, (err, out) => {
    const m = /free percentage: (\d+)%/.exec(out || '');
    if (m) memFreePct = Number(m[1]);
  });
}
setInterval(pollMemory, 30000); pollMemory();

// ── world control ───────────────────────────────────────────────────────
let busy = false;
// One thing at a time touches the world: controls, the watchdog's restarts and
// Eva's decisions. A caller waits its turn (a restart takes ~15 s) rather
// than racing it.
async function withLock(fn, waitMs = 60000) {
  const until = Date.now() + waitMs;
  while (busy) {
    if (Date.now() > until) throw new Error('another control action is still running');
    await sleep(250);
  }
  busy = true;
  try { return await fn(); } finally { busy = false; }
}

async function control(action, b = {}) {
  return withLock(() => controlLocked(action, b));
}

async function controlLocked(action, b) {
  {
    if (action === 'start') return await startWorld();
    if (action === 'stop') return await stopWorld();
    if (action === 'pause') return await pauseWorld();
    if (action === 'resume') return await resumeWorld();
    if (action === 'difficulty') return await setDifficulty(String(b.level || ''));
    if (action === 'note') return noteIntervention(b.text);
    throw new Error(`unknown action ${action}`);
  }
}

// Live via RCON; server.properties keeps it across restarts. Recorded as an
// operator intervention so the research record shows when the world changed.
const DIFFICULTIES = ['peaceful', 'easy', 'normal', 'hard'];
async function setDifficulty(level) {
  if (!DIFFICULTIES.includes(level)) throw new Error(`difficulty must be one of ${DIFFICULTIES.join(', ')}`);
  if (W.phase === 'running' || W.phase === 'paused') await RC(`difficulty ${level}`);
  const props = path.join(WORLD, 'server', 'server.properties');
  fs.writeFileSync(props, fs.readFileSync(props, 'utf8').replace(/^difficulty=.*$/m, `difficulty=${level}`));
  recordIntervention({ action: 'difficulty', level, text: `set the difficulty to ${level}` });
  return `difficulty ${level}`;
}

// An operator change to the harness itself (a fix to the hub, the body's
// settings...), so the record shows when the conditions changed. Eva's diary
// shows it as "the operator intervened: <text>".
function noteIntervention(text) {
  const t = String(text ?? '').trim().slice(0, 1000);
  if (!t) throw new Error('note needs a text');
  recordIntervention({ action: 'note', text: t });
  return 'noted';
}

function recordIntervention(ev) {
  // stopped: no session file to write to — recorded when the next one starts
  if (W.session) record('intervention', ev);
  else { W.pendingInterventions = [...(W.pendingInterventions || []), { ...ev, at: Date.now() }]; saveState(); }
}

async function waitFor(fn, ms, what) {
  const until = Date.now() + ms;
  while (Date.now() < until) { if (await fn()) return; await sleep(1000); }
  throw new Error(`timed out waiting for ${what}`);
}

async function startWorld() {
  if (W.phase !== 'stopped') throw new Error(`world is ${W.phase}`);
  W = { ...W, phase: 'starting', error: null };
  saveState();
  const fresh = !W.session;
  if (fresh) W.session = store.startSession(); else store.resumeSession(W.session);
  record('control', { action: 'start', text: 'world starting' });
  for (const iv of W.pendingInterventions || []) record('intervention', iv);
  W.pendingInterventions = [];
  saveState();
  resetDog();
  try {
    const ms0 = await probeGemma();
    record('control', { action: 'gemma_ok', text: `Gemma answered in ${ms0} ms` });
    execFileSync('/bin/bash', [path.join(APP, 'scripts', 'apply_patches.sh')], { stdio: 'pipe' });
    serverReadyAt = 0;
    if (!server.alive()) server.start();
    await waitFor(async () => { try { await RC('list'); return true; } catch { return false; } }, 240000, 'the Minecraft server');
    await RC('tick unfreeze').catch(() => {});
    if (!mindcraft.alive()) mindcraft.start();
    await sleep(3000);
    connectMindserver();
    await waitFor(() => bodyInGame(), 180000, 'the body to join the world');
    W.phase = 'running';
    W.wakeAt = Date.now();          // Eva looks as soon as the body is in
    attention.raise(fresh ? 'a new session began' : 'the world was started again', Date.now());
    saveState();
    record('control', { action: 'running', text: 'world running' });
    return 'running';
  } catch (e) {
    record('error', { text: `start failed: ${e.message}` });
    W.error = e.message;
    await teardown();
    throw e;
  }
}

async function teardown() {
  try { await RC('save-all flush', { timeoutMs: 20000 }); } catch { /* server down */ }
  disconnectMindserver();
  await mindcraft.stop(10000);
  try { await RC('stop'); } catch { /* down */ }
  if (!(await server.waitExit(60000))) await server.stop(10000);
  else await server.stop(1000);
  W.phase = 'stopped';
  W.session = null; W.wakeAt = null; W.lastLookTs = null; W.currentGoal = null;
  store.endSession();
  resetDog();
  attention.clear();
  saveState();
}

async function stopWorld() {
  if (W.phase === 'stopped') return 'already stopped';
  W.phase = 'stopping'; saveState();
  record('control', { action: 'stop', text: 'world stopping' });
  record('session_end', { text: 'session ended' });
  await teardown();
  return 'stopped';
}

async function pauseWorld() {
  if (W.phase !== 'running') throw new Error(`world is ${W.phase}`);
  await RC('tick freeze');
  ms?.emit('stop-agent', CFG.botName);
  W.phase = 'paused'; saveState();
  record('paused', { text: 'paused by operator' });
  return 'paused';
}

async function resumeWorld() {
  if (W.phase !== 'paused') throw new Error(`world is ${W.phase}`);
  await RC('tick unfreeze');
  ms?.emit('start-agent', CFG.botName);
  W.phase = 'running'; saveState();
  record('resumed', { text: 'resumed by operator' });
  return 'running';
}

// Hub restarted while a world was up: adopt it.
if (W.phase === 'running' || W.phase === 'paused') {
  if (server.alive() && mindcraft.alive()) {
    store.resumeSession(W.session);
    connectMindserver();
    log(`adopted running world, session ${W.session}`);
  } else {
    log('world processes are gone; marking stopped');
    if (W.session) { store.resumeSession(W.session); store.append('error', { text: 'world processes ended while the hub was down' }); }
    withLock(teardown).catch((e) => log('startup teardown', e.message));
  }
} else if (W.phase === 'starting' || W.phase === 'stopping') {
  withLock(teardown).catch((e) => log('startup teardown', e.message));
}

// ── Eva's mind ──────────────────────────────────────────────────────────
async function gameDay() {
  try {
    const r = await RC('time query day');
    const m = /(\d+)/.exec(r); return m ? Number(m[1]) : null;
  } catch { return null; }
}

// Every poll is a heartbeat: the monitor shows "offline" once they stop.
const MIND_OFFLINE_MS = 120000;
let mindLastPoll = 0;

function mindView(now = Date.now()) {
  if (!mindLastPoll) return mind;
  if (now - mindLastPoll > MIND_OFFLINE_MS) return { ...mind, status: 'offline', detail: `no poll for ${Math.round((now - mindLastPoll) / 1000)} s` };
  return mind.status === 'never seen' ? { ...mind, status: 'idle' } : mind;
}

function mindPoll() {
  const now = Date.now();
  mindLastPoll = now;
  const due = W.phase === 'running' && bodyInGame() && attention.due(now, W.lastLookTs, W.wakeAt);
  return { phase: W.phase, session: W.session, due, reasons: attention.pending(), wakeAt: W.wakeAt, lastLookTs: W.lastLookTs, now };
}

async function mindContext() {
  const now = Date.now();
  const reasons = attention.pending();
  if (W.wakeAt && now >= W.wakeAt && !reasons.length) reasons.push('the time you chose to look again');
  const events = store.eventsBetween(W.lastLookTs || 0, now).filter((e) => e.kind !== 'snapshot' && e.kind !== 'eva_decision');
  const diary = buildDiary({ state: latestState, day: await gameDay(), events, lastLookTs: W.lastLookTs, now, reasons, currentGoal: W.currentGoal });
  return { session: W.session, look_ts: now, diary, journal: store.journalTail(), reasons, current_goal: W.currentGoal };
}

async function mindDecision(body) {
  return withLock(() => mindDecisionLocked(body));
}

async function mindDecisionLocked(body) {
  if (W.phase !== 'running') { const e = new Error(`world is ${W.phase}; decision discarded`); e.status = 409; throw e; }
  const { actions, wakeInMinutes } = planDirectives(body.decision, CFG.botName);
  const applied = [];
  for (const a of actions) {
    try {
      // A running action (e.g. !stay(-1), "stay forever") keeps Mindcraft's
      // self-prompt loop from ever acting on a new goal: interrupt it first.
      if (a.interrupt) { sendToBody('!stop'); await sleep(1500); }
      if (a.channel === 'body') { sendToBody(a.payload); await sleep(1500); }
      else if (a.channel === 'rcon') await RC(a.payload);
      else if (a.channel === 'journal') store.appendJournal(a.payload);
      if (a.goal) W.currentGoal = a.goal;
      if (a.payload === '!stop') W.currentGoal = null;
      applied.push(a.summary);
      store.append('directive', { summary: a.summary, channel: a.channel });
    } catch (e) {
      applied.push(`FAILED (${e.message}): ${a.summary}`);
      store.append('error', { text: `directive failed: ${a.summary}: ${e.message}` });
    }
  }
  const now = Date.now();
  store.append('eva_decision', {
    look_ts: body.look_ts, reasons: attention.pending(), diary: body.diary, reasoning: body.reasoning || '',
    reply: body.reply || '', decision: body.decision || null, parse_error: body.parse_error || null,
    applied, wake_in_minutes: wakeInMinutes, prompt_hash: body.prompt_hash || null, timings: body.timings || null,
  });
  W.lastLookTs = Number(body.look_ts) || now;
  W.wakeAt = now + wakeInMinutes * 60000;
  attention.clearUpTo(W.lastLookTs);
  saveState();
  return { applied, wake_at: W.wakeAt };
}

function mindStatus(body) {
  const status = String(body.status || '').slice(0, 60);
  const detail = String(body.detail || '').slice(0, 500);
  const changed = status !== mind.status;
  mind = { status, detail, ts: Date.now() };
  if (changed && W.session && /skip|interrupt|error|abort/i.test(status)) store.append('mind_status', { status, detail });
}

// ── HTTP ────────────────────────────────────────────────────────────────
function authed(req) {
  const hdr = req.headers['x-world-key'];
  const cookie = /(?:^|;\s*)world_key=([^;]+)/.exec(req.headers.cookie || '')?.[1];
  let given = hdr;
  if (!given && cookie) { try { given = decodeURIComponent(cookie); } catch { return false; } }
  if (!given) return false;
  const a = Buffer.from(String(given)); const b = Buffer.from(KEY);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function send(res, code, obj, headers = {}) {
  const body = typeof obj === 'string' ? obj : JSON.stringify(obj);
  res.writeHead(code, { 'content-type': typeof obj === 'string' ? 'text/plain; charset=utf-8' : 'application/json', 'cache-control': 'no-store', ...headers });
  res.end(body);
}

function readBody(req, limit = 2 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let n = 0; const chunks = [];
    req.on('data', (c) => { n += c.length; if (n > limit) { reject(new Error('body too large')); req.destroy(); } else chunks.push(c); });
    req.on('end', () => { try { resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {}); } catch { reject(new Error('bad JSON')); } });
  });
}

const STATIC = { '/': 'index.html', '/index.html': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css' };
// The monitor loads nothing from outside nova: the browser enforces it.
const CSP = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'";
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8' };

function status() {
  const lat = gemma.latencies;
  return {
    phase: W.phase, session: W.session, error: W.error, wakeAt: W.wakeAt, lastLookTs: W.lastLookTs, currentGoal: W.currentGoal,
    reasons: attention.pending(), state: bodyInGame() ? latestState : null, mind: mindView(), now: Date.now(), memFreePct,
    gemma: { last_ms: lat.at(-1) ?? null, avg_ms: lat.length ? Math.round(lat.reduce((a, b) => a + b, 0) / lat.length) : null, waiting: !!gemma.waitStart },
    procs: { server: server.alive(), mindcraft: mindcraft.alive(), mindserver: !!ms?.connected }, controlBusy: busy,
    body: { inGame: bodyInGame(), lostSince: dog.lostSince, restarts: dog.attempts },
  };
}

const httpServer = http.createServer(async (req, res) => {
  let url, p;
  try { url = new URL(req.url, 'http://x'); p = url.pathname; } catch { return send(res, 400, { error: 'bad url' }); }
  try {
    if (req.method === 'GET' && STATIC[p]) {
      const f = path.join(HERE, 'public', STATIC[p]);
      return send(res, 200, fs.readFileSync(f, 'utf8'), { 'content-type': MIME[path.extname(f)], 'content-security-policy': CSP });
    }
    if (p === '/api/login' && req.method === 'POST') {
      const b = await readBody(req);
      const ok = typeof b.key === 'string' && b.key.length === KEY.length && crypto.timingSafeEqual(Buffer.from(b.key), Buffer.from(KEY));
      if (!ok) { await sleep(1000); return send(res, 401, { error: 'wrong key' }); }
      return send(res, 200, { ok: true }, { 'set-cookie': `world_key=${encodeURIComponent(KEY)}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000` });
    }
    if (!authed(req)) return send(res, 401, { error: 'key required' });

    if (p.startsWith('/viewer')) return proxyRequest(req, res, CFG.viewerPort);
    if (p === '/api/status') return send(res, 200, status());
    if (p === '/api/events') return send(res, 200, store.since(Number(url.searchParams.get('since') || 0), Number(url.searchParams.get('limit') || 500)).filter((x) => x.session === W.session));
    if (p === '/api/stream') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
      const since = Number(url.searchParams.get('since') || 0);
      for (const e of store.since(since, 2000).filter((x) => !W.session || x.session === W.session)) res.write(`event: ev\ndata: ${JSON.stringify(e)}\n\n`);
      const onEv = (e) => res.write(`event: ev\ndata: ${JSON.stringify(e)}\n\n`);
      store.listeners.add(onEv);
      const t = setInterval(() => res.write(`event: status\ndata: ${JSON.stringify(status())}\n\n`), 2000);
      req.on('close', () => { clearInterval(t); store.listeners.delete(onEv); });
      return;
    }
    if (p === '/api/journal') return send(res, 200, store.readJournal() || '');
    if (p === '/api/sessions') return send(res, 200, store.listSessions());
    if (p.startsWith('/api/sessions/')) return send(res, 200, store.readSession(p.split('/').pop()));
    if (p === '/api/control' && req.method === 'POST') {
      const b = await readBody(req);
      return send(res, 200, { result: await control(String(b.action || ''), b) });
    }
    if (p === '/api/mind/poll') return send(res, 200, mindPoll());
    if (p === '/api/mind/context') return send(res, 200, await mindContext());
    if (p === '/api/mind/decision' && req.method === 'POST') return send(res, 200, await mindDecision(await readBody(req)));
    if (p === '/api/mind/status' && req.method === 'POST') { mindStatus(await readBody(req)); return send(res, 200, { ok: true }); }
    if (p === '/api/mind/prompt' && req.method === 'POST') {
      const b = await readBody(req);
      store.savePrompt(String(b.hash || ''), String(b.text || ''));
      return send(res, 200, { ok: true });
    }
    return send(res, 404, { error: 'not found' });
  } catch (e) {
    log('request error', p, e.message);
    if (!res.headersSent) send(res, e.status || 500, { error: e.message });
  }
});

httpServer.on('upgrade', (req, socket, head) => {
  let ok = false;
  try { ok = req.url.startsWith('/viewer') && authed(req); } catch { /* malformed */ }
  if (!ok) { socket.destroy(); return; }
  proxyUpgrade(req, socket, head, CFG.viewerPort);
});

httpServer.listen(CFG.port, CFG.bind, () => log(`hub listening on http://${CFG.bind}:${CFG.port} (phase ${W.phase})`));

// A bug in one request must not take the world's supervisor down with it.
process.on('unhandledRejection', (e) => log('unhandled rejection', e?.stack || e));
process.on('uncaughtException', (e) => log('uncaught exception', e?.stack || e));

for (const sig of ['SIGTERM', 'SIGINT']) process.on(sig, () => { log(`${sig}: hub exiting (world processes keep running)`); process.exit(0); });

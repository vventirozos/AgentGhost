// Eva's World monitor. Read-only apart from the four world controls.
// Live data arrives over one EventSource (/api/stream): `ev` for every new
// record, `status` every 2 s. Text from the world is inserted with
// textContent only: players and models write it, so it is never HTML.
'use strict';

const $ = (id) => document.getElementById(id);
const BODY_KINDS = new Set(['body_response', 'command_result', 'body_lost', 'body_back', 'body_restart', 'bot_output', 'chat', 'join', 'leave', 'death', 'goal_dropped',
  'context_overflow', 'model_error', 'error', 'control', 'paused', 'resumed', 'session_end', 'mind_status', 'directive', 'intervention', 'body_process_exit']);
let lastSeq = 0;
let es = null;
let phase = null;
let viewerOn = false;
let st = null;
const notes = { mind: '', body: '', journal: 'written by Eva, for Eva' };

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
const hhmm = (ts) => new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

function gameClock(t) {
  t = ((t % 24000) + 24000) % 24000;
  const h = Math.floor(t / 1000 + 6) % 24, m = Math.floor((t % 1000) / 1000 * 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

async function api(path, opts = {}) {
  const r = await fetch(path, { credentials: 'same-origin', ...opts, headers: { 'content-type': 'application/json', ...(opts.headers || {}) } });
  if (r.status === 401) { showLogin(); throw new Error('key required'); }
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

function showLogin() { $('app').hidden = true; $('login').hidden = false; if (es) { es.close(); es = null; } }

$('login-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  $('login-err').textContent = '';
  const r = await fetch('/api/login', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ key: $('key').value.trim() }) });
  if (!r.ok) { $('login-err').textContent = 'Wrong key.'; return; }
  start();
});

// ── rendering ──────────────────────────────────────────────────────────
function bodyText(e) {
  switch (e.kind) {
    case 'body_response': return `${e.to && e.to !== 'system' ? `→${e.to}: ` : ''}${e.text}`;
    case 'command_result': return `  ↳ ${e.command}: ${e.text}`;
    case 'bot_output': return `says: ${e.text}`;
    case 'chat': return `<${e.player}> ${e.text}`;
    case 'join': return `${e.player} joined`;
    case 'leave': return `${e.player} left`;
    case 'death': return `☠ ${e.text}`;
    case 'body_lost': return '⚠ the body dropped out of the world';
    case 'body_back': return `✓ ${e.text.replace(/^your body/, 'the body')}`;
    case 'body_restart': return `↻ ${e.text}`;
    case 'body_process_exit': return `↻ Mindcraft restarted the body process (${e.code != null ? `exit code ${e.code}` : `signal ${e.signal}`})`;
    case 'goal_dropped': return 'gave up on its goal (no commands for 3 turns)';
    case 'context_overflow': return 'context overflow: oldest history trimmed';
    case 'model_error': return 'model error: no reply';
    case 'directive': return `Eva ${e.summary}`;
    case 'mind_status': return `Eva's mind: ${e.status}${e.detail ? ` (${e.detail})` : ''}`;
    default: return e.text || e.kind;
  }
}

// The body's words arrive three times (model log, chat output, server chat
// line). Show the model log (it has the command); hide its chat twins.
function hiddenTwin(e) {
  return (e.kind === 'chat' && e.from_body) || (e.kind === 'bot_output' && /^\s*!/.test(e.text || ''));
}

function bodyItem(e) {
  const d = el('div', `item k-${e.kind}`);
  d.append(el('span', 't', hhmm(e.ts)));
  const text = bodyText(e);
  const i = e.kind === 'body_response' && e.command ? text.indexOf(e.command) : -1;
  if (i >= 0) d.append(document.createTextNode(text.slice(0, i)), el('span', 'cmd', e.command), document.createTextNode(text.slice(i + e.command.length)));
  else d.append(document.createTextNode(text));
  return d;
}

// Her prose without the trailing decision JSON (shown separately): drop a
// trailing ``` fence, then cut at the first "{" from which the REST parses as
// JSON — braces in her prose ("mine {oak logs}") stay.
function stripDecision(reply) {
  let t = reply.replace(/<\/?tool_call>/g, '').trim();
  const fence = /```(?:json)?\s*([\s\S]*?)\s*```\s*$/.exec(t);
  if (fence) {
    try { JSON.parse(fence[1]); return t.slice(0, fence.index).trim(); } catch { /* not the decision */ }
  }
  for (let i = t.indexOf('{'); i !== -1; i = t.indexOf('{', i + 1)) {
    try { JSON.parse(t.slice(i)); return t.slice(0, i).trim(); } catch { /* keep looking */ }
  }
  return t;
}

function decisionCard(e) {
  const d = el('div', 'decision');
  const head = el('div', 'why');
  head.append(el('b', null, hhmm(e.ts)), el('span', null, `woke because: ${(e.reasons && e.reasons.length ? e.reasons.join('; ') : 'her chosen time')}`));
  if (e.timings?.elapsed_s) head.append(el('span', null, `thought ${Math.round(e.timings.elapsed_s)} s`));
  d.append(head);
  // she sometimes wraps the JSON in the agent's <tool_call> tags out of habit
  const prose = stripDecision(e.reply || '');
  if (prose) d.append(el('div', 'reply', prose));
  if (e.parse_error) d.append(el('div', 'why', `⚠ ${e.parse_error}`));
  const ul = el('ul');
  for (const a of e.applied || []) ul.append(el('li', null, a));
  ul.append(el('li', 'muted', `look again in ${e.wake_in_minutes} min`));
  d.append(ul);
  if (e.reasoning) { const det = el('details'); det.append(el('summary', null, `reasoning (${e.reasoning.length.toLocaleString()} chars)`), el('pre', null, e.reasoning)); d.append(det); }
  if (e.diary) { const det = el('details'); det.append(el('summary', null, 'what she saw'), el('pre', null, e.diary)); d.append(det); }
  return d;
}

function addEvent(e, live = true) {
  lastSeq = Math.max(lastSeq, e.seq || 0);
  if (e.kind === 'eva_decision') {
    $('eva-list').querySelector('.empty')?.remove();
    $('eva-list').prepend(decisionCard(e));
    if (live && (e.applied || []).some((a) => a.includes('journal'))) loadJournal();
  } else if (BODY_KINDS.has(e.kind) && !hiddenTwin(e)) {
    const list = $('body-list');
    list.prepend(bodyItem(e));
    while (list.children.length > 400) list.lastChild.remove();
  }
}

function tile(k, v, opts = {}) {
  const t = el('div', `tile${opts.wide ? ' wide' : ''}`);
  t.append(el('div', 'k', k));
  const val = el('div', 'v', v);
  if (opts.title) val.title = opts.title;
  t.append(val);
  if (opts.bar) {
    const b = el('div', `bar ${opts.bar}`); const i = el('i');
    i.style.width = `${Math.max(0, Math.min(100, opts.pct))}%`;
    b.append(i); t.append(b);
  }
  return t;
}

function renderStatus(s) {
  st = s;
  phase = s.phase;
  const chip = $('phase');
  chip.textContent = s.phase + (s.controlBusy ? '…' : '');
  chip.className = `chip ${s.phase}`;
  $('orb').className = `orb ${s.phase}`;
  $('btn-start').disabled = s.controlBusy || s.phase !== 'stopped';
  $('btn-pause').disabled = s.controlBusy || s.phase !== 'running';
  $('btn-resume').disabled = s.controlBusy || s.phase !== 'paused';
  $('btn-stop').disabled = s.controlBusy || !['running', 'paused'].includes(s.phase);
  const b = $('banner');
  if (s.error && s.phase === 'stopped') { b.hidden = false; b.textContent = `Last start failed: ${s.error}`; } else b.hidden = true;

  const g = s.state?.gameplay;
  $('clock').textContent = g ? `game ${gameClock(g.timeOfDay)} · ${g.timeLabel?.toLowerCase() || ''}` : '';
  const wantViewer = ['running', 'paused'].includes(s.phase) && s.procs.mindcraft && !!g;
  if (wantViewer !== viewerOn) {
    viewerOn = wantViewer;
    $('viewer').src = wantViewer ? '/viewer/' : 'about:blank';
    $('viewer-off').hidden = wantViewer;
    $('find-eva').hidden = !wantViewer;
  }
  if (s.phase === 'running' && s.wakeAt) {
    const mins = Math.max(0, Math.round((s.wakeAt - s.now) / 60000));
    notes.mind = s.mind.status === 'thinking' ? 'thinking now…' : (s.reasons.length ? `wants to look: ${s.reasons[0]}` : `next look in ~${mins} min`);
  } else notes.mind = '';
  notes.body = s.gemma.avg_ms ? `~${(s.gemma.avg_ms / 1000).toFixed(1)} s per move${s.gemma.waiting ? ' · thinking' : ''}` : '';
  $('feed-note').textContent = notes[feed];

  const box = $('vitals');
  box.replaceChildren();
  if (g) {
    const p = g.position;
    box.append(
      tile('Health', `${g.health} / 20`, { bar: 'hp', pct: g.health * 5 }),
      tile('Hunger', `${g.hunger} / 20`, { bar: 'food', pct: g.hunger * 5 }),
      tile('Position', `${Math.round(p.x)}, ${Math.round(p.y)}, ${Math.round(p.z)}`),
      tile('Biome · weather', `${g.biome || '?'} · ${String(g.weather || '?').toLowerCase()}`),
      tile('Doing', (s.state.action?.current || '?').replace(/^action:/, '')),
    );
    const inv = Object.entries(s.state.inventory?.counts || {}).sort((a, c) => c[1] - a[1]).map(([k, n]) => `${k.replace(/_/g, ' ')} ×${n}`).join(', ');
    box.append(tile('Inventory', inv || 'empty', { wide: true, title: inv }));
  }
  if (s.currentGoal) box.append(tile('Goal', s.currentGoal, { wide: true, title: s.currentGoal }));
  box.append(tile("Eva's mind", `${s.mind.status}`, { title: s.mind.detail || '' }));
  if (s.memFreePct != null) box.append(tile('nova memory free', `${s.memFreePct}%`));
}

// journal.md is "## <date> (session …)" headings, oldest first: shown as
// entries, newest on top like the other two feeds.
function journalEntries(md) {
  return md.split(/^## /m).map((x) => x.trim()).filter(Boolean).map((x) => {
    const nl = x.indexOf('\n');
    return nl < 0 ? { head: x, text: '' } : { head: x.slice(0, nl).trim(), text: x.slice(nl + 1).trim() };
  }).reverse();
}

async function loadJournal() {
  let md;
  try { md = await api('/api/journal'); } catch { return; /* login shown */ }
  const box = $('journal');
  box.replaceChildren();
  for (const j of journalEntries(md || '')) {
    const d = el('div', 'entry');
    d.append(el('div', 'when', j.head.replace(/\s*\(session [^)]*\)\s*$/, '')), el('div', 'words', j.text));
    box.append(d);
  }
  if (!box.children.length) box.append(el('div', 'empty', 'Eva has not written anything yet.'));
}

// ── one text panel: Mind / Body / Journal ──────────────────────────────
let feed = 'mind';
try { feed = localStorage.getItem('eva-feed') || 'mind'; } catch { /* storage blocked */ }
function showFeed(f) {
  if (!['mind', 'body', 'journal'].includes(f)) f = 'mind';
  feed = f;
  $('feed-pick').value = f;
  document.querySelector('.panel.text').dataset.feed = f;
  for (const x of document.querySelectorAll('.panel.text .feed')) x.hidden = x.dataset.for !== f;
  $('feed-note').textContent = notes[f];
  try { localStorage.setItem('eva-feed', f); } catch { /* storage blocked */ }
}
$('feed-pick').addEventListener('change', (ev) => showFeed(ev.target.value));
showFeed(feed);

// ── find Eva: re-centre the viewer camera on the body ──────────────────
// The viewer (same origin through the hub) has window.__evaRecenter from
// patch_viewer.cjs; without it, reloading the viewer re-centres too.
$('find-eva').addEventListener('click', () => {
  let ok = false;
  try { ok = !!$('viewer').contentWindow.__evaRecenter?.(); } catch { /* not loaded yet */ }
  if (!ok) $('viewer').src = '/viewer/?t=' + Date.now();
});

// ── controls ───────────────────────────────────────────────────────────
for (const btn of document.querySelectorAll('.controls button')) {
  btn.addEventListener('click', async () => {
    const action = btn.dataset.action;
    if (action === 'stop' && !confirm('Stop the world? It is saved and continues next time.')) return;
    btn.disabled = true;
    try { await api('/api/control', { method: 'POST', body: JSON.stringify({ action }) }); }
    catch (e) { alert(`${action} failed: ${e.message}`); }
  });
}

// ── tabs + history ─────────────────────────────────────────────────────
for (const t of document.querySelectorAll('.tabs button')) {
  t.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.toggle('on', b === t));
    $('tab-live').hidden = t.dataset.tab !== 'live';
    $('tab-history').hidden = t.dataset.tab !== 'history';
    if (t.dataset.tab === 'history') loadSessions();
  });
}

async function loadSessions() {
  const list = $('sessions');
  list.replaceChildren();
  for (const s of await api('/api/sessions')) {
    const li = el('li'); const b = el('button', null, `${s.id} · ${(s.bytes / 1024).toFixed(0)} KB`);
    b.addEventListener('click', () => showSession(s.id));
    li.append(b); list.append(li);
  }
  if (!list.children.length) list.append(el('li', 'muted', 'No sessions yet.'));
}

async function showSession(id) {
  $('hist-title').textContent = `Session ${id}`;
  const out = $('hist-list');
  out.replaceChildren(el('div', 'muted', 'loading…'));
  const events = await api(`/api/sessions/${encodeURIComponent(id)}`);
  out.replaceChildren();
  for (const e of events) {
    if (e.kind === 'eva_decision') out.append(decisionCard(e));
    else if (BODY_KINDS.has(e.kind) && !hiddenTwin(e)) out.append(bodyItem(e));
  }
}

// ── live stream ────────────────────────────────────────────────────────
function connect() {
  if (es) es.close();
  es = new EventSource(`/api/stream?since=${lastSeq}`);
  es.addEventListener('ev', (m) => addEvent(JSON.parse(m.data)));
  es.addEventListener('status', (m) => renderStatus(JSON.parse(m.data)));
  es.onerror = () => { es.close(); es = null; setTimeout(async () => { try { await api('/api/status'); connect(); } catch { /* login */ } }, 3000); };
}

async function start() {
  try { renderStatus(await api('/api/status')); } catch { return; }
  $('login').hidden = true; $('app').hidden = false;
  loadJournal();
  connect();
}
start();

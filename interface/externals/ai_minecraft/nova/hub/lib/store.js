// Event store: one JSONL file per session, plus Eva's journal.
//
// Append-only: the research value is in the record, so nothing here rewrites
// or deletes a line. Every event gets a monotonically increasing `seq` so the
// monitor can ask for "everything after N" after a reconnect.
import fs from 'node:fs';
import path from 'node:path';

const RING = 2000;

export class Store {
  constructor(dataDir) {
    this.dataDir = dataDir;
    this.sessionsDir = path.join(dataDir, 'sessions');
    this.journalPath = path.join(dataDir, 'eva', 'journal.md');
    this.promptsDir = path.join(dataDir, 'eva', 'prompts');
    for (const d of [this.sessionsDir, path.dirname(this.journalPath), this.promptsDir]) fs.mkdirSync(d, { recursive: true });
    // Seeded from the clock so seq keeps rising across hub restarts: a
    // monitor that reconnects with since=<old seq> must not miss new events.
    this.seq = Date.now();
    this.ring = [];
    this.session = null;
    this.listeners = new Set();
  }

  static newSessionId(now = new Date()) {
    const p = (n) => String(n).padStart(2, '0');
    return `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())}_${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
  }

  startSession(id = Store.newSessionId()) {
    this.session = id;
    this.ring = [];   // the first diary of a session must not show the last one's events
    fs.mkdirSync(path.join(this.sessionsDir, id), { recursive: true });
    return id;
  }

  // Resume writing into an existing session (hub restarted mid-session).
  resumeSession(id) {
    this.session = id;
    this.ring = [];
    const file = this.sessionFile(id);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    if (fs.existsSync(file)) {
      const lines = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
      for (const l of lines.slice(-RING)) {
        try { const e = JSON.parse(l); this.ring.push(e); this.seq = Math.max(this.seq, e.seq || 0); } catch { /* torn tail line */ }
      }
    }
  }

  endSession() {
    this.session = null;
  }

  sessionFile(id = this.session) {
    return path.join(this.sessionsDir, id, 'events.jsonl');
  }

  append(kind, fields = {}) {
    const e = { seq: ++this.seq, kind, ...fields, ts: fields.ts ?? Date.now(), session: this.session };
    this.ring.push(e);
    if (this.ring.length > RING) this.ring.shift();
    if (this.session) fs.appendFileSync(this.sessionFile(), JSON.stringify(e) + '\n');
    for (const fn of this.listeners) { try { fn(e); } catch { /* a dead SSE client */ } }
    return e;
  }

  since(seq, limit = 500) {
    return this.ring.filter((e) => e.seq > seq).slice(0, limit);
  }

  eventsBetween(fromTs, toTs = Infinity) {
    return this.ring.filter((e) => e.session === this.session && e.ts > fromTs && e.ts <= toTs);
  }

  listSessions() {
    if (!fs.existsSync(this.sessionsDir)) return [];
    return fs.readdirSync(this.sessionsDir).filter((d) => fs.existsSync(path.join(this.sessionsDir, d, 'events.jsonl'))).sort().reverse()
      .map((id) => ({ id, bytes: fs.statSync(path.join(this.sessionsDir, id, 'events.jsonl')).size }));
  }

  readSession(id) {
    if (!/^[\w-]+$/.test(id)) throw new Error('bad session id');
    const f = path.join(this.sessionsDir, id, 'events.jsonl');
    return fs.readFileSync(f, 'utf8').split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  }

  appendJournal(text, now = new Date()) {
    const p = (n) => String(n).padStart(2, '0');
    const stamp = `${now.getFullYear()}-${p(now.getMonth() + 1)}-${p(now.getDate())} ${p(now.getHours())}:${p(now.getMinutes())}`; // local time
    fs.appendFileSync(this.journalPath, `\n## ${stamp}${this.session ? ` (session ${this.session})` : ''}\n${text.trim()}\n`);
  }

  readJournal() {
    return fs.existsSync(this.journalPath) ? fs.readFileSync(this.journalPath, 'utf8') : '';
  }

  // The tail she is shown; the whole file stays on disk.
  journalTail(maxChars = 6000) {
    const j = this.readJournal();
    return j.length > maxChars ? '…' + j.slice(-maxChars) : j;
  }

  savePrompt(hash, text) {
    if (!/^[0-9a-f]{8,64}$/.test(hash)) throw new Error('bad prompt hash');
    const f = path.join(this.promptsDir, `${hash}.txt`);
    if (!fs.existsSync(f)) fs.writeFileSync(f, text);
    return f;
  }
}

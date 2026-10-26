import test from 'node:test';
import assert from 'node:assert/strict';
import net from 'node:net';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { encodePacket, decodePackets, rcon } from '../lib/rcon.js';
import { Store } from '../lib/store.js';

test('rcon packets round-trip, split across chunks', () => {
  const a = encodePacket(7, 2, 'time query day');
  const b = encodePacket(8, 0, 'The time is 3');
  const all = Buffer.concat([a, b]);
  let [p, rest] = decodePackets(all.slice(0, a.length + 5));
  assert.deepEqual(p, [{ id: 7, type: 2, body: 'time query day' }]);
  [p, rest] = decodePackets(Buffer.concat([rest, all.slice(a.length + 5)]));
  assert.deepEqual(p, [{ id: 8, type: 0, body: 'The time is 3' }]);
  assert.equal(rest.length, 0);
});

// A fake server that speaks the protocol the way vanilla does.
function fakeServer(password, reply) {
  return new Promise((resolve) => {
    const srv = net.createServer((sock) => {
      let buf = Buffer.alloc(0);
      sock.on('data', (c) => {
        let pk; [pk, buf] = decodePackets(Buffer.concat([buf, c]));
        for (const p of pk) {
          if (p.type === 3) sock.write(encodePacket(p.body === password ? p.id : -1, 2, ''));
          else if (p.type === 2) sock.write(encodePacket(p.id, 0, reply(p.body)));
        }
      });
    });
    srv.listen(0, '127.0.0.1', () => resolve(srv));
  });
}

test('rcon: command answered after auth; wrong password rejected', async () => {
  const srv = await fakeServer('pw', (cmd) => `ran ${cmd}`);
  const port = srv.address().port;
  assert.equal(await rcon('list', { port, password: 'pw' }), 'ran list');
  await assert.rejects(rcon('list', { port, password: 'nope' }), /auth failed/);
  srv.close();
});

test('store: append-only session file, since(), journal, prompts', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'evaw-'));
  const s = new Store(dir);
  const id = s.startSession('2026-10-25_120000');
  const e1 = s.append('chat', { text: 'a' });
  const e2 = s.append('death', { text: 'b' });
  assert.ok(e2.seq > e1.seq && e1.seq > Date.now() - 60000, 'seq is clock-seeded and rising');
  assert.equal(e1.session, id);
  assert.deepEqual(s.since(e1.seq).map((e) => e.text), ['b']);
  assert.equal(s.readSession(id).length, 2);
  assert.throws(() => s.readSession('../etc'), /bad session id/);
  s.appendJournal('first entry', new Date(2026, 9, 25, 23, 56));
  assert.match(s.readJournal(), /## 2026-10-25 23:56 \(session 2026-10-25_120000\)\nfirst entry/);
  s.savePrompt('abcdef0123456789', 'PROMPT');
  assert.throws(() => s.savePrompt('../x', 'y'), /bad prompt hash/);

  const s2 = new Store(dir);  // a hub restart resumes the session
  s2.resumeSession(id);
  const e3 = s2.append('chat', { text: 'c' });
  assert.ok(e3.seq > e2.seq);
  assert.equal(s2.readSession(id).length, 3);
  fs.rmSync(dir, { recursive: true });
});

test('store: a new session starts with an empty diary, not the last session\'s events', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'evaw-'));
  const s = new Store(dir);
  s.startSession('2026-01-01_000000');
  s.append('death', { text: 'old death', ts: 100 });
  s.endSession();
  s.append('intervention', { text: 'between sessions', ts: 150 });
  s.startSession('2026-01-02_000000');
  s.append('chat', { text: 'new', ts: 200 });
  assert.deepEqual(s.eventsBetween(0).map((e) => e.text), ['new']);
  // nothing is written to a file while no session is open
  assert.ok(!fs.readFileSync(s.sessionFile('2026-01-01_000000'), 'utf8').includes('between sessions'));
});

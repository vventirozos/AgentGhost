import test from 'node:test';
import assert from 'node:assert/strict';
import { gameClock, buildDiary, dedupe, eventLine } from '../lib/diary.js';

test('game clock: tick 0 is 06:00, 18000 is midnight', () => {
  assert.equal(gameClock(0), '06:00 (dawn)');
  assert.equal(gameClock(6000), '12:00 (day)');
  assert.equal(gameClock(12000), '18:00 (dusk)');
  assert.equal(gameClock(18000), '00:00 (night)');
  assert.equal(gameClock(3781), '09:46 (day)');
  assert.equal(gameClock(24000 + 6000), '12:00 (day)');
});

// state shape copied from a live /api/status (2026-10-25)
const STATE = {
  gameplay: { position: { x: 8.5, y: 72, z: 9.5 }, dimension: 'overworld', health: 20, hunger: 20, biome: 'plains', weather: 'Clear', timeOfDay: 3781 },
  action: { current: 'action:searchForBlock', kind: 'acting' },
  inventory: { counts: { oak_log: 3, dirt: 1 }, equipment: { mainHand: null } },
  nearby: { humanPlayers: [], entityTypes: ['sheep', 'sheep', 'cow'] },
  surroundings: { below: 'grass_block', firstBlockAboveHead: 'none' },
};

test('diary: header facts come from the state', () => {
  const d = buildDiary({ state: STATE, day: 0, events: [], lastLookTs: null, now: Date.now(), reasons: ['a new session began'], currentGoal: null });
  assert.match(d, /WORLD TIME: day 0, 09:46 \(day\) — this is your first look/);
  assert.match(d, /health 20\/20, hunger 20\/20, at x=9 y=72 z=10 \(plains, overworld\)/);
  assert.match(d, /INVENTORY: oak_log x3, dirt x1/);
  assert.match(d, /creatures sheep, cow/);
  assert.match(d, /nothing was recorded/);
  assert.match(d, /WHY YOU WOKE: a new session began/);
});

test('diary: no state means the body is absent, not a crash', () => {
  const d = buildDiary({ state: null, day: null, events: [], lastLookTs: 1, now: 60001, reasons: [] });
  assert.match(d, /BODY: not in the world right now/);
  assert.match(d, /real minutes since you last looked: 1/);
});

test('diary: caps event lines and says how many were omitted', () => {
  const events = Array.from({ length: 75 }, (_, i) => ({ kind: 'body_response', ts: 1e12 + i, to: 'system', text: `step ${i}` }));
  const d = buildDiary({ state: STATE, day: 1, events, lastLookTs: 1, now: 2e12, reasons: [] });
  assert.match(d, /\(15 earlier lines omitted\)/);
  assert.ok(d.includes('step 74') && !d.includes('step 14\n'));
});

test('dedupe: a chat echo of a response is dropped, a narration is kept', () => {
  const ev = [
    { kind: 'body_response', text: '!searchForBlock("oak", 64)' },
    { kind: 'bot_output', text: ' !searchForBlock("oak", 64)' },
    { kind: 'bot_output', text: 'Picking up item!' },
  ];
  assert.deepEqual(dedupe(ev).map((e) => e.text), ['!searchForBlock("oak", 64)', 'Picking up item!']);
});

test('eventLine: body chat lines are skipped, others phrased for Eva', () => {
  assert.equal(eventLine({ kind: 'chat', from_body: true, ts: 0, player: 'Eva', text: 'x' }), null);
  assert.match(eventLine({ kind: 'death', is_body: true, ts: 0, text: 'Eva was slain by Zombie' }), /YOUR BODY DIED/);
  assert.match(eventLine({ kind: 'command_result', ts: 0, command: '!searchForBlock', text: 'Invalid block type: oak.' }), /!searchForBlock result: Invalid block type/);
  assert.equal(eventLine({ kind: 'snapshot', ts: 0 }), null);
});

test('Eva reads that Mindcraft restarted her body, and what that cost it', () => {
  const line = eventLine({ kind: 'body_process_exit', code: 1, text: 'Agent process exited with code 1 and signal null', ts: Date.now() });
  assert.match(line, /your body's process stopped \(exit code 1\) and Mindcraft restarted it/);
  assert.match(line, /kept its saved memory but lost what it was doing/);
  assert.match(eventLine({ kind: 'body_process_exit', code: null, signal: 'SIGTERM', ts: Date.now() }), /stopped \(signal SIGTERM\)/);
});

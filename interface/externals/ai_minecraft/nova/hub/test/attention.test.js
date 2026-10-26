import test from 'node:test';
import assert from 'node:assert/strict';
import { Attention, MIN_GAP_MS, STUCK_MS, FAIL_RUN, isFailedResult } from '../lib/attention.js';

const pos = (x) => ({ gameplay: { position: { x, y: 64, z: 0 } }, action: { kind: 'acting' } });

test('her chosen time wakes her with no reasons', () => {
  const a = new Attention();
  assert.equal(a.due(1000, 0, 2000), false);
  assert.equal(a.due(2000, 0, 2000), true);
});

test('a death wakes her early, but not inside the minimum gap', () => {
  const a = new Attention();
  a.onEvent({ kind: 'death', is_body: true, ts: 10 });
  assert.deepEqual(a.pending(), ['your body died']);
  const last = 1_000_000;
  assert.equal(a.due(last + MIN_GAP_MS - 1, last, last + 3600e3), false);
  assert.equal(a.due(last + MIN_GAP_MS, last, last + 3600e3), true);
  a.clear();
  assert.equal(a.due(last + MIN_GAP_MS, last, last + 3600e3), false);
});

test('another player speaking or joining raises attention; the body talking does not', () => {
  const a = new Attention();
  a.onEvent({ kind: 'chat', from_body: true, player: 'Eva', ts: 1 });
  assert.deepEqual(a.pending(), []);
  a.onEvent({ kind: 'chat', from_body: false, player: 'Steve', ts: 2 });
  a.onEvent({ kind: 'join', is_body: false, player: 'Alex', ts: 3 });
  assert.deepEqual(a.pending(), ['Steve spoke in the world', 'Alex joined the world']);
});

test('the same command four times in a row is "repeating"', () => {
  const a = new Attention();
  for (let i = 0; i < 3; i++) a.onEvent({ kind: 'body_response', command: '!collectBlocks("oak_log", 10)', ts: i });
  assert.deepEqual(a.pending(), []);
  a.onEvent({ kind: 'body_response', command: '!collectBlocks("oak_log", 10)', ts: 4 });
  assert.deepEqual(a.pending(), ['your body is repeating the same thing']);
});

test('busy without moving for 5 minutes is "stuck"; moving resets it', () => {
  const a = new Attention();
  a.onState(pos(0), 0);
  a.onState(pos(1), STUCK_MS - 1);
  assert.deepEqual(a.pending(), []);
  a.onState(pos(1), STUCK_MS + 1);
  assert.equal(a.pending().length, 1);
  const b = new Attention();
  b.onState(pos(0), 0);
  b.onState(pos(10), STUCK_MS / 2);
  b.onState(pos(10), STUCK_MS + 1);
  assert.deepEqual(b.pending(), []);
});

test('the body going from working to stopped raises attention', () => {
  const a = new Attention();
  a.onState({ action: { kind: 'thinking' } }, 1);
  a.onState({ action: { kind: 'stopped' } }, 2);
  assert.deepEqual(a.pending(), ['your body stopped and is waiting']);
});

test('after a look, only what she saw is cleared; a reason raised while she thought stays', () => {
  const a = new Attention();
  a.raise('your body died', 1000);
  a.raise('Vasilis spoke in the world', 5000);
  a.clearUpTo(3000);
  assert.deepEqual(a.pending(), ['Vasilis spoke in the world']);
});

test('the same alert again while she thinks still wakes her after the look', () => {
  const a = new Attention();
  a.raise('your body died', 1000);
  a.raise('your body died', 5000);     // died again during a look that began at 3000
  a.clearUpTo(3000);
  assert.deepEqual(a.pending(), ['your body died']);
});

test('standing still on purpose (!stay) is not stuck; the clock restarts when the stay ends', () => {
  const stay = (x) => ({ gameplay: { position: { x, y: 64, z: 0 } }, action: { kind: 'acting', current: 'action:stay' } });
  const a = new Attention();
  a.onState(stay(1), 0);
  a.onState(stay(1), STUCK_MS + 1);
  a.onState(stay(1), 3 * STUCK_MS);
  assert.deepEqual(a.pending(), []);
  a.onState(pos(1), 3 * STUCK_MS + 10);
  a.onState(pos(1), 4 * STUCK_MS);
  assert.deepEqual(a.pending(), []);
  a.onState(pos(1), 4 * STUCK_MS + 11);
  assert.deepEqual(a.pending(), ['your body has not moved for 5 minutes while busy']);
});

test('standing idle does not count toward "stuck": the clock starts when she gets busy', () => {
  // 10-26: 32 min "stopped" in one place, then a new goal → the alert 41 s in
  const at = (kind) => ({ gameplay: { position: { x: -224, y: 64, z: -90 } }, action: { kind } });
  const a = new Attention();
  a.onState(at('thinking'), 0);
  a.onState(at('stopped'), 1000);
  a.onState(at('stopped'), 32 * 60e3);
  a.onState(at('thinking'), 32 * 60e3 + 23e3);
  a.onState(at('acting'), 32 * 60e3 + 41e3);
  assert.deepEqual(a.pending(), ['your body stopped and is waiting']);
  a.onState(at('acting'), 32 * 60e3 + STUCK_MS + 1000);   // 5 min busy since 32:00, not moving
  assert.deepEqual(a.pending(), ['your body stopped and is waiting', 'your body has not moved for 5 minutes while busy']);
});

test('a stuck body that answers a message now and then is still stuck', () => {
  // review 10-26: one non-busy sample must not restart the clock
  const at = (kind) => ({ gameplay: { position: { x: 5, y: 64, z: 5 } }, action: { kind } });
  const a = new Attention();
  for (let t = 0; t <= STUCK_MS + 30e3; t += 1000) {
    a.onState(at(t % 30e3 < 5e3 && t > 0 ? 'chatting' : 'acting'), t);   // 5 s of chat every 30 s
  }
  assert.ok(a.pending().includes('your body has not moved for 5 minutes while busy'));
});

// verbatim results from the 10-26 sessions
const FAILS = ['Invalid block type: oak.', 'Command !goToCoordinates was given 3 args, but requires 4 args.',
  'Path not found, but attempting to navigate anyway using destructive movements. Pathfinding stopped: Cannot break stone with current tools.',
  'Unable to reach -135, 64, -218, you are 2 blocks away.', 'Could not find any oak_log in 64 blocks.', 'move is not a command.',
  'Crafting wooden_pickaxe did not go through: the server undid it, so nothing was used up.'];
const OKS = ['Found non-destructive path. Moved away from (14, 62, 16) to (9, 61, 17).',
  'Path not found, but attempting to navigate anyway using destructive movements. You have reached at -135, 64, -228.',
  'Found crafting_table at (-136, 64, -179). Navigating... Found non-destructive path. You have reached at -136, 64, -179. Successfully crafted stone_pickaxe, you now have 1 stone_pickaxe.',
  'undefined', 'Equipped stone_pickaxe.'];

test('failed command results are told apart from successes (incl. "Path not found, but … reached")', () => {
  for (const t of FAILS) assert.equal(isFailedResult(t), true, t);
  for (const t of OKS) assert.equal(isFailedResult(t), false, t);
});

test('a run of different failing commands wakes her once; a success starts a new run', () => {
  const REASON = `your body's last ${FAIL_RUN} commands all failed`;
  const a = new Attention();
  const res = (text, ts) => a.onEvent({ kind: 'command_result', command: '!x', text, ts });
  // the 10-26 loop: oak, acacia, moveAway (ok), … never 4 identical in a row
  let ts = 0;
  for (let i = 0; i < FAIL_RUN - 1; i++) res(FAILS[i % 2], ++ts);
  assert.deepEqual(a.pending(), []);
  res(OKS[0], ++ts);                                    // a success resets the run
  for (let i = 0; i < FAIL_RUN - 1; i++) res(FAILS[i % 3], ++ts);
  assert.deepEqual(a.pending(), []);
  res(FAILS[5], ++ts);
  assert.deepEqual(a.pending(), [REASON]);
  a.clear();
  for (let i = 0; i < 3 * FAIL_RUN; i++) res(FAILS[0], ++ts);   // the same run goes on: no second wake
  assert.deepEqual(a.pending(), []);
  res(OKS[1], ++ts);
  for (let i = 0; i < FAIL_RUN; i++) res(FAILS[1], ++ts);       // a new run after a success: wakes again
  assert.deepEqual(a.pending(), [REASON]);
});

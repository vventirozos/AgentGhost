import test from 'node:test';
import assert from 'node:assert/strict';
import { planDirectives, clampWake, tellrawCommand, WAKE_DEFAULT } from '../lib/directives.js';

test('a goal becomes a !goal command with quotes made safe', () => {
  const { actions } = planDirectives({ goal: 'Build a "tiny" hut\nby the river' }, 'Eva');
  assert.equal(actions.length, 1);
  assert.equal(actions[0].payload, `!goal("Build a 'tiny' hut by the river")`);
  assert.equal(actions[0].goal, "Build a 'tiny' hut by the river");
});

test('stop comes before a new goal, tell_body never runs as a command', () => {
  const { actions } = planDirectives({ goal: 'explore', stop: true, tell_body: '!stop everything' }, 'Eva');
  assert.deepEqual(actions.map((a) => a.payload), ['!stop', '!goal("explore")', 'stop everything']);
});

test('tell_body never carries a command anywhere in the text', () => {
  const t = planDirectives({ tell_body: 'Stop. No more !moveAway. Walk to x=30! Then !goToCoordinates(1,2,3,1)' }, 'Eva').actions[0];
  assert.equal(t.payload, 'Stop. No more moveAway. Walk to x=30! Then goToCoordinates(1,2,3,1)');
  assert.ok(!/!\w/.test(t.payload));
});

test('say goes to RCON as escaped tellraw JSON, journal to the journal', () => {
  const { actions } = planDirectives({ say: 'hi "you"', journal: 'day one' }, 'Eva');
  assert.equal(actions[0].channel, 'rcon');
  assert.equal(actions[0].payload, 'tellraw @a [{"text":"<Eva> "},{"text":"hi \\"you\\""}]');
  assert.deepEqual([actions[1].channel, actions[1].payload], ['journal', 'day one']);
  assert.equal(tellrawCommand('Eva', 'x'), 'tellraw @a [{"text":"<Eva> "},{"text":"x"}]');
});

test('null, "null", blanks and junk do nothing', () => {
  for (const d of [null, {}, { goal: null, say: '  ', tell_body: 'null', stop: 'yes' }]) {
    assert.deepEqual(planDirectives(d, 'Eva').actions, []);
  }
});

test('wake is clamped to 2..60 and defaults on junk', () => {
  assert.equal(clampWake(1), 2);
  assert.equal(clampWake(600), 60);
  assert.equal(clampWake('12'), 12);
  assert.equal(clampWake(undefined), WAKE_DEFAULT);
  assert.equal(clampWake('soon'), WAKE_DEFAULT);
  assert.equal(planDirectives({ wake_in_minutes: 7.6 }, 'Eva').wakeInMinutes, 8);
});

test('long text is capped', () => {
  const { actions } = planDirectives({ goal: 'x'.repeat(1000) }, 'Eva');
  assert.equal(actions[0].goal.length, 600);
  const t = planDirectives({ tell_body: 'y'.repeat(2000) }, 'Eva').actions[0];
  assert.equal(t.payload.length, 1500);
});

test('say fits in one RCON command', () => {
  const r = planDirectives({ say: 'ω'.repeat(2000) }, 'Eva').actions[0];
  assert.ok(Buffer.byteLength(r.payload) < 1446);
});

test('a new goal interrupts the running action first (a !stay(-1) never ends), unless she stopped it herself', () => {
  const [g] = planDirectives({ goal: 'Walk to x=-216 z=-285 then stop.' }, 'Eva').actions;
  assert.equal(g.interrupt, true);
  const both = planDirectives({ goal: 'explore', stop: true }, 'Eva').actions;
  assert.deepEqual(both.map((a) => [a.payload, !!a.interrupt]), [['!stop', false], ['!goal("explore")', false]]);
  for (const d of [{ journal: 'x' }, { tell_body: 'hi' }, { say: 'hello' }]) {
    assert.ok(planDirectives(d, 'Eva').actions.every((a) => !a.interrupt), JSON.stringify(d));
  }
});

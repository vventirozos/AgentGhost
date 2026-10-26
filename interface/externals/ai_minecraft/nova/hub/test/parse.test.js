// Fixtures are real lines from the first live session (2026-10-25), verbatim.
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseServerLine, parseMindcraftLine } from '../lib/parse.js';

test('server: offline-mode chat carries "[Not Secure]" and is still chat', () => {
  const e = parseServerLine('[23:56:27 INFO]: [Not Secure] <Eva>  !searchForBlock("oak", 64)', 'Eva');
  assert.equal(e.kind, 'chat');
  assert.equal(e.player, 'Eva');
  assert.equal(e.from_body, true);
});

test('server: chat from another player is not the body', () => {
  const e = parseServerLine('[23:58:01 INFO]: [Not Secure] <Steve> hi Eva', 'Eva');
  assert.deepEqual([e.kind, e.player, e.text, e.from_body], ['chat', 'Steve', 'hi Eva', false]);
});

test('server: join / leave / ready', () => {
  assert.deepEqual(parseServerLine('[23:55:15 INFO]: Eva joined the game', 'Eva'), { kind: 'join', clock: '23:55:15', player: 'Eva', is_body: true });
  assert.equal(parseServerLine('[23:54:32 INFO]: Eva left the game', 'Eva').kind, 'leave');
  assert.equal(parseServerLine('[23:55:12 INFO]: Done (5.560s)! For help, type "help"', 'Eva').kind, 'server_ready');
});

test('server: the latest.log line form parses too', () => {
  const e = parseServerLine('[23:55:15] [Server thread/INFO]: Eva joined the game', 'Eva');
  assert.equal(e.kind, 'join');
});

test('server: deaths, but not look-alike lines', () => {
  const d = parseServerLine('[00:12:09 INFO]: Eva was slain by Zombie', 'Eva');
  assert.deepEqual([d.kind, d.is_body], ['death', true]);
  assert.equal(parseServerLine('[00:12:09 INFO]: Eva fell from a high place', 'Eva').kind, 'death');
  // real lines from the same log that must NOT read as deaths
  assert.equal(parseServerLine('[23:51:31 INFO]: Eva issued server command: /skin clear', 'Eva'), null);
  assert.equal(parseServerLine('[23:54:32 INFO]: Eva lost connection: Disconnected', 'Eva'), null);
  assert.equal(parseServerLine('[23:51:31 INFO]: Eva[/100.83.184.117:56330] logged in with entity id 39 at ([evaworld]8.5, 72.0, 9.5)', 'Eva'), null);
});

test('mindcraft: a full response with its command', () => {
  const e = parseMindcraftLine('Eva full response to system: ""!searchForBlock("oak", 64)""', 'Eva');
  assert.deepEqual([e.kind, e.to, e.text, e.command], ['body_response', 'system', '!searchForBlock("oak", 64)', '!searchForBlock("oak", 64)']);
});

test('mindcraft: an empty response is dropped', () => {
  assert.equal(parseMindcraftLine('Eva full response to system: """"', 'Eva'), null);
  assert.equal(parseMindcraftLine('Eva full response to Mind: """"', 'Eva'), null);
});

test('mindcraft: first line of a multi-line response, without closing quotes', () => {
  const e = parseMindcraftLine('Eva full response to Mind: ""On it! !collectBlocks("oak_log", 10)', 'Eva');
  assert.equal(e.text, 'On it! !collectBlocks("oak_log", 10)');
  assert.equal(e.command, '!collectBlocks("oak_log", 10)');
});

test('mindcraft: command result', () => {
  const e = parseMindcraftLine('Agent executed: !searchForBlock and got: Invalid block type: oak.', 'Eva');
  assert.deepEqual([e.kind, e.command, e.text], ['command_result', '!searchForBlock', 'Invalid block type: oak.']);
});

test('mindcraft: model timing and goal lines', () => {
  assert.equal(parseMindcraftLine('Awaiting openai api response from model gemma-4-e4b', 'Eva').kind, 'model_wait');
  assert.equal(parseMindcraftLine('Received.', 'Eva').kind, 'model_done');
  assert.equal(parseMindcraftLine('Self-prompting started.', 'Eva').kind, 'goal_started');
  assert.equal(parseMindcraftLine('Agent did not use command in the last 3 auto-prompts. Stopping auto-prompting.', 'Eva').kind, 'goal_dropped');
  assert.equal(parseMindcraftLine('\x1b[33mReceived.\x1b[0m', 'Eva').kind, 'model_done');
});

// verbatim from mindcraft.log lines 156-159 and 140-142 (2026-10-25)
test('MindcraftParser joins a multi-line command result', async () => {
  const { MindcraftParser } = await import('../lib/parse.js');
  const p = new MindcraftParser('Eva');
  const lines = [
    'Agent executed: !craftRecipe and got: Action output:',
    'You do not have the resources to craft a wooden_pickaxe. It requires: oak_planks: 3, stick: 2.',
    '',
    'Saved memory to: ./bots/Eva/memory.json',
    'Agent executed: !craftRecipe and got: Invalid item type: workbench.',
    'Storing memories...',
  ];
  const evs = lines.flatMap((l) => p.feed(l));
  assert.deepEqual(evs.map((e) => [e.kind, e.text]), [
    ['command_result', 'You do not have the resources to craft a wooden_pickaxe. It requires: oak_planks: 3, stick: 2.'],
    ['command_result', 'Invalid item type: workbench.'],
  ]);
});

test('MindcraftParser flushes a pending result when a known line arrives', async () => {
  const { MindcraftParser } = await import('../lib/parse.js');
  const p = new MindcraftParser('Eva');
  assert.deepEqual(p.feed('Agent executed: !collectBlocks and got: Action output:'), []);
  assert.deepEqual(p.feed('Collected 5 oak_log.'), []);
  const evs = p.feed('Awaiting openai api response from model gemma-4-e4b');
  assert.deepEqual(evs.map((e) => [e.kind, e.text]), [['command_result', 'Collected 5 oak_log.'], ['model_wait', undefined]]);
});

test('mindcraft: its own restart of the body process is recorded (10-26 07:55, verbatim)', () => {
  const e = parseMindcraftLine('Agent process exited with code 1 and signal null', 'Eva');
  assert.deepEqual([e.kind, e.code, e.signal], ['body_process_exit', 1, null]);
  const s = parseMindcraftLine('Agent process exited with code null and signal SIGTERM', 'Eva');  // 10-26 08:50
  assert.deepEqual([s.kind, s.code, s.signal], ['body_process_exit', null, 'SIGTERM']);
  assert.equal(parseMindcraftLine('Restarting agent...', 'Eva'), null);
});

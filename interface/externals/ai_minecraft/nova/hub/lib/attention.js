// Attention: things that should wake Eva before the time she chose.
//
// She picks when to look again; these interrupts exist so that a death or a
// body that gave up is not left unseen for an hour. Each reason is raised
// once and cleared when she looks. MIN_GAP_MS still applies: an early wake
// never comes sooner than that after her last look, so a body dying in a
// loop cannot hammer the agent's only slot.

export const MIN_GAP_MS = 2 * 60 * 1000;
export const STUCK_MS = 5 * 60 * 1000;
const STUCK_RADIUS = 2;
const REPEAT_LIMIT = 4;
export const FAIL_RUN = 6;

// A Mindcraft command result that reports a failure. Phrases from Mindcraft's
// own messages (src/agent/library/skills.js, commands/index.js). "Path not
// found, but attempting to navigate anyway…" is NOT one: it usually ends in
// "You have reached…" (a failed walk ends "Pathfinding stopped" / "Unable to
// reach"). Bare "undefined" (what !stay returns when stopped) is not either.
const FAILED = new RegExp([
  '^Invalid ', 'is not a command', 'was given \\d+ args', 'incorrectly formatted', 'Error: Param',
  'Could not ', 'Cannot ', 'Failed to ', 'Unable to reach', 'Pathfinding stopped',
  "Don't have ", 'You do not have ', "You don't have ", 'You have no ', '^No (more )?\\S+ (nearby|to )',
  'There is no ', 'An error occurred', 'did not go through', 'is either not an item', 'requires a crafting table', 'not found\\.',
].join('|'));
export const isFailedResult = (text) => FAILED.test(String(text || '').trim());

export class Attention {
  constructor() {
    this.reasons = new Map(); // reason -> latest ts
    this.anchor = null;       // {x, y, z, ts} last place the body moved away from
    this.lastResponses = [];
    this.lastKind = null;
    this.failRun = 0;         // command results in a row that failed
  }

  // The latest time counts: a second death while she is thinking about the
  // first must still wake her after that look (clearUpTo).
  raise(reason, ts) {
    this.reasons.set(reason, Math.max(ts, this.reasons.get(reason) ?? ts));
  }

  onEvent(e) {
    if (e.kind === 'death' && e.is_body) this.raise('your body died', e.ts);
    else if (e.kind === 'body_lost') this.raise('your body dropped out of the world', e.ts);
    else if (e.kind === 'body_back') this.raise('your body came back into the world', e.ts);
    else if (e.kind === 'goal_dropped') this.raise('your body gave up on its goal', e.ts);
    else if (e.kind === 'chat' && !e.from_body) this.raise(`${e.player} spoke in the world`, e.ts);
    else if (e.kind === 'join' && !e.is_body) this.raise(`${e.player} joined the world`, e.ts);
    else if (e.kind === 'command_result') {
      // 10-26: alternating "oak" / "acacia" / moveAway failed 57 times in 15 min
      // and never matched the identical-repeat rule below. Once per run: a
      // success starts a new one.
      if (!isFailedResult(e.text)) this.failRun = 0;
      else if (++this.failRun === FAIL_RUN) this.raise(`your body's last ${FAIL_RUN} commands all failed`, e.ts);
    } else if (e.kind === 'body_response') {
      const key = (e.command || e.text || '').trim();
      this.lastResponses.push(key);
      if (this.lastResponses.length > REPEAT_LIMIT) this.lastResponses.shift();
      if (this.lastResponses.length === REPEAT_LIMIT && key && this.lastResponses.every((r) => r === key)) {
        this.raise('your body is repeating the same thing', e.ts);
      }
    }
  }

  // Called with every Mindcraft state update.
  onState(state, ts) {
    const kind = state?.action?.kind;
    if (this.lastKind && this.lastKind !== 'stopped' && kind === 'stopped') {
      this.raise('your body stopped and is waiting', ts);
    }
    this.lastKind = kind;
    const p = state?.gameplay?.position;
    if (!p) return;
    const a = this.anchor;
    if (state?.action?.current === 'action:stay') {
      // Standing still on purpose (!stay) is not stuck: 10-26 this alert woke
      // her every 5 min for 90 min over the 30-40 min she chose. The clock
      // restarts when the stay ends.
      this.anchor = { x: p.x, y: p.y, z: p.z, ts };
      return;
    }
    if (!a || Math.hypot(p.x - a.x, p.y - a.y, p.z - a.z) > STUCK_RADIUS) {
      this.anchor = { x: p.x, y: p.y, z: p.z, ts };
    } else if ((kind === 'thinking' || kind === 'acting') && ts - a.ts > STUCK_MS) {
      this.raise('your body has not moved for 5 minutes while busy', ts);
      this.anchor = { ...a, ts }; // re-arm after another STUCK_MS
    }
  }

  pending() {
    return [...this.reasons.keys()];
  }

  // Should Eva wake now? wakeAt is the time she chose.
  due(now, lastLookTs, wakeAt) {
    if (wakeAt != null && now >= wakeAt) return true;
    if (!this.reasons.size) return false;
    return lastLookTs == null || now - lastLookTs >= MIN_GAP_MS;
  }

  clear() {
    this.reasons.clear();
    this.lastResponses = [];
  }

  // After a look: drop what she saw (raised at or before lookTs). A reason
  // raised while she was thinking (a look can take minutes) stays pending.
  clearUpTo(lookTs) {
    for (const [r, ts] of this.reasons) if (ts <= lookTs) this.reasons.delete(r);
    this.lastResponses = [];
  }
}

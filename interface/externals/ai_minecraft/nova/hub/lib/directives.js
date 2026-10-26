// Turn Eva's decision into what the world receives.
//
// The body (Mindcraft) gets messages "from Mind". A message that starts with
// a command (`!goal("...")`, `!stop`) is executed by Mindcraft directly,
// without asking the small model; anything else is answered by the small
// model in its own words. `say` goes to world chat through RCON.
//
// Mindcraft parses command arguments itself, so a double quote inside a goal
// would end the argument early: quotes become single quotes, newlines spaces.

// Eva's instructions run ~1,100 characters (first night: a 600 cap cut one
// mid-sentence); Gemma has 16k of context per slot, so these are generous.
export const MAX_GOAL = 600;
export const MAX_TEXT = 1500;
// say goes through RCON, whose command limit is ~1446 bytes: room for the
// tellraw wrapper and multi-byte text
export const MAX_SAY = 400;
export const WAKE_MIN = 2;
export const WAKE_MAX = 60;
export const WAKE_DEFAULT = 10;

function clean(s, n) {
  if (s == null) return null;
  s = String(s).replace(/[\r\n\t]+/g, ' ').replace(/\s{2,}/g, ' ').trim();
  if (!s || s.toLowerCase() === 'null') return null;
  return s.length > n ? s.slice(0, n) : s;
}

export function clampWake(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return WAKE_DEFAULT;
  return Math.min(WAKE_MAX, Math.max(WAKE_MIN, Math.round(n)));
}

// tellraw JSON text component; JSON.stringify does the escaping.
export function tellrawCommand(speaker, text) {
  return `tellraw @a ${JSON.stringify([{ text: `<${speaker}> ` }, { text }])}`;
}

export function planDirectives(decision, botName) {
  const d = decision || {};
  const out = []; // {channel: 'body'|'rcon'|'journal', payload, summary}
  if (d.stop === true) out.push({ channel: 'body', payload: '!stop', summary: 'told your body to stop' });
  const goal = clean(d.goal, MAX_GOAL);
  if (goal) {
    const g = goal.replace(/"/g, "'");
    // !goal only replaces the goal text: Mindcraft keeps running the current
    // action, and a !stay(-1) never ends (10-26: five goals in 90 min ignored).
    // So a new goal interrupts first, unless her own stop already did.
    out.push({ channel: 'body', payload: `!goal("${g}")`, summary: `set the goal: "${g}"`, goal: g, interrupt: d.stop !== true });
  }
  const tell = clean(d.tell_body, MAX_TEXT);
  if (tell) {
    // Mindcraft runs the first `!word` ANYWHERE in a message as a command and
    // the small model never reads the rest (first night: "No more !moveAway"
    // ran !moveAway). tell_body is meant to be read, so no `!` may precede a
    // word: "do not use !moveAway" reaches the body as "do not use moveAway".
    const t = tell.replace(/!(?=\w)/g, '');
    out.push({ channel: 'body', payload: t, summary: `told your body: "${t}"` });
  }
  const say = clean(d.say, MAX_SAY);
  if (say) out.push({ channel: 'rcon', payload: tellrawCommand(botName, say), summary: `said aloud: "${say}"` });
  const journal = clean(d.journal, 4000);
  if (journal) out.push({ channel: 'journal', payload: journal, summary: 'wrote in your journal' });
  return { actions: out, wakeInMinutes: clampWake(d.wake_in_minutes) };
}

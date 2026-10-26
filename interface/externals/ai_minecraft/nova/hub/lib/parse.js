// Turn raw log lines into world events.
//
// Two sources: the Paper server's log (chat, joins, deaths) and Mindcraft's
// stdout (the body's full responses, model latency, errors). Both are plain
// text written by other programs, so every pattern here was taken from a real
// line; tests/parse.test.js holds those lines verbatim.

// Console form "[23:41:07 INFO]: <Steve> hello there"; latest.log form
// "[23:41:07] [Server thread/INFO]: <Steve> hello there".
const SERVER_LINE = /^\[(\d\d:\d\d:\d\d)(?: (\w+)\]|\] \[[^\]]*\/(\w+)\]): (.*)$/;
const DEATH_VERBS = /^(was|died|drowned|fell|burned|blew up|hit the ground|starved|suffocated|froze|withered|went|tried|walked|experienced|discovered|didn't|left the confines|was killed|was slain|was shot|was blown|was squashed|was impaled|was stung|was poked|was pummeled|was fireballed|was obliterated|was skewered)\b/;

export function parseServerLine(line, botName) {
  const m = SERVER_LINE.exec(line.trim());
  if (!m) return null;
  const [, clock, lvlA, lvlB, msg] = m;
  const level = lvlA || lvlB;
  // Paper marks unsigned chat (every offline-mode player) "[Not Secure] ".
  let c = /^(?:\[Not Secure\] )?<([^>]+)> (.*)$/.exec(msg);
  if (c) return { kind: 'chat', clock, player: c[1], text: c[2], from_body: c[1] === botName };
  c = /^(\S+) joined the game$/.exec(msg);
  if (c) return { kind: 'join', clock, player: c[1], is_body: c[1] === botName };
  c = /^(\S+) left the game$/.exec(msg);
  if (c) return { kind: 'leave', clock, player: c[1], is_body: c[1] === botName };
  c = /^(\S+) (.+)$/.exec(msg);
  if (c && level === 'INFO' && DEATH_VERBS.test(c[2]) && !/[[\]:=]/.test(c[2])) {
    return { kind: 'death', clock, player: c[1], text: msg, is_body: c[1] === botName };
  }
  if (/^Done \([\d.]+s\)!/.test(msg)) return { kind: 'server_ready', clock };
  return null;
}

// Mindcraft prints `Eva full response to Mind: ""text""` for every model
// answer (the doubled quotes are Mindcraft's own).
export function parseMindcraftLine(line, botName) {
  const t = line.replace(/\x1b\[[0-9;]*m/g, '').trimEnd();
  const pref = `${botName} full response to `;
  if (t.startsWith(pref)) {
    const rest = t.slice(pref.length);
    const i = rest.indexOf(': ""');
    if (i > 0) {
      let text = rest.slice(i + 4);
      if (text.endsWith('""')) text = text.slice(0, -2);
      // A multi-line answer continues on the next stdout lines; the first
      // line (where the command almost always is) is what we keep.
      text = text.trim();
      if (!text) return null; // an empty answer: the body chose to say nothing
      const cmd = /(![a-zA-Z]+)(\([^)]*\))?/.exec(text);
      return { kind: 'body_response', to: rest.slice(0, i), text, command: cmd ? cmd[0] : null };
    }
  }
  const ex = /^Agent executed: (![A-Za-z]+) and got: (.*)$/.exec(t);
  if (ex) return { kind: 'command_result', command: ex[1], text: ex[2].trim() };
  if (/^Awaiting openai api response/.test(t)) return { kind: 'model_wait' };
  if (t === 'Received.') return { kind: 'model_done' };
  if (/Agent did not use command in the last \d+ auto-prompts/.test(t)) return { kind: 'goal_dropped', text: t };
  if (/^Self-prompting started/.test(t)) return { kind: 'goal_started' };
  if (/Context length exceeded/.test(t)) return { kind: 'context_overflow', text: t };
  if (/My brain disconnected/.test(t)) return { kind: 'model_error', text: t };
  // Mindcraft kills and restarts its own agent process, e.g. its "unstuck"
  // mode after 10 s of failing to move away (10-26 07:55); the hub saw
  // nothing of it. The body keeps memory.json and loses the conversation.
  // A signal exit logs "code null and signal SIGTERM".
  const exit = /^Agent process exited with code (\d+|null)(?: and signal (\w+))?/.exec(t);
  if (exit) {
    const signal = exit[2] && exit[2] !== 'null' ? exit[2] : null;
    return { kind: 'body_process_exit', code: exit[1] === 'null' ? null : Number(exit[1]), signal, text: t };
  }
  return null;
}

// Mindcraft prints most command results as
//   Agent executed: !craftRecipe and got: Action output:
//   You do not have the resources to craft a wooden_pickaxe. It requires: …
//   <blank line>
// so a result whose text ends in ':' collects the following lines until a
// blank line or any line the parser recognises. Stateful, one per log.
const MAX_RESULT_LINES = 6;

export class MindcraftParser {
  constructor(botName) {
    this.botName = botName;
    this.pending = null;
  }

  // Returns the events completed by this line (0, 1 or 2).
  feed(line) {
    const out = [];
    const e = parseMindcraftLine(line, this.botName);
    if (this.pending) {
      const t = line.replace(/\x1b\[[0-9;]*m/g, '').trim();
      if (!e && t && this.pending.lines.length < MAX_RESULT_LINES) {
        this.pending.lines.push(t);
        return out;
      }
      if (!e && t) return out; // past the cap: drop, keep waiting for the blank line
      const p = this.pending;
      this.pending = null;
      const body = p.lines.join(' ');
      out.push({ ...p.event, text: body ? `${p.event.text.replace(/Action output:$/, '').trim()} ${body}`.trim() : p.event.text });
    }
    if (e && e.kind === 'command_result' && /:$/.test(e.text)) {
      this.pending = { event: e, lines: [] };
    } else if (e) {
      out.push(e);
    }
    return out;
  }
}

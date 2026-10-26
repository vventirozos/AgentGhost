// The diary: what Eva reads when she wakes up.
//
// Built from the latest Mindcraft state plus the events since her last look.
// It is the ONLY view of the world she gets, so it reports what the logs say
// and nothing inferred. Kept to a few thousand characters: the 27B prefills
// it at ~140 tok/s, and every second of that holds the agent's only slot.

const MAX_EVENT_LINES = 60;
const MAX_LINE_CHARS = 240;

// Minecraft time: tick 0 is 06:00, 24000 ticks per day.
export function gameClock(timeOfDay) {
  const t = ((Number(timeOfDay) % 24000) + 24000) % 24000;
  const hours = Math.floor(t / 1000 + 6) % 24;
  const minutes = Math.floor(((t % 1000) / 1000) * 60);
  const hh = String(hours).padStart(2, '0');
  const mm = String(minutes).padStart(2, '0');
  let phase = 'night';
  if (t < 1000 || t >= 23000) phase = 'dawn';
  else if (t < 11000) phase = 'day';
  else if (t < 13000) phase = 'dusk';
  return `${hh}:${mm} (${phase})`;
}

function clip(s, n = MAX_LINE_CHARS) {
  s = String(s ?? '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function fmtInventory(counts) {
  const items = Object.entries(counts || {}).filter(([, n]) => n > 0);
  if (!items.length) return 'empty';
  items.sort((a, b) => b[1] - a[1]);
  return items.map(([k, n]) => `${k} x${n}`).join(', ');
}

function hhmm(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

// One diary line per event worth Eva's attention; null drops the event.
export function eventLine(e) {
  const at = hhmm(e.ts);
  switch (e.kind) {
    case 'body_response': return `${at} body${e.to && e.to !== 'system' ? ` (to ${e.to})` : ''}: ${clip(e.text)}`;
    case 'command_result': return `${at}   → ${e.command} result: ${clip(e.text)}`;
    case 'bot_output': return `${at} body said: ${clip(e.text)}`;
    case 'chat': return e.from_body ? null : `${at} ${e.player} said in chat: ${clip(e.text)}`;
    case 'join': return e.is_body ? `${at} your body entered the world` : `${at} ${e.player} joined the world`;
    case 'leave': return e.is_body ? `${at} your body left the world` : `${at} ${e.player} left the world`;
    case 'death': return e.is_body ? `${at} YOUR BODY DIED: ${clip(e.text)}` : `${at} ${clip(e.text)}`;
    case 'body_lost': return `${at} YOUR BODY DROPPED OUT OF THE WORLD (disconnected)`;
    case 'body_back': return `${at} ${e.text}`;
    case 'body_restart': return null;
    case 'body_process_exit': return `${at} your body's process stopped (${e.code != null ? `exit code ${e.code}` : `signal ${e.signal}`}) and Mindcraft restarted it: it kept its saved memory but lost what it was doing`;
    case 'goal_dropped': return `${at} your body gave up on its goal (it stopped issuing commands)`;
    case 'context_overflow': return `${at} your body's short-term memory overflowed and was trimmed`;
    case 'model_error': return `${at} your body's mind failed to respond once`;
    case 'directive': return `${at} you ${e.summary}`;
    case 'intervention': return `${at} the operator intervened: ${clip(e.text)}`;
    case 'paused': return `${at} time was paused`;
    case 'resumed': return `${at} time resumed`;
    default: return null;
  }
}

// The body's words arrive twice: Mindcraft logs the model's full response
// (body_response) and also emits what it said in chat (bot_output). Keep the
// full response; keep a bot_output only if no nearby response contains it
// (narrations like "Picking up item!" have no response twin).
export function dedupe(events) {
  const norm = (s) => String(s ?? '').replace(/\s+/g, ' ').trim();
  return events.filter((e, i) => {
    if (e.kind !== 'bot_output') return true;
    const t = norm(e.text);
    if (!t) return false;
    for (let j = Math.max(0, i - 6); j < Math.min(events.length, i + 7); j++) {
      const o = events[j];
      if (o.kind === 'body_response' && norm(o.text).includes(t)) return false;
    }
    return true;
  });
}

export function buildDiary({ state, day, events, lastLookTs, now, reasons, currentGoal }) {
  const lines = [];
  const g = state?.gameplay;
  const mins = lastLookTs ? Math.round((now - lastLookTs) / 60000) : null;
  lines.push(`WORLD TIME: ${day != null ? `day ${day}, ` : ''}${g ? gameClock(g.timeOfDay) : 'unknown'}` +
    (mins != null ? ` — real minutes since you last looked: ${mins}` : ' — this is your first look this session'));
  if (g) {
    const p = g.position || {};
    lines.push(`BODY: health ${g.health}/20, hunger ${g.hunger}/20, at x=${Math.round(p.x)} y=${Math.round(p.y)} z=${Math.round(p.z)}` +
      ` (${g.biome || '?'}, ${g.dimension || 'overworld'}), weather ${String(g.weather || '?').toLowerCase()}`);
    lines.push(`DOING: ${state.action?.current || 'unknown'}${currentGoal ? ` — current goal: "${clip(currentGoal, 200)}"` : ' — no goal set'}`);
    lines.push(`INVENTORY: ${fmtInventory(state.inventory?.counts)}`);
    const eq = state.inventory?.equipment || {};
    const worn = ['helmet', 'chestplate', 'leggings', 'boots'].map((k) => eq[k]).filter(Boolean);
    lines.push(`HOLDING: ${eq.mainHand || 'nothing'}${worn.length ? `; wearing ${worn.join(', ')}` : ''}`);
    const nb = state.nearby || {};
    const ents = [...new Set(nb.entityTypes || [])];
    lines.push(`NEARBY: players ${nb.humanPlayers?.length ? nb.humanPlayers.join(', ') : 'none'}; creatures ${ents.length ? ents.join(', ') : 'none'}`);
    const s = state.surroundings || {};
    if (s.below) lines.push(`STANDING ON: ${s.below}${s.firstBlockAboveHead && s.firstBlockAboveHead !== 'none' ? `; above head: ${s.firstBlockAboveHead}` : '; open sky above'}`);
  } else {
    lines.push('BODY: not in the world right now');
  }
  const evLines = dedupe(events || []).map(eventLine).filter(Boolean);
  const dropped = Math.max(0, evLines.length - MAX_EVENT_LINES);
  lines.push('');
  lines.push('WHAT HAPPENED SINCE YOUR LAST LOOK (oldest first):');
  if (dropped) lines.push(`  (${dropped} earlier lines omitted)`);
  if (!evLines.length) lines.push('  nothing was recorded');
  for (const l of evLines.slice(-MAX_EVENT_LINES)) lines.push('  ' + l);
  lines.push('');
  lines.push(`WHY YOU WOKE: ${reasons?.length ? reasons.join('; ') : 'you chose this time'}`);
  return lines.join('\n');
}

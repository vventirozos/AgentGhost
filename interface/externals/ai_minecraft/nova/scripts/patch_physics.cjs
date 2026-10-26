// Guard mineflayer's position packets against non-finite values.
//
// First live session (2026-10-26 00:05): the server kicked the body twice in
// six seconds with "Invalid move player packet received" — the vanilla check
// for a NaN/infinite coordinate or angle — while it stood at the water's edge
// (Mindcraft's own mineflayer patch already notes "water calc is bugged").
// Mindcraft restarted it once, then gave up, and the body was gone.
//
// This inserts a check into updatePosition(): a tick whose position or look is
// not finite is not sent; the last good position is restored and velocity
// zeroed, and the guard logs (at most once a minute) so the log shows whether
// this was the cause. Confirmed on the first run: it fired once at the same
// spot and the body stayed in (it had been kicked 4 s after every rejoin).
// Idempotent; exits non-zero if the anchor is missing.
const fs = require('fs');
const file = process.argv[2];
const MARK = 'EVA-WORLD-GUARD v2';
const OLD_MARK = '    // EVA-WORLD-GUARD:';
let src = fs.readFileSync(file, 'utf8');
if (src.includes(MARK)) { console.log('physics guard: already applied'); process.exit(0); }
// v1 logged the position AFTER restoring it, hiding which value was bad: drop it.
const v1 = src.indexOf(OLD_MARK);
if (v1 >= 0) {
  const end = src.indexOf('      return\n    }\n', v1);
  if (end < 0) { console.error('physics guard: cannot find the end of the v1 guard'); process.exit(1); }
  src = src.slice(0, v1) + src.slice(end + '      return\n    }\n'.length);
}
const anchor = '    const onGround = bot.entity.onGround\n';
const idx = src.indexOf(anchor, src.indexOf('function updatePosition'));
if (idx < 0) { console.error('physics guard: anchor not found in updatePosition'); process.exit(1); }
const guard = `    // ${MARK}: never send a non-finite position/look (the server kicks with
    // "Invalid move player packet received"); restore the last good state.
    if (![position.x, position.y, position.z, yaw, pitch].every(Number.isFinite)) {
      const bad = { x: position.x, y: position.y, z: position.z, yaw, pitch, vx: bot.entity.velocity?.x, vy: bot.entity.velocity?.y, vz: bot.entity.velocity?.z }
      if ([lastSent.x, lastSent.y, lastSent.z].every(Number.isFinite)) bot.entity.position.set(lastSent.x, lastSent.y, lastSent.z)
      if (bot.entity.velocity) bot.entity.velocity.set(0, 0, 0)
      if (!Number.isFinite(lastSentYaw)) lastSentYaw = 0
      if (!Number.isFinite(lastSentPitch)) lastSentPitch = 0
      if (!Number.isFinite(bot.entity.yaw)) bot.entity.yaw = 0
      if (!Number.isFinite(bot.entity.pitch)) bot.entity.pitch = 0
      if (!bot._evaGuardTs || now - bot._evaGuardTs > 60000) {
        bot._evaGuardTs = now
        console.log('[eva-world] blocked a non-finite position/look packet', Object.entries(bad).map(([k, v]) => k + '=' + v).join(' '))
      }
      return
    }
`;
src = src.slice(0, idx + anchor.length) + guard + src.slice(idx + anchor.length);
fs.writeFileSync(file, src);
console.log('physics guard: applied');

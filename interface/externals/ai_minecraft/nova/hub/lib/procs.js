// Child processes the hub owns: the Minecraft server and Mindcraft.
//
// Each runs DETACHED in its own process group with output to a log file and
// its pid in a pidfile, so a hub restart (launchd KeepAlive) adopts a running
// world instead of orphaning it. Mindcraft forks one node process per bot,
// which is why stop() signals the whole group (-pid), not just the parent.
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';

export function pidAlive(pid) {
  if (!pid) return false;
  try { process.kill(pid, 0); return true; } catch (e) { return e.code === 'EPERM'; }
}

function commandOf(pid) {
  try { return execFileSync('ps', ['-p', String(pid), '-o', 'command='], { encoding: 'utf8' }).trim(); } catch { return ''; }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Proc {
  constructor({ name, cmd, args, cwd, env, logFile, pidFile, match }) {
    Object.assign(this, { name, cmd, args, cwd, env, logFile, pidFile, match });
  }

  pid() {
    try { return Number(fs.readFileSync(this.pidFile, 'utf8').trim()) || null; } catch { return null; }
  }

  // Alive AND still the program we started: a recycled pid is not ours.
  alive() {
    const pid = this.pid();
    return pidAlive(pid) && commandOf(pid).includes(this.match);
  }

  start() {
    if (this.alive()) return this.pid();
    const fd = fs.openSync(this.logFile, 'a');
    const child = spawn(this.cmd, this.args, {
      cwd: this.cwd, env: { ...process.env, ...this.env }, detached: true, stdio: ['ignore', fd, fd],
    });
    fs.closeSync(fd);
    child.unref();
    fs.writeFileSync(this.pidFile, String(child.pid));
    return child.pid;
  }

  async stop(graceMs = 15000) {
    const pid = this.pid();
    if (!this.alive()) { try { fs.unlinkSync(this.pidFile); } catch { /* none */ } return 'not running'; }
    try { process.kill(-pid, 'SIGTERM'); } catch { try { process.kill(pid, 'SIGTERM'); } catch { /* gone */ } }
    const until = Date.now() + graceMs;
    while (Date.now() < until && pidAlive(pid)) await sleep(250);
    let how = 'stopped';
    if (pidAlive(pid)) {
      try { process.kill(-pid, 'SIGKILL'); } catch { try { process.kill(pid, 'SIGKILL'); } catch { /* gone */ } }
      how = 'killed';
    }
    try { fs.unlinkSync(this.pidFile); } catch { /* none */ }
    return how;
  }

  async waitExit(ms) {
    const pid = this.pid();
    const until = Date.now() + ms;
    while (Date.now() < until && pidAlive(pid)) await sleep(250);
    return !pidAlive(pid);
  }
}

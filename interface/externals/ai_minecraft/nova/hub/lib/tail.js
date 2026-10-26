// Follow a growing log file line by line (polling; fs.watch is unreliable
// for appends on macOS). A file that shrinks was truncated or replaced, so
// reading restarts from its beginning.
import fs from 'node:fs';

export class Tail {
  constructor(file, onLine, { fromEnd = true, intervalMs = 500 } = {}) {
    this.file = file;
    this.onLine = onLine;
    this.intervalMs = intervalMs;
    this.partial = '';
    this.offset = 0;
    if (fromEnd) { try { this.offset = fs.statSync(file).size; } catch { this.offset = 0; } }
    this.timer = null;
  }

  start() {
    if (!this.timer) this.timer = setInterval(() => this.poll(), this.intervalMs);
    return this;
  }

  stop() { clearInterval(this.timer); this.timer = null; }

  poll() {
    let size;
    try { size = fs.statSync(this.file).size; } catch { return; }
    if (size < this.offset) { this.offset = 0; this.partial = ''; }
    if (size === this.offset) return;
    const fd = fs.openSync(this.file, 'r');
    const len = Math.min(size - this.offset, 4 * 1024 * 1024);
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, this.offset);
    fs.closeSync(fd);
    this.offset += len;
    const text = this.partial + buf.toString('utf8');
    const lines = text.split('\n');
    this.partial = lines.pop();
    for (const l of lines) { try { this.onLine(l); } catch { /* a parser bug must not stop the tail */ } }
  }
}

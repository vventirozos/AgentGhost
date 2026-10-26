// Minimal Minecraft RCON client: one TCP connection per command.
//
// Packet: int32 LE length (of the rest), int32 id, int32 type, ASCII body,
// two NUL bytes. Auth is type 3, a command type 2, a reply type 0. A failed
// auth answers with id -1. One connection per command keeps this stateless:
// the server can restart between calls without a reconnect dance.
import net from 'node:net';

const AUTH = 3;
const EXEC = 2;

export function encodePacket(id, type, body) {
  const payload = Buffer.from(body, 'utf8');
  const buf = Buffer.alloc(14 + payload.length);
  buf.writeInt32LE(10 + payload.length, 0);
  buf.writeInt32LE(id, 4);
  buf.writeInt32LE(type, 8);
  payload.copy(buf, 12);
  return buf; // the last two bytes are already 0
}

// Returns [packets, rest]. Packets are {id, type, body}.
export function decodePackets(buf) {
  const out = [];
  let off = 0;
  while (buf.length - off >= 4) {
    const len = buf.readInt32LE(off);
    if (buf.length - off - 4 < len) break;
    const id = buf.readInt32LE(off + 4);
    const type = buf.readInt32LE(off + 8);
    const body = buf.slice(off + 12, off + 4 + len - 2).toString('utf8');
    out.push({ id, type, body });
    off += 4 + len;
  }
  return [out, buf.slice(off)];
}

export function rcon(command, { host = '127.0.0.1', port = 25575, password, timeoutMs = 5000 } = {}) {
  return new Promise((resolve, reject) => {
    const sock = net.connect({ host, port });
    let pending = Buffer.alloc(0);
    let authed = false;
    const timer = setTimeout(() => { sock.destroy(); reject(new Error(`rcon timeout: ${command}`)); }, timeoutMs);
    const done = (err, val) => { clearTimeout(timer); sock.destroy(); err ? reject(err) : resolve(val); };
    sock.on('error', (e) => done(e));
    sock.on('connect', () => sock.write(encodePacket(1, AUTH, password || '')));
    sock.on('data', (chunk) => {
      let packets;
      [packets, pending] = decodePackets(Buffer.concat([pending, chunk]));
      for (const p of packets) {
        if (!authed) {
          if (p.id === -1) return done(new Error('rcon auth failed'));
          if (p.type === EXEC || p.id === 1) { // auth response is type 2, id 1
            authed = true;
            sock.write(encodePacket(2, EXEC, command));
          }
          continue;
        }
        if (p.id === 2) return done(null, p.body);
      }
    });
  });
}

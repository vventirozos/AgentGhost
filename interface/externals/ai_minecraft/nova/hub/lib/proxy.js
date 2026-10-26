// Reverse proxy for the 3D viewer (prismarine-viewer on 127.0.0.1:3000).
//
// The viewer has no authentication of its own, so it listens on localhost
// only (patched by apply_patches.sh) and is reachable from outside solely
// through the hub, behind the hub's key. Plain HTTP requests are piped;
// the viewer's socket.io WebSocket arrives as an `upgrade` and is spliced
// byte-for-byte onto a fresh TCP connection.
import http from 'node:http';
import net from 'node:net';

export function proxyRequest(req, res, port) {
  const up = http.request({ host: '127.0.0.1', port, path: req.url, method: req.method, headers: req.headers }, (ur) => {
    res.writeHead(ur.statusCode, ur.headers);
    ur.pipe(res);
  });
  up.on('error', () => { if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' }); res.end('viewer not running'); });
  req.pipe(up);
}

export function proxyUpgrade(req, socket, head, port) {
  const up = net.connect(port, '127.0.0.1', () => {
    let raw = `${req.method} ${req.url} HTTP/${req.httpVersion}\r\n`;
    for (let i = 0; i < req.rawHeaders.length; i += 2) raw += `${req.rawHeaders[i]}: ${req.rawHeaders[i + 1]}\r\n`;
    up.write(raw + '\r\n');
    if (head && head.length) up.write(head);
    socket.pipe(up).pipe(socket);
  });
  const kill = () => { socket.destroy(); up.destroy(); };
  up.on('error', kill);
  socket.on('error', kill);
}

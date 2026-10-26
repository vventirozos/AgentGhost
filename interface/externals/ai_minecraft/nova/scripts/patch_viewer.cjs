// Give the monitor a "find Eva" button: expose a re-centre hook in the 3D
// viewer's client bundle (prismarine-viewer public/index.js, minified).
//
// The stock viewer aims its orbit camera at the body only on the FIRST
// position update; drag or zoom away and nothing brings it back, and the
// operator loses her (10-26). This keeps her last position on window.__evaLast
// and adds window.__evaRecenter(): it aims the orbit at her chest and puts
// the camera 12 blocks from her along the direction you were looking from,
// so your angle stays (v1 kept your distance too: the stock 28 blocks made
// her a speck behind terrain). Never from below: at least 4 blocks up.
// The monitor page calls it from the same origin (the hub proxies /viewer/).
// Idempotent (upgrades v1); exits non-zero if an anchor is missing.
const fs = require('fs');
const file = process.argv[2];
const MARK = '/*EVA-WORLD-RECENTER v2*/';
const V1 = '/*EVA-WORLD-RECENTER v1*/';
const LAST = 'window.__evaLast=t;';
let src = fs.readFileSync(file, 'utf8');
if (src.includes(MARK)) { console.log('viewer recenter: already applied'); process.exit(0); }
const v1 = src.indexOf(V1);
if (v1 >= 0) {
  const end = src.indexOf('return!0};', v1);
  if (end < 0) { console.error('viewer recenter: cannot find the end of the v1 hook'); process.exit(1); }
  src = src.slice(0, v1) + src.slice(end + 'return!0};'.length);
  src = src.replace(LAST, '');
}

const CTRL = 'const u=new r(l);let h=new THREE.OrbitControls(u.camera,l.domElement);';
const POS = 'o.on("position",(({pos:t,addMesh:i,yaw:r,pitch:o})=>{';
for (const [name, a] of [['controls', CTRL], ['position', POS]]) {
  const n = src.split(a).length - 1;
  if (n !== 1) { console.error(`viewer recenter: ${name} anchor found ${n} times (want 1)`); process.exit(1); }
}
const hook = `${MARK}window.__evaRecenter=()=>{const t=window.__evaLast;if(!t||!h)return!1;` +
  `const c=new THREE.Vector3(t.x,t.y+1,t.z),d=u.camera.position.clone().sub(h.target);` +
  `if(d.lengthSq()<1e-6)d.set(0,1,1);d.normalize();if(d.y<.35){d.y=.35;d.normalize()}` +
  `h.target.copy(c);u.camera.position.copy(c).addScaledVector(d,12);h.update();return!0};`;
src = src.replace(CTRL, CTRL + hook).replace(POS, POS + LAST);
fs.writeFileSync(file, src);
console.log('viewer recenter: applied');

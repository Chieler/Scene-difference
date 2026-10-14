# viewer.py
"""Self-contained HTML viewer (three.js from CDN; needs internet to load the library)."""
import base64, json
from pathlib import Path
import numpy as np

HTML = """<!doctype html><meta charset=utf-8><title>Room changes</title>
<style>body{margin:0;overflow:hidden;font:13px system-ui;color:#eee;background:#111}
#ui{position:fixed;top:0;left:0;width:290px;max-height:100%;overflow:auto;background:#000a;padding:10px}
#ui div.c{cursor:pointer;padding:4px 0;border-top:1px solid #333}.lb{position:fixed;font-size:11px;pointer-events:none;
text-shadow:0 0 3px #000}</style>
<div id=ui><b>View</b><br>
<label><input type=radio name=m value=diff checked> Diff</label>
<label><input type=radio name=m value=A> Video A</label>
<label><input type=radio name=m value=B> Video B</label>
<p style="color:#999;margin:6px 0">Diff: B cloud, grey = not seen in A (not comparable), red ghost = removed objects as they were in A.
Green = added, red = removed, blue = moved.</p><div id=list></div></div>
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const buf = s => Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer;
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200); cam.up.set(0, 0, 1);
const r = new THREE.WebGLRenderer({antialias: true}); r.setSize(innerWidth, innerHeight); document.body.append(r.domElement);
const ctl = new OrbitControls(cam, r.domElement);
const pts = (pos, col, size = 0.025) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf(pos)), 3));
  if (col) g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buf(col)), 3, true));
  return new THREE.Points(g, new THREE.PointsMaterial(col ? {size, vertexColors: true} : {size, color: 0xff3030}));
};
const sets = {A: [pts(D.A.pos, D.A.col)], B: [pts(D.B.pos, D.B.col)], diff: [pts(D.B.pos, D.B.gray), pts(D.ghost.pos, null, 0.03)]};
const COLOR = {added: 0x30e050, removed: 0xff3030, moved: 0x3090ff};
const overlay = new THREE.Group(); scene.add(overlay);
const box = (lo, hi, c) => new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi)), c);
const mid = (lo, hi) => lo.map((x, i) => (x + hi[i]) / 2);
const labels = [];
D.changes.forEach((c, i) => {
  overlay.add(box(c.lo, c.hi, COLOR[c.type]));
  if (c.from_lo) {
    overlay.add(box(c.from_lo, c.from_hi, COLOR[c.type]));
    overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(...mid(c.from_lo, c.from_hi)), new THREE.Vector3(...mid(c.lo, c.hi))]), new THREE.LineBasicMaterial({color: 0xffff00})));
  }
  const el = document.createElement('div'); el.className = 'lb'; el.textContent = c.text.split(':')[1].split(' in ')[0].split(',')[0];
  el.style.color = '#' + COLOR[c.type].toString(16); document.body.append(el);
  labels.push([el, new THREE.Vector3(c.lo[0], c.lo[1], c.hi[2] + 0.1).lerp(new THREE.Vector3(c.hi[0], c.hi[1], c.hi[2] + 0.1), 0.5)]);
  const d = document.createElement('div'); d.className = 'c'; d.textContent = c.text;
  d.onclick = () => { const m = mid(c.lo, c.hi); ctl.target.set(...m); cam.position.set(m[0] + 1.5, m[1] - 1.5, m[2] + 1.2); };
  document.getElementById('list').append(d);
});
if (!D.changes.length) document.getElementById('list').textContent = 'No changes detected.';
const show = m => { for (const k in sets) sets[k].forEach(p => (p.parent && scene.remove(p))); sets[m].forEach(p => scene.add(p)); };
document.querySelectorAll('input[name=m]').forEach(i => i.onchange = () => show(i.value)); show('diff');
const g = new THREE.Box3().setFromBufferAttribute(sets.B[0].geometry.attributes.position), c0 = g.getCenter(new THREE.Vector3());
ctl.target.copy(c0); cam.position.set(c0.x + 3, c0.y - 3, c0.z + 3);
(function loop() {
  requestAnimationFrame(loop); ctl.update(); r.render(scene, cam);
  labels.forEach(([el, p]) => { const q = p.clone().project(cam); const on = q.z < 1;
    el.style.display = on ? '' : 'none'; el.style.left = (q.x + 1) / 2 * innerWidth + 'px'; el.style.top = (1 - q.y) / 2 * innerHeight + 'px'; });
})();
</script>"""


def _b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def write_viewer(path, data):
    f = lambda d, keys: {k: _b64(d[k], np.float32 if k == "pos" else np.uint8) for k in keys}
    payload = {"A": f(data["A"], ["pos", "col"]), "B": f(data["B"], ["pos", "col", "gray"]),
               "ghost": f(data["ghost"], ["pos"]), "changes": data["changes"]}
    Path(path).write_text(HTML.replace("__DATA__", json.dumps(payload)))

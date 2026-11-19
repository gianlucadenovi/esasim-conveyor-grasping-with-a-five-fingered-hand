// ── EsaSim block ──────────────────────────────────────────────────────
// Conveyor grasping view (ws-conveyor-grasping-view) · version 2 · view, JavaScript: function render(ctx)
// The grasping cell in 3D (belt, bin, camera, object, five-fingered hand and the xArm posed by inverse kinematics), the camera's picture with the network's box, the touch sensors, and the test's object
// Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
// Saved: 2026-11-19
//
// Inputs   ctx.input("id") gives { files, stream }; ctx.rows("id"), ctx.json("id"), ctx.text("id")
//   live        Stream                    Live · The Conveyor grasping block's telemetry (robot and events channels), straight or through a Telemetry block
//   arm         Files (gz)                Arm model · optional · xarm6.robot.json.gz from the xArm model block (or the same file in the project's assets)
// Settings ctx.params.key
//   robot_channel text      Robot channel = "robot"
//   events_channel text      Events channel = "events"
//   control_channel text      Controls channel = "control"
// EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
// ── end of EsaSim block header ────────────────────────────────────────

/* Conveyor grasping view: the cell in 3D (the belt, the bin, the camera on its pole, the object, the five-fingered hand
 * and the xArm carrying it, posed by inverse kinematics on the hand's pose), what the camera sees with the box the
 * network found, the touch sensors of the fingertips and the palm, and, in the test, the choice of the object and how
 * it lies on the belt.
 * Inputs: live (telemetry: the robot and events channels), arm (the xArm model, xarm6.robot.json.gz; optional). */
function render(ctx) {
  var THREE = ctx.THREE, A = THREE.addons || {}, el = ctx.el, P = ctx.params || {};
  var CH_BOT = P.robot_channel || 'robot', CH_EV = P.events_channel || 'events', CH_CTL = P.control_channel || 'control';
  el.innerHTML = '';
  var css = document.createElement('style');
  css.textContent = [
    '.gc{position:absolute;inset:0;display:flex;background:#0c0f14;color:#e6ecf5;font:13px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow:hidden}',
    '.gc .stage{position:relative;flex:1;min-width:0}',
    '.gc canvas.gl{position:absolute;inset:0;width:100%;height:100%;display:block}',
    '.gc .title{position:absolute;left:14px;top:12px;pointer-events:none;text-shadow:0 1px 3px #000;max-width:70%}',
    '.gc .title b{font-size:17px;font-weight:650}',
    '.gc .badge{display:inline-block;margin-left:8px;padding:1px 8px;border-radius:9px;font-size:12px;font-weight:600;vertical-align:2px}',
    '.gc .ok{background:#166534;color:#dcfce7}.gc .bad{background:#7f1d1d;color:#fee2e2}.gc .run{background:#1e3a8a;color:#dbeafe}',
    '.gc .bar{position:absolute;left:12px;bottom:12px;display:flex;gap:6px;flex-wrap:wrap}',
    '.gc button,.gc select{background:#161c26;color:#e6ecf5;border:1px solid rgba(255,255,255,.15);border-radius:7px;padding:4px 9px;font:inherit;cursor:pointer}',
    '.gc button.on{background:#0e7490;border-color:#22d3ee}',
    '.gc button.go{background:#76b900;color:#0b1200;font-weight:700;border-color:#a3e635}',
    '.gc .side{width:340px;max-width:42%;border-left:1px solid rgba(255,255,255,.08);background:#0a0d12;display:flex;flex-direction:column;overflow:auto}',
    '.gc .sec{padding:11px 13px;border-bottom:1px solid rgba(255,255,255,.06)}',
    '.gc .sec h3{margin:0 0 8px;font-size:11.5px;font-weight:650;opacity:.75;text-transform:uppercase;letter-spacing:.5px;display:flex;justify-content:space-between}',
    '.gc .cam{width:100%;image-rendering:pixelated;border-radius:6px;background:#000;display:block}',
    '.gc .muted{opacity:.6}',
    '.gc .row{display:flex;align-items:center;gap:8px;margin:5px 0}',
    '.gc .row span{width:86px;opacity:.75;font-size:12px}',
    '.gc .row select,.gc .row input[type=range]{flex:1;min-width:0}',
    '.gc .row b{width:52px;text-align:right;font-variant-numeric:tabular-nums}',
    '.gc input[type=range]{accent-color:#facc15}',
    '.gc .ev{font-size:12px;margin:3px 0;opacity:.9}',
    '.gc .ev.okc{color:#86efac}.gc .ev.badc{color:#fca5a5}',
    '.gc .note{position:absolute;right:12px;bottom:14px;font-size:11.5px;opacity:.6;pointer-events:none}',
    '.gc .empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:30px;opacity:.7;pointer-events:none}',
    '.gc .stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;text-align:center}',
    '.gc .stats div{background:rgba(255,255,255,.04);border-radius:7px;padding:5px}',
    '.gc .stats b{display:block;font-size:18px;font-variant-numeric:tabular-nums}',
    '.gc .stats span{font-size:10.5px;opacity:.6;text-transform:uppercase;letter-spacing:.4px}'
  ].join('\n');
  el.appendChild(css);
  var root = document.createElement('div'); root.className = 'gc'; el.appendChild(root);
  root.innerHTML =
    '<div class="stage"><canvas class="gl"></canvas>' +
    '<div class="title"><b data-k="label">Conveyor grasping</b><span data-k="badge"></span><div class="muted" data-k="sub"></div></div>' +
    '<div class="bar"><button data-k="vCam">Camera view</button><button data-k="vClose">Close up</button><button data-k="vWide">Whole cell</button>' +
    '<button data-k="box" class="on">Camera box</button><button data-k="arm" class="on">xArm</button><button data-k="finish">Finish now</button></div>' +
    '<div class="note" data-k="armNote"></div>' +
    '<div class="empty" data-k="empty">Waiting for the cell: start a run of this flow.</div></div>' +
    '<div class="side">' +
    '<div class="sec" data-k="testSec" style="display:none"><h3>Test with a new object</h3>' +
    '<div class="row"><span>Object</span><select data-k="obj"></select></div>' +
    '<div class="row"><span>Lies</span><select data-k="pose"><option value="upright">upright</option><option value="side">on its side</option><option value="any">tumbled (any way)</option></select></div>' +
    '<div class="row"><span>Turned</span><input type="range" data-k="yaw" min="0" max="179" step="1" value="30"><b data-k="yawv"></b></div>' +
    '<div class="row"><span>Belt speed</span><input type="range" data-k="spd" min="2" max="8" step="0.5" value="5"><b data-k="spdv"></b></div>' +
    '<div class="row"><button class="go" data-k="drop" style="flex:1">Put it on the belt</button></div>' +
    '<div class="stats"><div><b data-k="nT">0</b><span>trials</span></div><div><b data-k="nG">0</b><span>grasped</span></div><div><b data-k="nP">0</b><span>in the bin</span></div></div></div>' +
    '<div class="sec" data-k="camSec" style="display:none"><h3>What the camera sees <span class="muted" data-k="camNote"></span></h3><canvas class="cam" data-k="cam" width="96" height="72"></canvas>' +
    '<div class="muted" style="font-size:11.5px;margin-top:4px" data-k="camLab">The network’s box of the object in yellow.</div></div>' +
    '<div class="sec"><h3>Touch <span class="muted">N</span></h3><canvas data-k="touch" width="300" height="190" style="width:100%"></canvas></div>' +
    '<div class="sec"><h3>Events</h3><div data-k="events" class="muted">—</div></div>' +
    '</div>';
  function $(k) { return root.querySelector('[data-k="' + k + '"]'); }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // ---------------------------------------------------------------- three.js
  var canvas = root.querySelector('canvas.gl');
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1f27);
  try { if (A.RoomEnvironment) { var pm = new THREE.PMREMGenerator(renderer); scene.environment = pm.fromScene(new A.RoomEnvironment(), 0.04).texture; } } catch (e) { }
  var camera = new THREE.PerspectiveCamera(42, 1, 0.02, 30);
  camera.up.set(0, 0, 1);
  camera.position.set(0.95, -0.75, 0.95);
  var controls = null;
  if (A.OrbitControls) { controls = new A.OrbitControls(camera, canvas); controls.enableDamping = true; controls.dampingFactor = 0.08; controls.target.set(0.05, 0.25, 0.05); }
  else camera.lookAt(0.05, 0.25, 0.05);
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30302a, 0.55));
  var sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(0.8, -0.6, 2.2); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  var sc = sun.shadow.camera; sc.left = -1.2; sc.right = 1.2; sc.top = 1.2; sc.bottom = -1.2; sc.near = 0.5; sc.far = 5;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  var fill = new THREE.DirectionalLight(0xbfd4ff, 0.5); fill.position.set(-1, 1, 1); scene.add(fill);
  // the hall floor
  var floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshStandardMaterial({ color: 0x4a4f57, roughness: 0.85, metalness: 0.0 }));
  floor.position.z = -0.75; floor.receiveShadow = true; scene.add(floor);
  var grid = new THREE.GridHelper(8, 32, 0x5b6370, 0x535a66); grid.rotation.x = Math.PI / 2; grid.position.z = -0.749; scene.add(grid);

  function resize() {
    var r = root.querySelector('.stage').getBoundingClientRect();
    var w = Math.max(10, r.width), h = Math.max(10, r.height);
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }

  // ---------------------------------------------------------------- the cell, from the run's "cell" item
  var st = { cell: null, bodies: [], bodyGroups: [], jointGroups: [], objGroup: null, objKey: null, beltTex: null, beltBody: -1, handBody: -1,
    frames: [], playT: 0, playing: null, live: false, episodes: {}, epOrder: [], showBox: true, showArm: true, lastStreamLen: 0,
    camItem: null, events: [], trials: { n: 0, g: 0, p: 0 }, lastFrame: null, label: '', result: null, catalog: null };
  var cellRoot = new THREE.Group(); scene.add(cellRoot);
  var boxHelper = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), new THREE.LineBasicMaterial({ color: 0xfacc15 }));
  boxHelper.visible = false; scene.add(boxHelper);
  var Q = new THREE.Quaternion(), V = new THREE.Vector3();

  function mat(c, opts) {
    var m = new THREE.MeshStandardMaterial(Object.assign({ color: new THREE.Color(c[0], c[1], c[2]), roughness: 0.55, metalness: 0.15 }, opts || {}));
    if (c[3] < 0.999) { m.transparent = true; m.opacity = c[3]; }
    return m;
  }
  function geomMesh(t, s, material) {
    var g = null, rot = null;
    if (t === 'box') g = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    else if (t === 'sphere') g = new THREE.SphereGeometry(s[0], 28, 18);
    else if (t === 'ellipsoid') { g = new THREE.SphereGeometry(1, 28, 18); g.scale(s[0], s[1], s[2]); }
    else if (t === 'capsule') { g = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 20); rot = true; }
    else if (t === 'cylinder') { g = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32); rot = true; }
    if (!g) return null;
    if (rot) g.rotateX(Math.PI / 2);          // three.js: along y; MuJoCo: along z
    var m = new THREE.Mesh(g, material); m.castShadow = true; m.receiveShadow = true;
    return m;
  }
  function beltTexture() {
    var c = document.createElement('canvas'); c.width = 256; c.height = 64;
    var g = c.getContext('2d'); g.fillStyle = '#1c1d20'; g.fillRect(0, 0, 256, 64);
    for (var i = 0; i < 16; i++) { g.fillStyle = i % 2 ? '#232428' : '#18191c'; g.fillRect(i * 16, 0, 3, 64); }
    for (var k = 0; k < 400; k++) { g.fillStyle = 'rgba(255,255,255,' + (Math.random() * 0.05) + ')'; g.fillRect(Math.random() * 256, Math.random() * 64, 1, 1); }
    var t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace;
    return t;
  }
  function buildCell(cell) {
    while (cellRoot.children.length) cellRoot.remove(cellRoot.children[0]);
    st.cell = cell; st.bodyGroups = []; st.jointGroups = []; st.objGroup = null; st.objKey = null;
    cell.bodies.forEach(function (b, i) {
      var outer = new THREE.Group(), inner = new THREE.Group();
      outer.position.set(b.pos[0], b.pos[1], b.pos[2]);
      outer.quaternion.set(b.quat[1], b.quat[2], b.quat[3], b.quat[0]);
      outer.add(inner);
      st.bodyGroups.push(inner); st.jointGroups.push(outer);
      var parent = i === 0 ? cellRoot : st.bodyGroups[b.parent];
      parent.add(outer);
      if (b.name === 'belt') st.beltBody = i;
      if (b.name === 'hand') st.handBody = i;
      if (b.name === 'obj') st.objGroup = inner;
    });
    cell.geoms.forEach(function (g) {
      if (g.t === 'plane' || g.n === 'floor') return;
      var m;
      if (g.n === 'belt') {
        st.beltTex = beltTexture(); st.beltTex.repeat.set(2 * g.s[0] / 0.25, 1);
        m = geomMesh('box', g.s, new THREE.MeshStandardMaterial({ map: st.beltTex, roughness: 0.9, metalness: 0 }));
      } else {
        var metal = /pedestal|frame|rail|pole/.test(g.n) ? 0.6 : 0.12;
        m = geomMesh(g.t, g.s, mat(g.c, { metalness: metal, roughness: metal > 0.3 ? 0.35 : 0.5 }));
      }
      if (!m) return;
      m.position.set(g.p[0], g.p[1], g.p[2]); m.quaternion.set(g.q[1], g.q[2], g.q[3], g.q[0]);
      st.bodyGroups[g.b].add(m);
    });
    // the camera's housing on its pole
    var C = cell.camera, M = C.mat;
    var cam = new THREE.Group();
    var body = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.05, 0.06), new THREE.MeshStandardMaterial({ color: 0x22252b, roughness: 0.4, metalness: 0.5 }));
    body.position.z = 0.03; cam.add(body);
    var lens = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.018, 0.02, 24).rotateX(Math.PI / 2), new THREE.MeshStandardMaterial({ color: 0x0a0a0c, roughness: 0.1, metalness: 0.8 }));
    lens.position.z = -0.008; cam.add(lens);
    var m4 = new THREE.Matrix4().set(M[0], M[1], M[2], C.pos[0], M[3], M[4], M[5], C.pos[1], M[6], M[7], M[8], C.pos[2], 0, 0, 0, 1);
    cam.applyMatrix4(m4); cellRoot.add(cam);
    // the arm's adapter on the back of the hand (the flange of the xArm screws onto it)
    if (st.handBody >= 0) {
      var ad = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.03, 32), new THREE.MeshStandardMaterial({ color: 0x9aa3ad, roughness: 0.35, metalness: 0.7 }));
      ad.rotation.z = Math.PI / 2; ad.position.set(FLANGE_X + 0.015, 0, 0.002); st.bodyGroups[st.handBody].add(ad);
    }
    $('empty').style.display = 'none';
    if (cell.catalog) setupTest(cell.catalog);
  }
  var FLANGE_X = -0.118;     // the adapter's far face, on the hand's x axis (the hand's own wrist ends at -0.088)

  function buildObject(o) {
    if (!st.objGroup || !o) return;
    if (st.objKey === o.key + ':' + (o.parts || []).length) return;
    while (st.objGroup.children.length) st.objGroup.remove(st.objGroup.children[0]);
    var material = new THREE.MeshStandardMaterial({ color: new THREE.Color(o.rgba[0], o.rgba[1], o.rgba[2]), roughness: 0.45, metalness: 0.05 });
    o.parts.forEach(function (p) {
      var m = geomMesh(p.t, p.s, material); if (!m) return;
      m.position.set(p.p[0], p.p[1], p.p[2]); m.quaternion.set(p.q[1], p.q[2], p.q[3], p.q[0]);
      st.objGroup.add(m);
    });
    st.objKey = o.key + ':' + (o.parts || []).length;
    st.objLabel = o.label + (o.test ? ' (new to it)' : '');
  }

  // pose every body from a frame [t, qpos..., touch(9), box(7)]
  function applyFrame(f) {
    if (!st.cell || !f) return;
    var nq = st.cell.nq;
    st.cell.bodies.forEach(function (b, i) {
      var j = b.joint, g = st.bodyGroups[i];
      if (!j) return;
      if (j.type === 'free') {
        g.position.set(f[1 + j.q], f[2 + j.q], f[3 + j.q]);
        g.quaternion.set(f[5 + j.q], f[6 + j.q], f[7 + j.q], f[4 + j.q]);
        // (a free body's frame is the world: undo the outer offset)
        st.jointGroups[i].position.set(0, 0, 0); st.jointGroups[i].quaternion.identity();
      } else if (j.type === 'hinge') {
        Q.setFromAxisAngle(V.set(j.axis[0], j.axis[1], j.axis[2]), f[1 + j.q]); g.quaternion.copy(Q);
      } else {
        var q = f[1 + j.q]; g.position.set(j.axis[0] * q, j.axis[1] * q, j.axis[2] * q);
      }
    });
    if (st.beltTex && st.beltBody >= 0) {   // the belt's surface runs; the slab stays put
      var bq = f[1 + st.cell.bodies[st.beltBody].joint.q];
      st.bodyGroups[st.beltBody].position.set(0, 0, 0);
      st.beltTex.offset.x = -bq / 0.25;
    }
    var k = 1 + nq;
    st.touch = f.slice(k, k + 9);
    var bx = f.slice(k + 9, k + 16);
    if (bx.length === 7 && st.showBox) {
      boxHelper.visible = true;
      boxHelper.position.set(bx[0], bx[1], bx[2]);
      boxHelper.scale.set(Math.max(0.005, 2 * bx[3]), Math.max(0.005, 2 * bx[4]), Math.max(0.005, 2 * bx[5]));
      boxHelper.rotation.set(0, 0, bx[6]);
    } else boxHelper.visible = false;
    st.lastFrame = f;
    poseArm();
  }

  // ---------------------------------------------------------------- the xArm, posed by inverse kinematics on the hand
  var arm = { model: null, links: [], joints: [], q: [0, -0.5, -1.0, 0, 1.2, 0], group: new THREE.Group() };
  scene.add(arm.group);
  function rpyMat(r, p, y) { return new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(r, p, y, 'ZYX')); }
  function loadArm(model) {
    arm.model = model;
    var s = (model.scale || 1) * (model.units === 'mm' ? 0.001 : 1);
    arm.joints = model.joints.map(function (j) {
      var u = model.units === 'mm' ? 0.001 : 1;
      return { T: new THREE.Matrix4().makeTranslation(j.xyz[0] * u, j.xyz[1] * u, j.xyz[2] * u).multiply(rpyMat(j.rpy[0], j.rpy[1], j.rpy[2])),
        lo: (j.limits ? j.limits[0] : -360) * Math.PI / 180, hi: (j.limits ? j.limits[1] : 360) * Math.PI / 180 };
    });
    var white = new THREE.MeshStandardMaterial({ color: 0xe9ebee, roughness: 0.4, metalness: 0.15 });
    var dark = new THREE.MeshStandardMaterial({ color: 0x2a2d33, roughness: 0.5, metalness: 0.3 });
    arm.links = model.links.map(function (l, i) {
      var g = new THREE.BufferGeometry();
      var v = new Float32Array(l.v.length); for (var k = 0; k < l.v.length; k++) v[k] = l.v[k] * s;
      g.setAttribute('position', new THREE.BufferAttribute(v, 3)); g.setIndex(l.f); g.computeVertexNormals();
      var m = new THREE.Mesh(g, i === 0 ? dark : white); m.castShadow = true; m.receiveShadow = true; m.matrixAutoUpdate = false;
      arm.group.add(m); return m;
    });
    poseArm();
  }
  function fk(q, out) {       // out[i]: the frame of link i (0: the base)
    var T = new THREE.Matrix4(); out[0] = T.clone();
    for (var i = 0; i < arm.joints.length; i++) {
      T = T.clone().multiply(arm.joints[i].T).multiply(new THREE.Matrix4().makeRotationZ(q[i]));
      out[i + 1] = T;
    }
    return T;
  }
  var _m = new THREE.Matrix4(), _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3();
  function flangeTarget() {
    if (st.handBody < 0) return null;
    var h = st.bodyGroups[st.handBody]; h.updateWorldMatrix(true, false);
    var R = new THREE.Matrix4().extractRotation(h.matrixWorld);
    var hx = new THREE.Vector3().setFromMatrixColumn(R, 0), hy = new THREE.Vector3().setFromMatrixColumn(R, 1), hz = new THREE.Vector3().setFromMatrixColumn(R, 2);
    var pos = new THREE.Vector3(FLANGE_X, 0, 0.002).applyMatrix4(h.matrixWorld);
    // the flange faces the hand: its z along the hand's x (the fingers carry on from the arm), its x along the back of the hand
    var Rt = new THREE.Matrix4().makeBasis(hz, hy.clone().negate(), hx);
    return { p: pos, R: Rt };
  }
  function err6(T, tgt) {
    _p.setFromMatrixPosition(T);
    var e = [tgt.p.x - _p.x, tgt.p.y - _p.y, tgt.p.z - _p.z];
    var Rc = new THREE.Matrix4().extractRotation(T);
    var Re = tgt.R.clone().multiply(Rc.clone().transpose());       // R_t R_c^T
    var m = Re.elements;     // column-major
    var w = [(m[6] - m[9]) * 0.5, (m[8] - m[2]) * 0.5, (m[1] - m[4]) * 0.5];
    return e.concat(w.map(function (x) { return x * 0.25; }));
  }
  var SEEDS = [[0, -0.5, -1.0, 0, 1.2, 0], [1.5, -0.3, -1.2, 0, 1.3, 0], [-1.5, -0.3, -1.2, 0, 1.3, 0], [2.5, 0, -1, 0, 1, 0]];
  function solveIK(tgt) {
    var best = ikFrom(arm.q.slice(), tgt);
    for (var i = 0; best.e > 0.003 && i < SEEDS.length; i++) { var r = ikFrom(SEEDS[i].slice(), tgt); if (r.e < best.e) best = r; }
    arm.q = best.q;
    return best.frames;
  }
  function ikFrom(q, tgt) {
    var n = q.length, frames = [], en = 1e9;
    for (var it = 0; it < 40; it++) {
      var e = err6(fk(q, frames), tgt);
      en = Math.hypot.apply(null, e);
      if (en < 1e-4) break;
      var J = [];
      for (var j = 0; j < n; j++) {
        var q2 = q.slice(); q2[j] += 1e-4;
        var e2 = err6(fk(q2, []), tgt);
        J.push(e.map(function (x, k) { return (x - e2[k]) / 1e-4; }));     // d(pose)/dq
      }
      // dq = J^T (J J^T + λ² I)^-1 e
      var lam = 0.02, JJ = [];
      for (var a = 0; a < 6; a++) { JJ.push([]); for (var b = 0; b < 6; b++) { var s = a === b ? lam * lam : 0; for (var k2 = 0; k2 < n; k2++) s += J[k2][a] * J[k2][b]; JJ[a].push(s); } }
      var y = solve6(JJ, e); if (!y) break;
      for (var j2 = 0; j2 < n; j2++) {
        var d = 0; for (var a2 = 0; a2 < 6; a2++) d += J[j2][a2] * y[a2];
        d = Math.max(-0.3, Math.min(0.3, d));
        q[j2] = Math.max(arm.joints[j2].lo, Math.min(arm.joints[j2].hi, q[j2] + d));
      }
    }
    frames = []; var ef = err6(fk(q, frames), tgt);
    return { q: q, frames: frames, e: Math.hypot(ef[0], ef[1], ef[2]) };
  }
  function solve6(M, b) {
    var n = 6, A2 = M.map(function (r, i) { return r.concat([b[i]]); });
    for (var c = 0; c < n; c++) {
      var p = c; for (var r = c + 1; r < n; r++) if (Math.abs(A2[r][c]) > Math.abs(A2[p][c])) p = r;
      if (Math.abs(A2[p][c]) < 1e-12) return null;
      var t = A2[c]; A2[c] = A2[p]; A2[p] = t;
      for (var r2 = 0; r2 < n; r2++) if (r2 !== c) { var f = A2[r2][c] / A2[c][c]; for (var k = c; k <= n; k++) A2[r2][k] -= f * A2[c][k]; }
    }
    return A2.map(function (r, i) { return r[n] / r[i]; });
  }
  function poseArm() {
    if (!arm.model || !arm.links.length) return;
    arm.group.visible = st.showArm;
    if (!st.showArm) return;
    var tgt = flangeTarget(); if (!tgt) return;
    var frames = solveIK(tgt);
    for (var i = 0; i < arm.links.length && i < frames.length; i++) { arm.links[i].matrix.copy(frames[i]); }
  }
  function gunzip(buf) {
    var u = new Uint8Array(buf);
    if (u[0] !== 0x1f || u[1] !== 0x8b) return Promise.resolve(buf);
    return new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  // the arm's model: its file may still be on its way from the worker, so it is asked for again until it comes
  var armTry = { busy: false, n: 0, timer: null };
  function armNote(t) { var e = $('armNote'); if (e) e.textContent = t || ''; }
  function timed(p, ms) {
    return new Promise(function (res, rej) {
      var tm = setTimeout(function () { rej(new Error('timed out')); }, ms);
      Promise.resolve(p).then(function (v) { clearTimeout(tm); res(v); }, function (e) { clearTimeout(tm); rej(e); });
    });
  }
  function findArm() {
    if (arm.model || armTry.busy) return;
    var fs = ((ctx.input('arm') || {}).files || []).filter(function (f) { return /\.robot\.json(\.gz)?$/i.test(f.name || f.path || ''); });
    var p;
    if (fs.length) p = ctx.files.buffer(fs[0]);
    else if (ctx.assets && ctx.assets.find) p = Promise.resolve(ctx.assets.find(['xarm6.robot.json.gz', 'models/xarm6.robot.json.gz'])).then(function (a) { return a ? ctx.assets.buffer(a) : null; });
    else p = Promise.resolve(null);
    if (!fs.length) armNote('xArm: wire the xArm model block into Arm model');
    else armNote('xArm: loading its model…');
    armTry.busy = true; armTry.n++;
    timed(p, 45000).then(function (b) { return b ? gunzip(b) : null; }).then(function (b) {
      armTry.busy = false;
      if (!b) return;
      loadArm(JSON.parse(new TextDecoder().decode(b)));
      armNote('');
    }).catch(function (e) {
      armTry.busy = false;
      armNote('xArm: ' + (e && e.message || e) + ' (trying again)');
      if (armTry.n < 40) { clearTimeout(armTry.timer); armTry.timer = setTimeout(findArm, 5000); }
    });
  }

  // ---------------------------------------------------------------- the camera's picture and the touch panel
  function b64bytes(s) { var bin = atob(s), u = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  function drawCam(item) {
    var c = $('cam'), w = item.w, h = item.h, g = c.getContext('2d');
    c.width = w * 3; c.height = h * 3;
    var raw = b64bytes(item.rgb), img = g.createImageData(w, h);
    for (var i = 0, j = 0; i < w * h; i++, j += 3) { img.data[i * 4] = raw[j]; img.data[i * 4 + 1] = raw[j + 1]; img.data[i * 4 + 2] = raw[j + 2]; img.data[i * 4 + 3] = 255; }
    var tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h; tmp.getContext('2d').putImageData(img, 0, 0);
    g.imageSmoothingEnabled = false; g.drawImage(tmp, 0, 0, w * 3, h * 3);
    if (item.box && st.cell) {
      var C = st.cell.camera, M = C.mat, f = 0.5 * h / Math.tan(C.fovy * Math.PI / 360) * 3;
      var b = item.box, cz = Math.cos(b.yaw), sz = Math.sin(b.yaw), pts = [];
      for (var k = 0; k < 8; k++) {
        var lx = (k & 1 ? 1 : -1) * b.e[0], ly = (k & 2 ? 1 : -1) * b.e[1], lz = (k & 4 ? 1 : -1) * b.e[2];
        var wx = b.c[0] + cz * lx - sz * ly - C.pos[0], wy = b.c[1] + sz * lx + cz * ly - C.pos[1], wz = b.c[2] + lz - C.pos[2];
        var xc = M[0] * wx + M[3] * wy + M[6] * wz, yc = M[1] * wx + M[4] * wy + M[7] * wz, zc = M[2] * wx + M[5] * wy + M[8] * wz;
        pts.push([w * 1.5 + f * xc / -zc, h * 1.5 - f * yc / -zc]);
      }
      g.strokeStyle = '#facc15'; g.lineWidth = 1.5; g.beginPath();
      [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (e) { g.moveTo(pts[e[0]][0], pts[e[0]][1]); g.lineTo(pts[e[1]][0], pts[e[1]][1]); });
      g.stroke();
    }
    if (item.label) $('camLab').textContent = item.label;
  }
  var TOUCH_POS = [[0.17, 0.80, 'thumb'], [0.33, 0.12, 'index'], [0.50, 0.08, 'middle'], [0.66, 0.12, 'ring'], [0.80, 0.22, 'little'],
    [0.40, 0.42, 'palm'], [0.62, 0.42, ''], [0.40, 0.66, ''], [0.62, 0.66, '']];
  function drawTouch() {
    var c = $('touch'), g = c.getContext('2d'), W = c.width, H = c.height, f = st.touch || [];
    g.clearRect(0, 0, W, H);
    // a hand seen from the palm: the palm, four fingers, the thumb
    g.fillStyle = 'rgba(255,255,255,0.05)'; g.strokeStyle = 'rgba(255,255,255,0.18)'; g.lineWidth = 1;
    g.beginPath(); g.roundRect ? g.roundRect(W * 0.28, H * 0.30, W * 0.46, H * 0.50, 14) : g.rect(W * 0.28, H * 0.30, W * 0.46, H * 0.50); g.fill(); g.stroke();
    for (var i = 0; i < 9; i++) {
      var x = TOUCH_POS[i][0] * W, y = TOUCH_POS[i][1] * H, v = f[i] || 0, a = Math.min(1, Math.log1p(v) / Math.log1p(15));
      if (i >= 1 && i <= 4) { g.strokeStyle = 'rgba(255,255,255,0.12)'; g.lineWidth = 12; g.beginPath(); g.moveTo(x, y); g.lineTo(x, H * 0.32); g.stroke(); }
      if (i === 0) { g.strokeStyle = 'rgba(255,255,255,0.12)'; g.lineWidth = 12; g.beginPath(); g.moveTo(x, y); g.lineTo(W * 0.30, H * 0.62); g.stroke(); }
      var r = i < 5 ? 12 : 17;
      g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI);
      g.fillStyle = 'rgba(' + Math.round(60 + 195 * a) + ',' + Math.round(80 + 120 * a * (1 - a)) + ',' + Math.round(160 * (1 - a)) + ',' + (0.35 + 0.65 * a) + ')';
      g.fill();
      g.fillStyle = '#fff'; g.font = '11px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle';
      g.fillText(v >= 0.05 ? (v < 10 ? v.toFixed(1) : v.toFixed(0)) : '', x, y);
    }
    g.fillStyle = 'rgba(255,255,255,0.45)'; g.font = '10.5px system-ui'; g.textAlign = 'left';
    g.fillText('thumb', 6, H * 0.94); g.textAlign = 'right'; g.fillText('fingertips above, palm in the middle', W - 6, H * 0.94);
  }

  // ---------------------------------------------------------------- what comes in
  function stream() { return ((ctx.input('live') || {}).stream || []); }
  function ingest() {
    var s = stream();
    if (s.length < st.lastStreamLen) { st.lastStreamLen = 0; }     // a new run
    for (var i = st.lastStreamLen; i < s.length; i++) {
      var it = s[i], d = it && it.data;
      if (!d) continue;
      if (it.channel === CH_EV || (d.text && !d.format)) { if (d.text) addEvent(d); continue; }
      if (d.format !== 'esasim.grasp/1') continue;
      if (d.kind === 'cell') { if (!st.cell || (d.catalog && !st.catalog) || d.bodies.length !== st.cell.bodies.length) buildCell(d); }
      else if (d.kind === 'camera') { st.camItem = d; $('camSec').style.display = ''; drawCam(d); }
      else if (d.kind === 'object') { buildObject(d.object); st.live = true; st.frames = []; st.playT = 0; st.label = 'Trial ' + d.trial + ': ' + d.object.label + (d.object.test ? ' (new to it)' : ''); st.result = null; }
      else if (d.kind === 'live') {
        st.live = true;
        if (!st.frames.length || d.trial !== st.liveTrial) { st.liveTrial = d.trial; }
        Array.prototype.push.apply(st.frames, d.frames || []);
        if (st.frames.length > 4000) st.frames.splice(0, st.frames.length - 4000);
        if (d.label) st.label = d.label;
      } else if (d.kind === 'training') {
        var ep = st.episodes[d.ep] || (st.episodes[d.ep] = { parts: {}, n: 0, last: false, object: null, label: d.label, result: d.result });
        ep.parts[d.seq] = d.frames; ep.n++; if (d.object) ep.object = d.object; if (d.last) ep.lastSeq = d.seq;
        if (ep.lastSeq != null && ep.n === ep.lastSeq + 1 && !ep.done) {
          ep.done = true; ep.frames = [];
          for (var k = 0; k <= ep.lastSeq; k++) Array.prototype.push.apply(ep.frames, ep.parts[k] || []);
          st.epOrder.push(d.ep);
          if (!st.live) startEpisode(ep);
        }
      }
    }
    st.lastStreamLen = s.length;
  }
  function startEpisode(ep) {
    if (!ep.object && ep.frames.length) return;
    buildObject(ep.object); st.playing = ep; st.frames = ep.frames; st.playT = ep.frames.length ? ep.frames[0][0] : 0;
    st.label = ep.label; st.result = ep.result;
  }
  function addEvent(d) {
    st.events.push(d); if (st.events.length > 60) st.events.shift();
    if (d.trial && d.result) {
      st.trials.n = Math.max(st.trials.n, d.trial); if (d.grasped) st.trials.g++; if (d.placed) st.trials.p++;
      st.result = { success: d.placed || d.result === 'grasped and held', fail: !d.placed, text: d.result };
      $('nT').textContent = st.trials.n; $('nG').textContent = st.trials.g; $('nP').textContent = st.trials.p;
    }
    $('events').innerHTML = st.events.slice(-12).reverse().map(function (e) {
      var c = e.placed ? 'okc' : (e.result ? 'badc' : '');
      return '<div class="ev ' + c + '">' + esc(e.text) + '</div>';
    }).join('');
  }

  // ---------------------------------------------------------------- the test's controls
  function setupTest(catalog) {
    if (st.catalog) return;
    st.catalog = catalog; $('testSec').style.display = '';
    var news = catalog.filter(function (o) { return o.test; }), olds = catalog.filter(function (o) { return !o.test; });
    $('obj').innerHTML = '<optgroup label="New to it">' + news.map(function (o) { return '<option value="' + esc(o.key) + '">' + esc(o.label) + '</option>'; }).join('') + '</optgroup>' +
      '<optgroup label="Seen in training">' + olds.map(function (o) { return '<option value="' + esc(o.key) + '">' + esc(o.label) + '</option>'; }).join('') + '</optgroup>';
  }
  function sliders() { $('yawv').textContent = $('yaw').value + '°'; $('spdv').textContent = $('spd').value + ' cm/s'; }
  $('yaw').oninput = sliders; $('spd').oninput = sliders; sliders();
  $('drop').onclick = function () {
    var v = { object: $('obj').value, pose: $('pose').value, yaw: +$('yaw').value, speed: +$('spd').value / 100 };
    $('drop').disabled = true; setTimeout(function () { $('drop').disabled = false; }, 1500);
    Promise.resolve(ctx.send && ctx.send(CH_CTL, { control: 'drop', value: v })).catch(function (e) { addEvent({ text: 'Could not send: ' + (e && e.message || e) }); });
    addEvent({ text: 'Asked: ' + $('obj').selectedOptions[0].textContent + ', ' + $('pose').selectedOptions[0].textContent + ', turned ' + v.yaw + '°' });
  };
  $('finish').onclick = function () { if (ctx.send) ctx.send(CH_CTL, { control: 'Finish now' }); addEvent({ text: 'Asked to finish now' }); };
  $('box').onclick = function () { st.showBox = !st.showBox; this.classList.toggle('on', st.showBox); if (!st.showBox) boxHelper.visible = false; };
  $('arm').onclick = function () { st.showArm = !st.showArm; this.classList.toggle('on', st.showArm); arm.group.visible = st.showArm; poseArm(); };
  function view(p, t) { camera.position.set(p[0], p[1], p[2]); if (controls) { controls.target.set(t[0], t[1], t[2]); controls.update(); } else camera.lookAt(t[0], t[1], t[2]); }
  $('vCam').onclick = function () { if (!st.cell) return; var C = st.cell.camera, M = C.mat; view([C.pos[0] - M[2] * 0.05, C.pos[1] - M[5] * 0.05, C.pos[2] - M[8] * 0.05], [C.pos[0] - M[2], C.pos[1] - M[5], C.pos[2] - M[8]]); };
  $('vClose').onclick = function () { var h = st.handBody >= 0 ? new THREE.Vector3().setFromMatrixPosition(st.bodyGroups[st.handBody].matrixWorld) : new THREE.Vector3(0.1, 0.3, 0.1); view([h.x + 0.35, h.y - 0.35, h.z + 0.22], [h.x, h.y + 0.05, h.z - 0.05]); };
  $('vWide').onclick = function () { view([0.95, -0.75, 0.95], [0.05, 0.25, 0.05]); };

  // ---------------------------------------------------------------- playing the frames
  var lastNow = performance.now();
  function frameAt(t) {
    var F = st.frames; if (!F.length) return null;
    var lo = 0, hi = F.length - 1;
    if (t <= F[0][0]) return F[0];
    if (t >= F[hi][0]) return F[hi];
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (F[m][0] <= t) lo = m; else hi = m; }
    return F[lo];
  }
  function tick(now) {
    if (!el.isConnected) return;
    requestAnimationFrame(tick);
    var dt = Math.min(0.1, (now - lastNow) / 1000); lastNow = now;
    var F = st.frames;
    if (F.length) {
      if (st.live) {
        var end = F[F.length - 1][0];
        st.playT = Math.max(st.playT + dt, end - 1.0);        // a little behind the live end, smooth
        if (st.playT > end) st.playT = end;
      } else {
        st.playT += dt;
        if (st.playT > F[F.length - 1][0] + 1.5) {             // the newest replay, again
          var last = st.epOrder.length ? st.episodes[st.epOrder[st.epOrder.length - 1]] : null;
          if (last && last !== st.playing) startEpisode(last); else st.playT = F[0][0];
        }
      }
      var f = frameAt(st.playT);
      if (f !== st.lastFrame) { applyFrame(f); drawTouch(); }
    }
    $('label').textContent = st.label || 'Conveyor grasping';
    var r = st.result;
    $('badge').innerHTML = r ? '<span class="badge ' + (r.success ? 'ok' : 'bad') + '">' + esc(r.text || (r.success ? 'success' : (r.fail ? 'failed' : 'out of time'))) + '</span>' : (F.length ? '<span class="badge run">' + (st.live ? 'live' : 'replay') + '</span>' : '');
    $('sub').textContent = F.length ? 't = ' + (st.playT || 0).toFixed(1) + ' s' + (st.objLabel ? ' · ' + st.objLabel : '') : '';
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  ctx.on('update', function () { ingest(); findArm(); });
  ctx.on('resize', resize);
  resize(); drawTouch(); ingest(); findArm();
  requestAnimationFrame(tick);
  setTimeout(resize, 60);
}

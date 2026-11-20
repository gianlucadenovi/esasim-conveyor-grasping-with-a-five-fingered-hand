// ── EsaSim block ──────────────────────────────────────────────────────
// Hand pressure map (ws-hand-pressure-map) · version 1 · view, JavaScript: function render(ctx)
// A five-fingered hand seen from the palm: each touch sensor where it is on the hand, coloured from black to white by its pressure, with its reading.
// Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
// Saved: 2026-11-20
//
// Inputs   ctx.input("id") gives { files, stream }; ctx.rows("id"), ctx.json("id"), ctx.text("id")
//   data        Stream                    Sensors · several · Items with one field per sensor (N)
// Settings ctx.params.key
//   channel     text      Channel = "touch"
//   max         number    Highest pressure (N) = 30
//   scale       select    Colour scale = "log" of log | linear
//   hand        select    Hand = "right" of right | left
// EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
// ── end of EsaSim block header ────────────────────────────────────────

/* Hand pressure map: a five-fingered hand seen from the palm, each touch sensor drawn where it is on the hand and coloured by
   how hard it is pressed, from black (nothing) through blue, green, yellow and red to white (the most), with its reading. */
function render(ctx) {
  var p = ctx.params;
  var chan = String(p.channel || 'touch').trim();
  var hi = +p.max > 0 ? +p.max : 30;
  var logScale = String(p.scale || 'log').toLowerCase().indexOf('lin') !== 0;
  var mirror = String(p.hand || 'right').toLowerCase().indexOf('left') === 0;
  var DIG = ['thumb', 'index', 'middle', 'ring', 'little'];

  // the colour scale: black, blue, cyan, green, yellow, red, white
  var STOPS = [[0, 0, 0, 0], [0.18, 20, 50, 230], [0.34, 0, 190, 230], [0.5, 30, 200, 60], [0.66, 245, 225, 30], [0.82, 235, 40, 30], [1, 255, 255, 255]];
  function frac(v) {
    if (!(v > 0)) return 0;
    return Math.max(0, Math.min(1, logScale ? Math.log(1 + v) / Math.log(1 + hi) : v / hi));
  }
  function colour(f) {
    for (var i = 1; i < STOPS.length; i++) {
      if (f <= STOPS[i][0]) {
        var a = STOPS[i - 1], b = STOPS[i], t = (f - a[0]) / (b[0] - a[0]);
        return 'rgb(' + [1, 2, 3].map(function (k) { return Math.round(a[k] + (b[k] - a[k]) * t); }).join(',') + ')';
      }
    }
    return 'rgb(255,255,255)';
  }
  function ink(f) { return (f > 0.42 && f < 0.75) || f > 0.9 ? '#111' : '#f4f4f4'; }

  // the hand's layout (in a 300 x 330 box, palm side, fingers up): every sensor a rounded shape
  var S = [];      // {name, x, y, w, h, rot, cx, cy}
  function seg(name, x, y, w, h, rot) { S.push({ name: name, x: x, y: y, w: w, h: h, rot: rot || 0 }); }
  var fingers = [  // x of the finger's centre, the length of its three phalanges (prox, mid, tip), its width
    ['index', 112, [46, 30, 26], 26], ['middle', 146, [50, 33, 27], 27], ['ring', 180, [47, 31, 26], 26], ['little', 212, [38, 24, 22], 23]];
  var palmTop = 160, palmL = 96, palmR = 228, palmBot = 290;
  fingers.forEach(function (f) {
    var y = palmTop - 4 + (f[0] === 'little' ? 14 : f[0] === 'index' ? 4 : f[0] === 'ring' ? 2 : 0), w = f[3];
    ['prox', 'mid', 'tip'].forEach(function (s, k) {
      var L = f[2][k];
      y -= L + 3;
      seg(f[0] + '_' + s, f[1] - w / 2, y, w, L, 0);
    });
  });
  // the palm's four zones: palm_1 fingers' side by the index, palm_2 fingers' side by the little finger,
  // palm_3 wrist side by the index, palm_4 wrist side by the little finger
  var midX = (palmL + palmR) / 2, midY = (palmTop + palmBot) / 2;
  seg('palm_1', palmL, palmTop, midX - palmL - 2, midY - palmTop - 2);
  seg('palm_2', midX + 2, palmTop, palmR - midX - 2, midY - palmTop - 2);
  seg('palm_3', palmL, midY + 2, midX - palmL - 2, palmBot - midY - 2);
  seg('palm_4', midX + 2, midY + 2, palmR - midX - 2, palmBot - midY - 2);
  // the thumb, from the wrist side of the palm by the index, angled out
  (function () {
    var bx = palmL - 6, by = palmBot - 40, ang = -40, w = 28, lens = [44, 32, 28], d = 0;
    ['prox', 'mid', 'tip'].forEach(function (s, k) {
      var L = lens[k], a = ang * Math.PI / 180;
      var cx = bx + Math.sin(a) * (d + L / 2), cy = by - Math.cos(a) * (d + L / 2);
      seg('thumb_' + s, cx - w / 2, cy - L / 2, w, L, ang);
      d += L + 3;
    });
  })();
  S.forEach(function (s) { s.cx = s.x + s.w / 2; s.cy = s.y + s.h / 2; });

  var LABEL = { tip: 'tip', mid: 'middle', prox: 'first' };
  function title(n) {
    var m = n.match(/^(\w+?)_(tip|mid|prox)$/);
    if (m) return m[1] + ' ' + (m[1] === 'thumb' && m[2] === 'prox' ? 'metacarpal' : LABEL[m[2]] + (m[2] === 'tip' ? '' : ' phalanx'));
    var z = n.match(/^palm_(\d)$/);
    return z ? 'palm, ' + ['', 'fingers side by the index', 'fingers side by the little finger', 'wrist side by the index', 'wrist side by the little finger'][+z[1]] : n;
  }

  var shapes = S.map(function (s) {
    var r = Math.min(s.w, s.h) * (s.name.indexOf('palm') === 0 ? 0.18 : 0.45);
    return '<g class="sn" data-n="' + s.name + '" transform="rotate(' + s.rot + ' ' + s.cx + ' ' + s.cy + ')">'
      + '<title>' + title(s.name) + '</title>'
      + '<rect x="' + s.x + '" y="' + s.y + '" width="' + s.w + '" height="' + s.h + '" rx="' + r + '" ry="' + r + '" fill="#000" stroke="var(--line)" stroke-width="1.2"/>'
      + '<text x="' + s.cx + '" y="' + (s.cy + 3.5) + '" text-anchor="middle" font-size="10" font-weight="600" fill="#f4f4f4" style="font-variant-numeric:tabular-nums">–</text></g>';
  }).join('');
  // the colour bar
  var grad = STOPS.map(function (s) { return '<stop offset="' + s[0] + '" stop-color="rgb(' + s[1] + ',' + s[2] + ',' + s[3] + ')"/>'; }).join('');
  var ticks = (logScale ? [0, 1, 3, 10, hi] : [0, hi / 4, hi / 2, 3 * hi / 4, hi]).filter(function (v, i, a) { return v <= hi && a.indexOf(v) === i; });
  var bar = '<defs><linearGradient id="hm-g" x1="0" x2="0" y1="1" y2="0">' + grad + '</linearGradient></defs>'
    + '<rect x="262" y="40" width="12" height="240" rx="3" fill="url(#hm-g)" stroke="var(--line)"/>'
    + ticks.map(function (v) { var y = 280 - frac(v) * 240; return '<line x1="274" x2="279" y1="' + y + '" y2="' + y + '" stroke="var(--ink-3)"/><text x="282" y="' + (y + 3) + '" font-size="9" fill="var(--ink-3)">' + (v >= 10 ? Math.round(v) : +v.toFixed(1)) + '</text>'; }).join('')
    + '<text x="268" y="30" text-anchor="middle" font-size="9" fill="var(--ink-3)">N</text>';

  ctx.el.innerHTML = '<div class="viz-pad" style="height:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:6px">'
    + '<div style="display:flex;gap:14px;align-items:baseline;flex-wrap:wrap"><span class="hm-hold" style="font-weight:700;padding:2px 8px;border-radius:9px;background:var(--line)">–</span>'
    + '<span class="viz-muted viz-small">grip <b class="hm-grip" style="color:var(--ink)">–</b> N</span>'
    + '<span class="viz-muted viz-small">sensors pressed <b class="hm-n" style="color:var(--ink)">–</b> of ' + S.length + '</span></div>'
    + '<svg viewBox="0 0 305 335" style="flex:1;min-height:0;width:100%"><g transform="' + (mirror ? 'translate(250 0) scale(-1 1)' : '') + '">' + shapes + '</g>' + bar + '</svg></div>';
  if (mirror) ctx.el.querySelectorAll('.sn text').forEach(function (t) { t.setAttribute('transform', 'translate(' + (2 * +t.getAttribute('x')) + ' 0) scale(-1 1)'); });
  var nodes = {};
  ctx.el.querySelectorAll('.sn').forEach(function (g) { nodes[g.getAttribute('data-n')] = { rect: g.querySelector('rect'), text: g.querySelector('text') }; });

  function latestRow() {
    var row = null;
    ctx.ports().forEach(function (port) {
      (ctx.input(port).stream || []).forEach(function (it) {
        if ((!chan || it.channel === chan) && it.data && typeof it.data === 'object' && !Array.isArray(it.data)) row = it.data;
      });
    });
    return row;
  }
  function draw() {
    var row = latestRow();
    if (!row) { ctx.status(ctx.sample ? '' : 'Waiting for the hand\'s sensors (channel ' + chan + ')…'); return; }
    ctx.status('');
    var pressed = 0, total = 0;
    S.forEach(function (s) {
      var v = +row[s.name], nd = nodes[s.name];
      if (!isFinite(v)) { nd.rect.setAttribute('fill', '#000'); nd.text.textContent = '·'; return; }
      var f = frac(v);
      total += Math.max(0, v);
      if (v > 0.3) pressed++;
      nd.rect.setAttribute('fill', colour(f));
      nd.text.setAttribute('fill', ink(f));
      nd.text.textContent = v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(0) : v.toFixed(1);
    });
    var hold = row.holding;
    var hb = ctx.el.querySelector('.hm-hold');
    if (hold === undefined || hold === null) { hb.textContent = pressed ? 'touching' : 'not touching'; hb.style.background = 'var(--line)'; hb.style.color = 'var(--ink)'; }
    else { hb.textContent = +hold ? 'holding it' : (pressed ? 'touching' : 'empty'); hb.style.background = +hold ? 'var(--good, #1f9d55)' : 'var(--line)'; hb.style.color = +hold ? '#fff' : 'var(--ink)'; }
    ctx.el.querySelector('.hm-grip').textContent = (isFinite(+row.grip_n) ? +row.grip_n : total).toFixed(1);
    ctx.el.querySelector('.hm-n').textContent = pressed;
  }
  ctx.on('update', draw);
  draw();
}

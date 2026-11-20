// ── EsaSim block ──────────────────────────────────────────────────────
// Camera picture (ws-camera-picture) · version 1 · view, JavaScript: function render(ctx)
// The newest picture of a live camera, filling its panel (no margins, no grid), with how long ago it came.
// Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
// Saved: 2026-11-20
//
// Inputs   ctx.input("id") gives { files, stream }; ctx.rows("id"), ctx.json("id"), ctx.text("id")
//   images      Stream                    Pictures · several · Live items with an image (a data URL)
// Settings ctx.params.key
//   channel     text      Channel = "camera_picture"
//   pixels      select    Pixels = "smooth" of smooth | sharp
// EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
// ── end of EsaSim block header ────────────────────────────────────────

/* Camera picture: the newest live picture of a camera, filling its panel. */
function render(ctx) {
  var chan = String(ctx.params.channel == null ? 'camera_picture' : ctx.params.channel).trim();
  var sharp = String(ctx.params.pixels || 'smooth').toLowerCase() === 'sharp';
  ctx.el.innerHTML = '<div style="position:absolute;inset:0;background:#0b0e14;border-radius:6px;overflow:hidden">'
    + '<img class="cp" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;image-rendering:' + (sharp ? 'pixelated' : 'auto') + '">'
    + '<span class="badge" style="position:absolute;left:8px;top:8px;background:rgba(10,14,22,.72);color:#fff;font-size:11px;font-weight:600;padding:2px 8px;border-radius:999px;display:none">'
    + '<span class="dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#ff4d4d;margin-right:5px;vertical-align:1px"></span><span class="lbl">Live</span></span></div>';
  if (getComputedStyle(ctx.el).position === 'static') ctx.el.style.position = 'relative';
  ctx.el.style.height = '100%';
  var img = ctx.el.querySelector('.cp'), badge = ctx.el.querySelector('.badge'), dot = ctx.el.querySelector('.dot'), lbl = ctx.el.querySelector('.lbl');
  var PIC = /^data:image\/(png|jpeg|webp|gif);base64,/, lastT = null;
  function newest() {
    var best = null;
    ctx.ports().forEach(function (p) {
      var s = ctx.input(p).stream || [];
      for (var k = s.length - 1; k >= 0; k--) {
        var it = s[k], d = it && it.data;
        if (!d || typeof d.image !== 'string' || !PIC.test(d.image)) continue;
        if (chan && it.channel !== chan) continue;
        if (!best || (it.t || 0) > (best.t || 0)) best = it;
        break;
      }
    });
    return best;
  }
  function age() {
    if (lastT == null) return;
    var a = Math.max(0, Date.now() / 1000 - lastT), fresh = a < 15;
    dot.style.background = fresh ? '#ff4d4d' : '#7b8494';
    lbl.textContent = fresh ? 'Live' : 'Last picture ' + (a < 90 ? Math.round(a) + ' s' : Math.round(a / 60) + ' min') + ' ago';
  }
  function draw() {
    var it = newest();
    if (!it) { ctx.status(ctx.sample ? '' : 'Waiting for the camera (channel ' + (chan || 'any') + ')…'); return; }
    ctx.status('');
    if (img.getAttribute('src') !== it.data.image) img.src = it.data.image;
    img.style.display = 'block'; badge.style.display = 'inline-block';
    lastT = it.t != null ? +it.t : Date.now() / 1000;
    age();
  }
  ctx.on('update', draw);
  setInterval(age, 1000);
  draw();
}

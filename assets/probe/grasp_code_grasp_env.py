# ── the task: see the object coming on the belt, reach it, grasp it well, put it in the bin ─────────────────────────
# What the policy senses (as the robot itself could): its hand's pose and speed, its fingers' angles and commands, the
# touch sensors (four fingertips, the thumb tip and four zones of the palm), the conveyor's speed (the cell knows it),
# where the bin is, and what the camera network says of the object (where it is, how big, which way it lies), carried
# along at the belt speed between camera frames. What it does: moves its hand (6 stiff servos: x, y, z, yaw, pitch,
# roll: kinematic positioning) and its fingers (each finger's curl and the thumb's swing under the palm). The fingers
# and the object touch through MuJoCo's contacts: every force between them is simulated.
import math

import numpy as np

import scene as S
from sims import CTRL_DT

ACT_DIM = 12
OBS_DIM = 79
PRIV_DIM = 112
CAM_EVERY = 4                      # a camera frame every 4 control steps (6 per second)
STAGES = ("reach", "grasp", "place")
EP_SECONDS = {"reach": 9.0, "grasp": 12.0, "place": 16.0}
LIFT_OK, HOLD_STEPS = 0.08, 25     # a grasp: lifted 8 cm and held for 1 s (pushed about in the last half)
WORK_LO = np.array([-0.62, -0.08, 0.005])   # where the grasp point may go (the arm's reach around the cell)
WORK_HI = np.array([0.62, 0.62, 0.45])
DROP = np.array([S.BIN_C[0], S.BIN_C[1], 0.20])


def quat_mat(q):
    """(n, 4) w x y z -> (n, 3, 3)"""
    import torch
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def random_quat(torch, n, gen, dev):
    u = torch.rand(n, 3, generator=gen, device=dev)
    a, b, c = u.unbind(-1)
    return torch.stack([torch.sqrt(1 - a) * torch.sin(2 * math.pi * b), torch.sqrt(1 - a) * torch.cos(2 * math.pi * b),
                        torch.sqrt(a) * torch.sin(2 * math.pi * c), torch.sqrt(a) * torch.cos(2 * math.pi * c)], -1)


def quat_mul(a, b):
    import torch
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def obb_of(torch, pts_world):
    """an object's box as the camera can see it: the long axis on the belt (yaw, mod pi), half-extents along it,
    across it and up, and the box's centre. pts_world (n, k, 3)"""
    xy = pts_world[..., :2]
    m = xy.mean(1, keepdim=True)
    d = xy - m
    cxx, cyy, cxy = (d[..., 0] ** 2).mean(1), (d[..., 1] ** 2).mean(1), (d[..., 0] * d[..., 1]).mean(1)
    psi = 0.5 * torch.atan2(2 * cxy, cxx - cyy)
    c, s = torch.cos(psi), torch.sin(psi)
    u = d[..., 0] * c[:, None] + d[..., 1] * s[:, None]
    v = -d[..., 0] * s[:, None] + d[..., 1] * c[:, None]
    u0, u1, v0, v1 = u.min(1).values, u.max(1).values, v.min(1).values, v.max(1).values
    z0, z1 = pts_world[..., 2].min(1).values, pts_world[..., 2].max(1).values
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    cx = m[:, 0, 0] + uc * c - vc * s
    cy = m[:, 0, 1] + uc * s + vc * c
    centre = torch.stack([cx, cy, (z0 + z1) / 2], -1)
    ext = torch.stack([(u1 - u0) / 2, (v1 - v0) / 2, (z1 - z0) / 2], -1)
    return centre, ext, torch.stack([torch.cos(2 * psi), torch.sin(2 * psi)], -1)


class Camera:
    """turns the camera's depth and segmentation into what the network sees: per pixel its point in the cell
    (x, y, height) and its colour (shaded by its slope)"""

    def __init__(self, torch, dev, mjm):
        from sims import camera_dirs
        self.t = torch
        d = camera_dirs()                                       # (H*W, 3) unit rays in the camera's frame
        w = d / (-d[:, 2:3])                                    # per unit of planar depth
        cam = mjm.camera("cam")
        import mujoco
        dd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, dd)
        R = dd.cam_xmat[cam.id].reshape(3, 3)
        self.pos = torch.as_tensor(dd.cam_xpos[cam.id], dtype=torch.float32, device=dev)
        self.w = torch.as_tensor(w @ R.T, dtype=torch.float32, device=dev).reshape(S.CAM_H, S.CAM_W, 3)

    def image(self, depth, seg, rgba):
        """depth (n, H, W) planar, seg (n, H, W) geom ids (-1: none), rgba (n, ngeom, 4) -> (n, 6, H, W)"""
        torch = self.t
        depth = torch.nan_to_num(depth, nan=9.0, posinf=9.0, neginf=9.0)    # (a ray that hit nothing: far away)
        P = self.pos + depth[..., None] * self.w                # (n, H, W, 3)
        far = depth > 3.0
        P = torch.where(far[..., None], torch.zeros_like(P), P)
        ids = seg.clamp(min=0).long()
        col = torch.gather(rgba[..., :3], 1, ids.reshape(ids.shape[0], -1, 1).expand(-1, -1, 3)).reshape(*ids.shape, 3)
        col = torch.where((seg < 0)[..., None], torch.zeros_like(col), col)
        # shading: the surface's slope from neighbouring points (lit from above)
        gx = torch.zeros_like(P); gy = torch.zeros_like(P)
        gx[:, :, 1:-1] = P[:, :, 2:] - P[:, :, :-2]
        gy[:, 1:-1] = P[:, 2:] - P[:, :-2]
        nrm = torch.cross(gx, gy, dim=-1)
        nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-9)
        shade = 0.45 + 0.55 * nrm[..., 2].abs()
        col = col * shade[..., None]
        x = (P[..., 0] - 0.25) / 0.4
        y = (P[..., 1] - S.BELT_Y) / 0.4
        h = P[..., 2].clamp(-0.05, 0.3) / 0.1
        img = torch.stack([x, y, h, col[..., 0], col[..., 1], col[..., 2]], 1)
        return torch.where(far[:, None], torch.zeros_like(img), img)


def make_seenet(torch):
    import torch.nn as nn
    F = nn.functional

    class SeeNet(nn.Module):
        """camera image (point map and colour) -> which pixels are the object (a mask), and from them and what the
        network makes of the whole picture, the object's box: centre (3), half-extents (3), yaw (cos 2, sin 2), seen"""

        def __init__(self):
            super().__init__()
            self.d1 = nn.Sequential(nn.Conv2d(6, 32, 3, 2, 1), nn.ELU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ELU())      # 36x48
            self.d2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.ELU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ELU())     # 18x24
            self.d3 = nn.Sequential(nn.Conv2d(64, 96, 3, 1, 1), nn.ELU(), nn.Conv2d(96, 96, 3, 2, 1), nn.ELU())     # 9x12
            self.u2 = nn.Sequential(nn.Conv2d(96 + 64, 64, 3, 1, 1), nn.ELU())
            self.u1 = nn.Sequential(nn.Conv2d(64 + 32, 32, 3, 1, 1), nn.ELU())
            self.m = nn.Conv2d(32 + 6, 1, 3, 1, 1)
            self.h = nn.Sequential(nn.Linear(96 + 12, 256), nn.ELU(), nn.Linear(256, 9))
            with torch.no_grad():
                self.m.bias.fill_(-2.0)          # (most of the picture is not the object)

        def forward(self, x, mask_out=False):
            f1 = self.d1(x); f2 = self.d2(f1); f3 = self.d3(f2)
            g2 = self.u2(torch.cat([F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
            g1 = self.u1(torch.cat([F.interpolate(g2, size=f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
            x = x[:, :, ::2, ::2]                                   # (the mask at half the camera's resolution: 36x48)
            logit = self.m(torch.cat([g1, x], 1))[:, 0]
            # the object's points, weighted by the mask: where they are, how they spread, how high they reach
            w = torch.sigmoid(logit) * (x[:, 3:6].abs().sum(1) > 0).float()
            ws = w.sum((1, 2)).clamp(min=1e-3)
            X = 0.25 + 0.4 * x[:, 0]; Y = S.BELT_Y + 0.4 * x[:, 1]; H = 0.1 * x[:, 2]
            mx = (w * X).sum((1, 2)) / ws; my = (w * Y).sum((1, 2)) / ws; mh = (w * H).sum((1, 2)) / ws
            dx, dy = X - mx[:, None, None], Y - my[:, None, None]
            cxx = (w * dx * dx).sum((1, 2)) / ws; cyy = (w * dy * dy).sum((1, 2)) / ws; cxy = (w * dx * dy).sum((1, 2)) / ws
            hmax = (torch.softmax((H * 200).masked_fill(w < 0.5, -1e4).flatten(1), 1) * H.flatten(1)).sum(1)
            area = ws / 100.0
            stats = torch.stack([(mx - 0.25) / 0.4, (my - S.BELT_Y) / 0.4, mh * 10, cxx * 400, cyy * 400, cxy * 400, hmax * 10, area,
                                 torch.sqrt(cxx + cyy + 1e-8) * 20, (cxx - cyy) * 400, 2 * cxy * 400, (ws > 1).float()], 1)
            o = self.h(torch.cat([f3.mean((2, 3)), stats], 1))
            c = torch.stack([mx + 0.05 * torch.tanh(o[:, 0]), my + 0.05 * torch.tanh(o[:, 1]), 0.5 * hmax + 0.03 * torch.tanh(o[:, 2])], 1)
            e = 0.008 + F.softplus(o[:, 3:6]) * 0.04
            yw = torch.stack([cxx - cyy, 2 * cxy], 1) * 400 + o[:, 6:8] * 0.3
            yw = yw / (yw.norm(dim=1, keepdim=True) + 1e-6)
            out = (c, e, yw, o[:, 8])
            return out + (logit,) if mask_out else out
    return SeeNet()


def see_loss(torch, out, centre, ext, yaw2, seen, mask=None):
    c, e, yw, logit = out[:4]
    el = ((ext[:, 0] - ext[:, 1]) / ext[:, 0].clamp(min=1e-3)).clamp(0, 1)      # yaw matters for long things only
    m = seen.float()
    lc = (torch.nn.functional.smooth_l1_loss(c * 50, centre * 50, reduction="none").sum(1) * m).sum() / m.sum().clamp(min=1)
    le = (torch.nn.functional.smooth_l1_loss(e * 50, ext * 50, reduction="none").sum(1) * m).sum() / m.sum().clamp(min=1)
    ly = (((yw - yaw2) ** 2).sum(1) * el * m).sum() / m.sum().clamp(min=1)
    lv = torch.nn.functional.binary_cross_entropy_with_logits(logit, m)
    loss = lc + le + 2 * ly + 0.2 * lv
    if mask is not None and len(out) > 4:
        g = mask[:, ::2, ::2].float()
        pw = torch.tensor(20.0, device=mask.device)       # (the object is a small part of the picture)
        p = torch.sigmoid(out[4])
        dice = 1 - (2 * (p * g).sum((1, 2)) + 1) / (p.sum((1, 2)) + g.sum((1, 2)) + 1)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(out[4], g, pos_weight=pw) + 2.0 * dice.mean()
    return loss, {"centre_cm": float((((c.detach() - centre).norm(dim=1)) * m).sum() / m.sum().clamp(min=1) * 100),
                  "size_cm": float((((e.detach() - ext).abs().mean(1)) * m).sum() / m.sum().clamp(min=1) * 100)}


class GraspEnv:
    def __init__(self, torch, sim, lib, stage="reach", seed=1, test_objects=False, see=None, log=print):
        self.t = torch
        self.sim = sim
        self.n = n = sim.n
        self.dev = dev = sim.tdev
        self.stage = stage
        self.log = log
        tmpl, models, gs, ob, arrays, pts, info = lib
        self.info = info
        self.mjm = mjm = tmpl
        self.gs, self.ob = gs, ob
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self.pts = torch.as_tensor(pts, device=dev)                     # (nobj, k, 3) surface points
        self.train_ids = torch.tensor([i for i, x in enumerate(info) if not x["test"]], device=dev)
        self.test_ids = torch.tensor([i for i, x in enumerate(info) if x["test"]], device=dev)
        self.pool = self.test_ids if test_objects else self.train_ids
        self.rgba_lib = torch.as_tensor(arrays["geom_rgba"], device=dev)    # (nobj, nslot, 4)
        self.rgba = torch.as_tensor(np.array(tmpl.geom_rgba), dtype=torch.float32, device=dev)[None].repeat(n, 1, 1)
        J = lambda nm: mjm.joint(nm).qposadr[0]      # noqa: E731
        Dv = lambda nm: mjm.joint(nm).dofadr[0]      # noqa: E731
        self.qh = torch.tensor([J(j) for j in S.HAND_JOINTS], device=dev)
        self.dh = torch.tensor([Dv(j) for j in S.HAND_JOINTS], device=dev)
        self.qf = torch.tensor([J(j) for j in S.FINGER_JOINTS], device=dev)
        self.qo, self.do_ = J("obj"), Dv("obj")
        self.qb, self.db = J("belt"), Dv("belt")
        self.a_belt = mjm.actuator("a_belt").id
        self.gsite = mjm.site("grasp").id
        self.tadr = torch.tensor([mjm.sensor("t_" + nm).adr[0] for nm in S.TOUCH_NAMES], device=dev)
        self.qpos0 = torch.as_tensor(np.array(mjm.qpos0), dtype=torch.float32, device=dev)
        self.cam = Camera(torch, dev, mjm)
        self.see = see                       # the camera network (None: the true box, blurred, while it is learned)
        z = lambda *s: torch.zeros(*s, device=dev)   # noqa: E731
        self.obj = torch.zeros(n, dtype=torch.long, device=dev)
        self.tgt_p, self.tgt_e = z(n, 3), z(n, 3)
        self.curl, self.opp = z(n, 5), z(n)
        self.belt_v = z(n)
        self.z_rest = z(n)
        self.est_c, self.est_e, self.est_y, self.est_v, self.est_age = z(n, 3), z(n, 3), z(n, 2), z(n), z(n)
        self.prev_a = z(n, ACT_DIM)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.ok_cnt, self.hold_cnt, self.bin_cnt = z(n), z(n), z(n)
        self.hv_avg = z(n, 2)                # the hand's speed over the belt, smoothed (exploration jitters it step to step)
        self.prev_d = z(n) - 1               # the grasp point's distance a step ago (-1: just reset)
        self.prev_dd = z(n) - 1              # (place) the object's distance from over the bin a step ago
        self.lifted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ep_ret = z(n)
        self.max_steps = int(EP_SECONDS[stage] / CTRL_DT)
        self.k = 0
        self.n_bad = 0
        self.assist = {"reach": 0.0, "grasp": 0.5, "place": 0.3}[stage]   # a share of episodes start with the hand over the object
        if stage == "place" and n > 1:
            install_imitation(self)

    # ── resets ────────────────────────────────────────────────────────────────────────────────────────────────────
    def reset_all(self):
        self.reset(self.t.arange(self.n, device=self.dev))
        self.sim.refresh()

    def reset(self, idx, obj=None, mode=None, yaw=None, speed=None, x0=None, assist=None):
        torch = self.t
        k = len(idx)
        if k == 0:
            return
        dev, g = self.dev, self.gen
        if obj is None:
            obj = self.pool[torch.randint(len(self.pool), (k,), generator=g, device=dev)]
        self.obj[idx] = obj
        self.sim.set_objects(idx, obj)
        self.rgba[idx[:, None], torch.as_tensor(self.gs, device=dev)[None]] = self.rgba_lib[obj]
        # the object's pose: upright, on a side, or anyhow (it settles on the belt in the first half second)
        r = torch.rand(k, generator=g, device=dev)
        if mode is None:
            mode = torch.where(r < 0.5, 0, torch.where(r < 0.85, 1, 2))
        else:
            mode = torch.full((k,), int(mode), device=dev)
        yw = (torch.rand(k, generator=g, device=dev) * 2 * math.pi) if yaw is None else torch.full((k,), float(yaw), device=dev)
        qz = torch.stack([torch.cos(yw / 2), torch.zeros_like(yw), torch.zeros_like(yw), torch.sin(yw / 2)], -1)
        side = torch.rand(k, generator=g, device=dev) < 0.5
        h = math.sqrt(0.5)
        q_side = torch.where(side[:, None], torch.tensor([h, h, 0, 0], device=dev), torch.tensor([h, 0, h, 0], device=dev))
        q_up = torch.tensor([1.0, 0, 0, 0], device=dev).expand(k, 4)
        q_any = random_quat(torch, k, g, dev)
        q = torch.where((mode == 0)[:, None], q_up, torch.where((mode == 1)[:, None], q_side, q_any))
        q = quat_mul(qz, q)
        R = quat_mat(q)
        p = torch.einsum("kij,kmj->kmi", R, self.pts[obj])
        zlow = p[..., 2].min(1).values
        if x0 is None:
            x0 = S.SPAWN_X[0] + (S.SPAWN_X[1] - S.SPAWN_X[0]) * torch.rand(k, generator=g, device=dev)
        else:
            x0 = torch.full((k,), float(x0), device=dev)
        y0 = S.BELT_Y + 0.06 * (torch.rand(k, generator=g, device=dev) * 2 - 1)
        qpos = self.qpos0[None].repeat(k, 1)
        qpos[:, self.qo:self.qo + 3] = torch.stack([x0, y0, -zlow + 0.006], -1)
        qpos[:, self.qo + 3:self.qo + 7] = q
        # the belt's speed
        v = (0.03 + 0.05 * torch.rand(k, generator=g, device=dev)) if speed is None else torch.full((k,), float(speed), device=dev)
        self.belt_v[idx] = v
        qvel = torch.zeros(k, self.mjm.nv, device=dev)
        qvel[:, self.db] = -v
        # the hand: at rest, palm down, open (a little off each time)
        hp = torch.as_tensor(S.HOME, dtype=torch.float32, device=dev)[None] + 0.04 * (torch.rand(k, 3, generator=g, device=dev) * 2 - 1)
        hy = 0.3 * (torch.rand(k, generator=g, device=dev) * 2 - 1)
        qpos[:, self.qf[0]] = S.THUMB_OPEN
        # a share of the grasp and place episodes start with the hand already over the object, turned across it
        # (the reach it learned before), so that it finds out early what closing the hand on things does
        pa = self.assist if assist is None else float(assist)
        if pa > 0:
            cw, ew, yw2 = obb_of(torch, p + torch.stack([x0, y0, -zlow + 0.006], -1)[:, None])
            over = torch.rand(k, generator=g, device=dev) < pa
            hp_a = torch.stack([cw[:, 0] - 0.04 * v, cw[:, 1], cw[:, 2] + ew[:, 2] + 0.03 + 0.03 * torch.rand(k, generator=g, device=dev)], -1)
            hy_a = 0.5 * torch.atan2(yw2[:, 1], yw2[:, 0]) + math.pi / 2
            hy_a = torch.remainder(hy_a + math.pi / 2, math.pi) - math.pi / 2       # (a turn of pi is the same grasp)
            hp = torch.where(over[:, None], hp_a, hp)
            hy = torch.where(over, hy_a, hy)
        R0 = self._euler_R(torch.stack([hy, torch.zeros_like(hy), torch.zeros_like(hy)], -1))
        root = hp - torch.einsum("kij,j->ki", R0, torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=dev))
        qpos[:, self.qh[0]], qpos[:, self.qh[1]], qpos[:, self.qh[2]] = root[:, 0], root[:, 1], root[:, 2]
        qpos[:, self.qh[3]] = hy
        self.tgt_p[idx] = hp
        self.tgt_e[idx] = torch.stack([hy, torch.zeros_like(hy), torch.zeros_like(hy)], -1)
        self.curl[idx] = 0.0
        self.opp[idx] = 0.0
        ctrl = torch.zeros(k, self.mjm.nu, device=dev)
        self._fill_ctrl(ctrl, self.tgt_p[idx], self.tgt_e[idx], self.curl[idx], self.opp[idx])
        ctrl[:, self.a_belt] = -v
        self.sim.set_rows(idx, qpos, qvel, ctrl)
        self.steps[idx] = 0
        self.ok_cnt[idx] = 0
        self.hv_avg[idx] = 0
        self.prev_d[idx] = -1
        self.prev_dd[idx] = -1
        self.hold_cnt[idx] = 0
        self.bin_cnt[idx] = 0
        self.lifted_ever[idx] = False
        self.ep_ret[idx] = 0
        self.prev_a[idx] = 0
        self.z_rest[idx] = -zlow
        self.est_age[idx] = 9.0
        self.est_v[idx] = 0.0
        self.est_c[idx] = torch.stack([x0, y0, -zlow / 2], -1)
        self.est_e[idx] = 0.03
        self.est_y[idx] = torch.tensor([1.0, 0.0], device=dev)

    def _fill_ctrl(self, ctrl, p, e, curl, opp):
        """the hand mount's servos from where the grasp point should be (palm down, turned by yaw); the fingers' joints
        from their curls; the thumb's base from its swing"""
        torch = self.t
        R = self._euler_R(e)
        gp = torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=self.dev)
        root = p - torch.einsum("kij,j->ki", R, gp)
        ctrl[:, 0:3] = root
        ctrl[:, 3:6] = e
        cm = torch.as_tensor(S.CURL_MAX, dtype=torch.float32, device=self.dev)
        ctrl[:, 6] = S.THUMB_OPEN + opp * (S.THUMB_CLOSED - S.THUMB_OPEN)
        ctrl[:, 7] = curl[:, 0] * 1.0
        ctrl[:, 8] = curl[:, 0] * 1.1
        for f in range(4):
            ctrl[:, 9 + 3 * f:12 + 3 * f] = curl[:, 1 + f:2 + f] * cm
        return ctrl

    def _euler_R(self, e):
        torch = self.t
        yw, pt, rl = e.unbind(-1)
        cz, sz, cy, sy, cx, sx = torch.cos(yw), torch.sin(yw), torch.cos(pt), torch.sin(pt), torch.cos(rl), torch.sin(rl)
        one, zero = torch.ones_like(yw), torch.zeros_like(yw)
        Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(-1, 3, 3)
        Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(-1, 3, 3)
        Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(-1, 3, 3)
        return Rz @ Ry @ Rx

    # ── what is true (for the critic and the rewards) ─────────────────────────────────────────────────────────────
    def truth(self):
        torch = self.t
        sim = self.sim
        op = sim.xpos[:, self.ob].float()
        oq = sim.xquat[:, self.ob].float()
        R = quat_mat(oq)
        pw = torch.einsum("kij,kmj->kmi", R, self.pts[self.obj]) + op[:, None]
        c, e, y = obb_of(torch, pw)
        return op, oq, c, e, y

    def perceive(self):
        """a camera frame: the network's box of what it sees (or, without a network, the true box with errors like
        a camera's while the network is still to be learned)"""
        torch = self.t
        depth, seg = self.sim.render()
        depth = depth.float()
        img = self.cam.image(depth, seg, self.rgba)
        objpix = torch.isin(seg, torch.as_tensor(self.gs, device=self.dev)).sum((1, 2))
        if self.see is not None:
            with torch.no_grad():            # (in pieces: thousands of pictures at once would fill the GPU's memory)
                outs = [self.see(img[k:k + 512]) for k in range(0, img.shape[0], 512)]
                c, e, yw, logit = [torch.cat([o[i] for o in outs]) for i in range(4)]
            seen = torch.sigmoid(logit) > 0.5
        else:
            _, _, c, e, yw = self.truth()
            c = c + 0.006 * torch.randn(c.shape, generator=self.gen, device=self.dev)
            e = e * (1 + 0.08 * torch.randn(e.shape, generator=self.gen, device=self.dev))
            seen = objpix >= 6
        ok_ = torch.isfinite(c).all(1) & torch.isfinite(e).all(1) & torch.isfinite(yw).all(1)
        seen = seen & ok_
        c, e, yw = torch.nan_to_num(c), torch.nan_to_num(e), torch.nan_to_num(yw)
        old = self.est_c.clone()
        dt = self.est_age.clamp(min=CTRL_DT)
        vx = (c[:, 0] - old[:, 0]) / dt
        upd = seen
        self.est_v = torch.where(upd & (self.est_age < 1.0), 0.6 * self.est_v + 0.4 * vx.clamp(-0.3, 0.3), self.est_v)
        self.est_c = torch.where(upd[:, None], c, self.est_c)
        self.est_e = torch.where(upd[:, None], e, self.est_e)
        self.est_y = torch.where(upd[:, None], yw, self.est_y)
        self.est_age = torch.where(upd, torch.zeros_like(self.est_age), self.est_age)
        self.last_img = img
        self.last_seg = seg
        return img, objpix

    # ── a control step ────────────────────────────────────────────────────────────────────────────────────────────
    def step(self, a):
        torch = self.t
        sim = self.sim
        a = a.clamp(-1, 1)
        self.tgt_p = self.tgt_p + a[:, 0:3] * 0.018
        lo = torch.as_tensor(WORK_LO, dtype=torch.float32, device=self.dev)
        hi = torch.as_tensor(WORK_HI, dtype=torch.float32, device=self.dev)
        self.tgt_p = torch.max(torch.min(self.tgt_p, hi), lo)
        de = torch.stack([a[:, 3] * 0.10, a[:, 4] * 0.06, a[:, 5] * 0.06], -1)
        self.tgt_e = self.tgt_e + de
        self.tgt_e[:, 0] = self.tgt_e[:, 0].clamp(-2.4, 2.4)        # (any grasp yaw is within pi/2 of the next one)
        self.tgt_e[:, 1:] = self.tgt_e[:, 1:].clamp(-0.7, 0.7)
        self.curl = (self.curl + a[:, 6:11] * 0.12).clamp(0, 1)
        self.opp = (self.opp + a[:, 11] * 0.12).clamp(0, 1)
        ctrl = sim.ctrl.clone().float()
        self._fill_ctrl(ctrl, self.tgt_p, self.tgt_e, self.curl, self.opp)
        ctrl[:, self.a_belt] = -self.belt_v
        sim.ctrl[:] = ctrl.to(sim.ctrl.dtype)
        # the grasp test: once it holds the object up, it is pushed about (1.2 N, a new way every 0.2 s)
        sim.xfrc[:, self.ob] = 0
        if self.stage in ("grasp", "place"):
            push = (self.hold_cnt >= HOLD_STEPS // 2) & (self.hold_cnt < HOLD_STEPS)
            if bool(push.any()):
                if self.k % 5 == 0 or not hasattr(self, "_push_dir"):
                    d_ = torch.randn(self.n, 3, generator=self.gen, device=self.dev)
                    self._push_dir = d_ / d_.norm(dim=1, keepdim=True)
                sim.xfrc[:, self.ob, :3] = (1.2 * self._push_dir * push[:, None].float()).to(sim.xfrc.dtype)
        sim.step()
        self.k += 1
        self.steps += 1
        self.est_age += CTRL_DT
        self.est_c[:, 0] -= self.belt_v * CTRL_DT                 # carried by the belt between camera frames
        if self.k % CAM_EVERY == 0:
            self.perceive()
        rew, done, info = self.reward(a)
        # a cell whose physics blew up (rare, among thousands): it ends here and starts again, and teaches nothing
        bad = ~torch.isfinite(rew) | ~torch.isfinite(sim.qpos).all(1) | ~torch.isfinite(sim.qvel).all(1)
        if bool(bad.any()):
            self.n_bad += int(bad.sum())
            rew = torch.where(bad, torch.zeros_like(rew), rew)
            done = done | bad
            info["timeout"] = info["timeout"] | bad
            info["success"] = info["success"] & ~bad
            info["fail"] = info["fail"] & ~bad
            info["d"] = torch.nan_to_num(info["d"], nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_a = a
        self.ep_ret += rew
        return rew, done, info

    # ── rewards, success and failure ──────────────────────────────────────────────────────────────────────────────
    def hand_state(self):
        sim = self.sim
        g = sim.site_xpos[:, self.gsite].float()
        R = sim.site_xmat[:, self.gsite].float().reshape(-1, 3, 3)
        return g, R

    def touches(self):
        f = self.sim.sens[:, self.tadr].float()
        return f

    def reward(self, a):
        torch = self.t
        st = self.stage
        sim = self.sim
        g, R = self.hand_state()
        op, oq, c, e, y = self.truth()
        ov = sim.qvel[:, self.do_:self.do_ + 3].float()
        top = c[:, 2] + e[:, 2]
        T = torch.stack([c[:, 0], c[:, 1], torch.max(c[:, 2], top - 0.05)], -1)       # where the grasp point should go
        d = (g - T).norm(dim=1)
        dxy = (g[:, :2] - T[:, :2]).norm(dim=1)
        dz = (g[:, 2] - T[:, 2]).abs()
        # near (within ~15 cm) and far (from across the cell) terms: tanh(8 d) alone is flat a hand's length away,
        # and the policy then never finds the object
        r_reach = 0.5 * (1 - torch.tanh(8 * d)) + 0.5 * (1 - torch.tanh(2.5 * d))
        # and the distance gained this step (a potential: it adds up to the start's distance minus the end's)
        gain = torch.where(self.prev_d >= 0, (self.prev_d - d).clamp(-0.05, 0.05), torch.zeros_like(d))
        self.prev_d = d.detach().clone()
        # palm down; fingers across the object's long side (the thumb and the fingers close along the hand's x)
        level = R[:, 2, 2]
        hx = R[:, :2, 0]
        yaw_h = torch.atan2(hx[:, 1], hx[:, 0])
        psi = 0.5 * torch.atan2(y[:, 1], y[:, 0])
        align = torch.cos(2 * (yaw_h - psi - math.pi / 2))                 # 1: across the long axis
        elong = ((e[:, 0] - e[:, 1]) / e[:, 0].clamp(min=1e-3)).clamp(0, 1)
        r_align = elong * (align + 1) / 2
        f = self.touches()
        thumb, fing, palm = f[:, 0], f[:, 1:5], f[:, 5:9]
        tn = torch.log1p(f) / math.log1p(10.0)
        pinch = torch.min(tn[:, 0], tn[:, 1:5].max(1).values).clamp(0, 1)
        wrap = (tn[:, 1:5].clamp(0, 1).sum(1) / 4) * (tn[:, 0].clamp(0, 1) + tn[:, 5:9].max(1).values.clamp(0, 1)).clamp(0, 1)
        held = ((thumb > 0.3) | (palm.max(1).values > 0.3)) & ((fing > 0.3).sum(1) >= 1)
        lift = op[:, 2] - self.z_rest
        lifted = (lift > LIFT_OK) & held
        rel_v = (ov[:, 0] + self.belt_v).abs() + ov[:, 1].abs()
        on_belt = (op[:, 2] < self.z_rest + 0.015)
        knock = torch.where(on_belt & ~held, (rel_v - 0.03).clamp(min=0), torch.zeros_like(rel_v))
        rate = ((a - self.prev_a) ** 2).mean(1)
        crush = (f.max(1).values - 25.0).clamp(min=0) / 25.0
        low = (0.004 - (g[:, 2] - 0.0)).clamp(min=0) * 50
        hv = sim.qvel[:, self.dh[0]:self.dh[0] + 2].float()
        self.hv_avg = 0.85 * self.hv_avg + 0.15 * hv
        match = (self.hv_avg[:, 0] + self.belt_v).abs() + self.hv_avg[:, 1].abs()
        early = self.steps < 12                                            # (the object settling on the belt)
        self.z_rest = torch.where(self.steps == 12, op[:, 2], self.z_rest)
        r = 0.6 * r_reach + 20.0 * gain + 0.25 * r_align + 0.1 * level.clamp(min=0) - 0.05 * rate - 1.0 * knock - 0.5 * crush - low
        success = torch.zeros_like(held)
        fail = (op[:, 0] < S.EXIT_X) | (op[:, 2] < -0.08)
        in_bin = ((op[:, 0] - S.BIN_C[0]).abs() < S.BIN_HALF[0]) & ((op[:, 1] - S.BIN_C[1]).abs() < S.BIN_HALF[1]) & (op[:, 2] < S.BIN_H) & (op[:, 2] > -0.02)
        if st == "reach":
            ok = (dxy < 0.025) & (dz < 0.025) & (level > 0.85) & ((elong < 0.3) | (align > 0.7)) & (match < 0.12) & ~early
            self.ok_cnt = torch.where(ok, self.ok_cnt + 1, torch.zeros_like(self.ok_cnt))
            success = self.ok_cnt >= 3
            r = r + 1.0 * r_reach * (dxy < 0.04).float() * (1 - torch.tanh(15 * dz)) + 10.0 * success.float()
        else:
            near = (d < 0.05).float()
            self.hold_cnt = torch.where(lifted, self.hold_cnt + 1, torch.zeros_like(self.hold_cnt))
            grasped = self.hold_cnt >= HOLD_STEPS
            if st == "grasp":
                r = r + 0.6 * pinch + 0.6 * wrap + 0.3 * near * self.curl[:, 1:].mean(1) * (self.opp > 0.5).float()
                r = r + torch.where(held, 10.0 * lift.clamp(0, 0.14), torch.zeros_like(lift))
                r = r + 0.3 * lifted.float()
                success = grasped
                r = r + 20.0 * success.float()
            else:
                # placing: no reward for holding as such (a policy would hold forever); what pays is the object's
                # distance gained towards over the bin while it is in the hand (a potential: lifting it counts, holding
                # it still does not), letting it go over the bin, and it resting there
                self.lifted_ever = self.lifted_ever | grasped
                r = r + (0.3 * pinch + 0.3 * near * self.curl[:, 1:].mean(1) * (self.opp > 0.5).float()) * (~self.lifted_ever).float()
                dd = (op - torch.as_tensor(DROP, dtype=torch.float32, device=self.dev)).norm(dim=1)
                up = (op[:, 2] > self.z_rest + 0.03) & ~in_bin & ~early     # carried (the touch sites need not feel it)
                gain_b = torch.where((self.prev_dd >= 0) & up, (self.prev_dd - dd).clamp(-0.05, 0.05), torch.zeros_like(dd))
                self.prev_dd = torch.where(up, dd, -torch.ones_like(dd))
                r = r + 40.0 * gain_b
                over = ((op[:, 0] - S.BIN_C[0]).abs() < S.BIN_HALF[0] - 0.02) & ((op[:, 1] - S.BIN_C[1]).abs() < S.BIN_HALF[1] - 0.02)
                lifted_some = op[:, 2] > self.z_rest + 0.03
                r = r + torch.where(over & lifted_some & (op[:, 2] < 0.25), 1.0 * (1 - self.curl.mean(1)) + 0.5 * (1 - self.opp), torch.zeros_like(dd))
                resting = in_bin & (f.max(1).values < 0.2) & (ov.norm(dim=1) < 0.05)
                self.bin_cnt = torch.where(resting, self.bin_cnt + 1, torch.zeros_like(self.bin_cnt))
                success = self.bin_cnt >= 8
                r = r + 30.0 * success.float()
                fail = fail | (self.lifted_ever & ~in_bin & (op[:, 2] < self.z_rest - 0.03))
        r = torch.where(fail, r - 5.0, r)
        # success ends the episode: it is worth at least what the rest of the episode could have paid step by step
        # (else the policy learns to hover just short of it, collecting the shaping instead)
        left = (self.max_steps - self.steps).clamp(min=0).float()
        r = r + torch.where(success, (2.5 if st == "reach" else 5.0) * left, torch.zeros_like(r))
        timeout = self.steps >= self.max_steps
        done = success | fail | timeout
        info = {"success": success, "fail": fail, "timeout": timeout & ~success & ~fail, "d": d, "lift": lift, "held": held,
                "lifted": lifted, "in_bin": in_bin, "touch": f, "T": T, "g": g}
        self._last = (g, R, op, oq, c, e, y, f, held, lift)
        return r, done, info

    # ── what the policy and the critic get ────────────────────────────────────────────────────────────────────────
    def obs(self):
        torch = self.t
        sim = self.sim
        g, R = self.hand_state()
        hv = sim.qvel[:, self.dh].float()
        fq = sim.qpos[:, self.qf].float()
        f = self.touches()
        tn = torch.log1p(f) / math.log1p(20.0)
        ec, ee, ey = self.est_c, self.est_e, self.est_y
        top = ec[:, 2] + ee[:, 2]
        T = torch.stack([ec[:, 0], ec[:, 1], torch.max(ec[:, 2], top - 0.05)], -1)
        hx = R[:, :2, 0]
        yaw_h = torch.atan2(hx[:, 1], hx[:, 0])
        psi = 0.5 * torch.atan2(ey[:, 1], ey[:, 0])
        dy = 2 * (yaw_h - psi - math.pi / 2)
        drop = torch.as_tensor(DROP, dtype=torch.float32, device=self.dev)
        obs = torch.cat([
            g, R[:, :, 0], R[:, :, 2],                               # 3 + 3 + 3
            (self.tgt_p - g) * 10, hv[:, :3], hv[:, 3:] * 0.2,       # 3 + 3 + 3
            fq / 1.5, self.curl, self.opp[:, None],                  # 15 + 5 + 1
            tn,                                                      # 9
            (T - g) * 5, ec, ee * 10, ey, torch.stack([torch.cos(dy), torch.sin(dy)], -1),    # 3 + 3 + 3 + 2 + 2
            (self.est_age.clamp(max=3) / 3)[:, None], (self.belt_v * 10)[:, None], (self.est_v * 10)[:, None],   # 3
            (drop - g) * 3,                                           # 3
            self.prev_a,                                              # 12
        ], 1)
        op, oq, c, e, y = self.truth()
        ov = sim.qvel[:, self.do_:self.do_ + 6].float()
        st = torch.zeros(self.n, 3, device=self.dev)
        st[:, STAGES.index(self.stage)] = 1
        priv = torch.cat([obs, (c - g) * 5, oq, ov[:, :3], ov[:, 3:] * 0.1, e * 10, y, ((op[:, 2] - self.z_rest) * 10)[:, None],
                          (f > 0.3).float(), st, (self.steps.float() / self.max_steps)[:, None], self.lifted_ever.float()[:, None]], 1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100, 100), torch.nan_to_num(priv, nan=0.0, posinf=0.0, neginf=0.0).clamp(-100, 100)


# ── learning from demonstrations (the place stage) ──────────────────────────────────────────────────────────────────
# Carrying an object a metre to the bin and letting go there is too long a chain for PPO to stumble on. A scripted
# controller that uses only what the robot senses (its hand, and the camera network's box of the object) shows how:
# the policy first imitates it (behaviour cloning), then PPO carries on with the same rewards and improves on it.
class Demo:
    """the scripted controller: over the object, down, close, lift, carry over the bin, open"""

    def __init__(self, env):
        t = env.t
        self.env = env
        self.phase = t.zeros(env.n, dtype=t.long, device=env.dev)
        self.timer = t.zeros(env.n, device=env.dev)

    def reset(self, idx):
        self.phase[idx] = 0
        self.timer[idx] = 0

    def act(self):
        env, torch = self.env, self.env.t
        n, dev, phase, timer = env.n, env.dev, self.phase, self.timer
        g, R = env.hand_state()
        ec, ee, ey = env.est_c, env.est_e, env.est_y
        T = torch.stack([ec[:, 0] - env.belt_v * 0.15, ec[:, 1], torch.max(ec[:, 2], ec[:, 2] + ee[:, 2] - 0.05)], -1)
        psi = 0.5 * torch.atan2(ey[:, 1], ey[:, 0])
        want = torch.remainder(psi + math.pi, math.pi) - math.pi / 2
        a = torch.zeros(n, ACT_DIM, device=dev)
        tgt = T.clone()
        tgt[:, 2] = torch.where(phase == 0, T[:, 2] + 0.07, T[:, 2])
        lift = torch.stack([env.tgt_p[:, 0], env.tgt_p[:, 1], torch.full((n,), 0.20, device=dev)], -1)
        drop = torch.as_tensor(DROP, dtype=torch.float32, device=dev).expand(n, 3)
        tgt = torch.where((phase == 2)[:, None], env.tgt_p, tgt)
        tgt = torch.where((phase == 3)[:, None], lift, tgt)
        tgt = torch.where((phase >= 4)[:, None], drop, tgt)
        a[:, :3] = ((tgt - env.tgt_p) / 0.018).clamp(-1, 1)
        a[:, 3] = ((want - env.tgt_e[:, 0]) / 0.10).clamp(-1, 1)
        a[:, 4:6] = (-env.tgt_e[:, 1:] / 0.06).clamp(-1, 1)
        close = (phase >= 2) & (phase < 5)
        a[:, 6:11] = torch.where(close[:, None], torch.ones(n, 5, device=dev), -torch.ones(n, 5, device=dev))
        a[:, 11] = torch.where(close | ((phase == 1) & (timer > 4)), torch.ones(n, device=dev), -torch.ones(n, device=dev))
        d = g - T
        timer += 1
        nxt = (phase == 0) & (d[:, :2].norm(dim=1) < 0.012) & (timer > 8)
        nxt |= (phase == 1) & (d.norm(dim=1) < 0.012) & (timer > 6)
        nxt |= (phase == 2) & (timer > 12)
        nxt |= (phase == 3) & (env.tgt_p[:, 2] > 0.19) & (timer > 30)
        nxt |= (phase == 4) & ((env.tgt_p - drop).norm(dim=1) < 0.01)
        self.phase = torch.where(nxt, phase + 1, phase)
        self.timer = torch.where(nxt, torch.zeros_like(timer), timer)
        return a


def imitate(env, algo, log, steps=1200, epochs=4000):
    """the policy learns to act as the scripted controller does: the controller drives the cells (with a little noise,
    so the policy also sees how to get back on track) while (what the policy senses, what the controller does) pairs
    are kept; then the policy's network is fitted to them"""
    import time
    torch = env.t
    t0 = time.time()
    demo = Demo(env)
    env.reset_all()
    env.perceive()
    O, A = [], []
    ok = tot = 0
    for k in range(steps):
        a = demo.act()
        o, _ = env.obs()
        if k % 2 == 0:
            O.append(o.half())
            A.append(a.half())
        r, done, info = env.step((a + 0.15 * torch.randn_like(a)).clamp(-1, 1))
        if bool(done.any()):
            idx = done.nonzero().flatten()
            ok += int(info["success"][idx].sum())
            tot += len(idx)
            env.reset(idx)
            env.sim.refresh()
            demo.reset(idx)
    O, A = torch.cat(O), torch.cat(A)
    log("Demonstrations: %d episodes, %.0f%% put in the bin by the scripted controller; %d examples." % (tot, 100.0 * ok / max(1, tot), len(O)))
    for k in range(0, len(O), 65536):
        algo.on.update(O[k:k + 65536].float())
    actor = algo.ac.actor
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    N = len(O)
    for ep in range(epochs):
        j = torch.randint(N, (4096,), device=O.device)
        loss = ((actor(algo.on(O[j].float())) - A[j].float()) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        algo.ac.log_std.fill_(math.log(0.15))
    log("The policy imitates it: error %.3f after %d steps (%.0f s)." % (loss.item(), epochs, time.time() - t0))
    env.reset_all()
    env.perceive()


def policy_rate(env, algo, steps):
    """the share of episodes the policy (without exploration noise) ends with the object in the bin"""
    torch = env.t
    env.reset_all()
    env.perceive()
    ok = tot = 0
    for k in range(steps):
        o, p = env.obs()
        a = algo.act(o, p, deterministic=True)
        r, done, info = env.step(a)
        if bool(done.any()):
            idx = done.nonzero().flatten()
            ok += int(info["success"][idx].sum())
            tot += len(idx)
            env.reset(idx)
            env.sim.refresh()
    return 100.0 * ok / max(1, tot), tot


def install_imitation(env):
    """(the place stage) before PPO's first update: if the policy rarely puts things in the bin, it learns from the
    scripted controller first; its critic then learns for a while before the policy moves again"""
    import sys
    pp = sys.modules.get("ppo_part")
    if pp is None or getattr(pp.PPO, "_imitation", False):
        return
    orig = pp.PPO.update
    state = {"done": False, "critic": 0}

    def update(self, priv_last, critic_only=False):
        if not state["done"]:
            state["done"] = True
            rate, tot = policy_rate(env, self, env.max_steps + 10)
            env.log("The policy as it is puts %.0f%% of %d objects in the bin." % (rate, tot))
            if rate < 20:
                imitate(env, self, env.log)
                rate, tot = policy_rate(env, self, env.max_steps + 10)
                env.log("After imitating: %.0f%% of %d objects in the bin. Now PPO." % (rate, tot))
                state["critic"] = 30
        if state["critic"] > 0:
            state["critic"] -= 1
            critic_only = True
        return orig(self, priv_last, critic_only)
    pp.PPO.update = update
    pp.PPO._imitation = True

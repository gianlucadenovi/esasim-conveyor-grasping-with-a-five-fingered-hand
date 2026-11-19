# ── EsaSim block ──────────────────────────────────────────────────────
# Conveyor grasping (xArm + 5-finger hand) (ws-conveyor-grasping-xarm-5-finger-hand) · version 4 · worker block, Python: def run(ctx)
# A five-fingered hand on an xArm learns to see objects on a conveyor, reach them, grasp them with real contact forces and put them in a bin (MuJoCo, GPU or CPU); then a test with new objects.
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-19
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt)                Policies · several · optional · Camera networks (grasp_see_*.pt) and policies (grasp_reach/grasp/place_*.pt) to carry on from or to test; those left on the machine by earlier runs are found by themselves
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Network · grasp_see_<date>.pt (Learn to see) or grasp_<stage>_<date>_<success>.pt with the camera network it used (Learn to reach, grasp, place)
#   curve       Table (csv)               Learning curve · see.csv, reach.csv, grasp.csv, place.csv, or test.csv (one row per trial)
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn to reach" of Learn to see | Learn to reach | Learn to grasp | Learn to place | …
#   accelerator select    Run on = "auto" of auto | NVIDIA GPU | CPU
#   worlds      number    Cells learning at once = 2048
#   minutes     number    Minutes = 60
#   training    select    Training = "Continue" of Continue | Start over
#   replay_every number    Replay every (s) = 20
#   pace        number    Test pace (1 = real time) = 1
#   autoplay    select    Test: pick objects when idle = "Yes" of Yes | No
#   idle        number    Test: idle before it picks (s) = 8
#   objects     select    Test objects = "New objects only" of New objects only | All objects
#   seed        number    Seed = 1
# Runs on  Any worker, NVIDIA GPU. Return a dict to fill the "result" output.
# Packages torch, mujoco>=3.3, mujoco-warp, warp-lang, numpy (installed by the worker with pip when missing)
# EsaSim rewrites these lines from the block's definition each time it is saved: change the definition, not them.
# ── end of EsaSim block header ────────────────────────────────────────

"""Conveyor grasping: an xArm with a five-fingered hand learns to see the objects coming on a conveyor belt (a camera
next to the robot), to reach them, to grasp them well (every contact force between the fingers, the palm and the
object is simulated by MuJoCo; touch sensors in the fingertips and the palm), and to put them in a bin. Then the test:
objects it never learned on, chosen in the view, in the pose chosen.

Modes: Learn to see (the camera network), Learn to reach, Learn to grasp, Learn to place (PPO, each stage carrying on
from the one before), Test with new objects. On the NVIDIA GPU (MuJoCo Warp, thousands of cells at once, with its
batch renderer for the cameras) or on the CPU (MuJoCo)."""
import sys as _sys
import types as _types

_SOURCES = {}
_SOURCES['scene'] = r'''# ── the cell: belt, bin, camera, the hand on its kinematic mount, and the objects ────────────────────────────────────
# World frame: the xArm's base at the origin, z up, metres. The belt runs along x in front of the robot (objects come
# from +x and travel towards -x); the bin is next to the belt, downstream; the camera is on a pole beside the robot.
import math

import numpy as np

BELT_Y, BELT_W, BELT_TOP = 0.42, 0.30, 0.0       # belt centre line (y), width, top surface height
BELT_LEN = 6.4                                   # long enough to run a whole episode without being reset
BELT_X0 = 0.2                                    # belt centre (x) at the start of an episode
SPAWN_X = (0.55, 0.72)                           # where objects are put on the belt
EXIT_X = -0.70                                   # an object past here has gone by
BIN_C = np.array([-0.40, 0.10, 0.0])             # bin floor centre
BIN_HALF = (0.12, 0.09)                          # inner half sizes
BIN_WALL, BIN_H = 0.010, 0.10
CAM_POS = np.array([0.30, 0.02, 0.80])           # the camera on its pole
CAM_LOOK = np.array([0.24, 0.42, 0.0])
CAM_FOVY = 58.0
CAM_W, CAM_H = 96, 72
HOME = np.array([0.12, 0.22, 0.30])              # the hand's rest pose (palm down, fingers along +x)
GRASP_PT = (0.048, 0.010, -0.058)                # between the thumb and the fingertips, in the hand's frame
THUMB_OPEN, THUMB_CLOSED = -2.0, 0.45            # the thumb's base joint: swung back, and opposing the fingers

# the hand (metres, in its own frame: palm centre at the origin, fingers along +x, palm facing -z, thumb on +y)
PALM_HALF = (0.046, 0.042, 0.012)
FINGERS = [  # name, base y, phalanx lengths, radius
    ("index", 0.028, (0.044, 0.026, 0.021), 0.0085),
    ("middle", 0.0095, (0.047, 0.029, 0.022), 0.0088),
    ("ring", -0.009, (0.044, 0.027, 0.021), 0.0085),
    ("little", -0.027, (0.036, 0.021, 0.018), 0.0078),
]
THUMB = {"base": (-0.030, 0.012, -0.014), "len": (0.040, 0.032, 0.026), "r": 0.0098}
THUMB_U = np.array([0.30, -0.08, -0.95]); THUMB_U = THUMB_U / np.linalg.norm(THUMB_U)      # the thumb's direction (open-ish)
THUMB_A = np.cross(THUMB_U, [1.0, 0.0, 0.25]); THUMB_A = THUMB_A / np.linalg.norm(THUMB_A)   # its flexion axis: curls it forward
# what the policy moves: the hand's pose (6 stiff position servos) and 6 finger motors: each finger's curl (its three
# joints together, as tendons do) and the thumb's opposition (swinging it under the palm); the thumb's curl is the 5th
CURL_SPLIT = (0.45, 0.33, 0.22)                  # how a finger's curl shares out over its three joints
CURL_MAX = (1.45, 1.55, 1.2)                     # their limits (rad)
TOUCH_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip", "palm_1", "palm_2", "palm_3", "palm_4"]
N_SLOT = {"box": 3, "cylinder": 3, "sphere": 2, "capsule": 3, "ellipsoid": 1}
SLOT_ORDER = [t for t in ("box", "cylinder", "sphere", "capsule", "ellipsoid") for _ in range(N_SLOT[t])]


def _f(v):
    return " ".join("%.6g" % x for x in v)


def quat_from_euler(rx, ry, rz):
    """w x y z of the rotation rz(about z) * ry * rx (radians)"""
    cx, sx, cy, sy, cz, sz = math.cos(rx / 2), math.sin(rx / 2), math.cos(ry / 2), math.sin(ry / 2), math.cos(rz / 2), math.sin(rz / 2)
    return (cz * cy * cx + sz * sy * sx, cz * cy * sx - sz * sy * cx, cz * sy * cx + sz * cy * sx, sz * cy * cx - cz * sy * sx)


def part_quat(eul):
    """a part's orientation from (x, y, z) angles in degrees, applied x, then y, then z about the object's axes
    (R = Rz Ry Rx: a lying capsule (0, 90, 0) turned by z degrees about the vertical)"""
    rx, ry, rz = [math.radians(e) for e in eul]
    return quat_from_euler(rx, ry, rz)


def look_xyaxes(pos, look, up=(0, 0, 1)):
    f = np.asarray(look, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return list(x) + list(y)


# ── the objects ─────────────────────────────────────────────────────────────────────────────────────────────────────
# Each object: parts (type, MuJoCo size, position, euler angles in degrees) in its own frame (resting upright, z up,
# origin inside it) and a colour. Sizes are MuJoCo's: box half-sizes, cylinder/capsule (radius, half-length),
# sphere (radius), ellipsoid (radii). The robot learns on TRAIN; TEST are new to it.
def P(t, size, pos=(0, 0, 0), eul=(0, 0, 0)):
    return (t, tuple(size), tuple(pos), tuple(eul))


TRAIN = {
    "cube": ("Cube", [P("box", (0.024, 0.024, 0.024))], (0.85, 0.25, 0.20)),
    "brick": ("Brick", [P("box", (0.060, 0.024, 0.020))], (0.78, 0.36, 0.22)),
    "tall_box": ("Tall box", [P("box", (0.024, 0.024, 0.050))], (0.25, 0.55, 0.85)),
    "flat_box": ("Flat box", [P("box", (0.050, 0.038, 0.013))], (0.90, 0.75, 0.25)),
    "can": ("Can", [P("cylinder", (0.032, 0.040))], (0.80, 0.80, 0.82)),
    "rod": ("Rod", [P("cylinder", (0.014, 0.075))], (0.35, 0.75, 0.40)),
    "disc": ("Disc", [P("cylinder", (0.040, 0.012))], (0.55, 0.40, 0.75)),
    "ball": ("Ball", [P("sphere", (0.034,))], (0.95, 0.55, 0.15)),
    "small_ball": ("Small ball", [P("sphere", (0.024,))], (0.20, 0.70, 0.75)),
    "capsule": ("Capsule", [P("capsule", (0.020, 0.040))], (0.85, 0.30, 0.55)),
    "egg": ("Egg", [P("ellipsoid", (0.028, 0.028, 0.040))], (0.92, 0.88, 0.75)),
    "l_block": ("L block", [P("box", (0.045, 0.018, 0.018), (0.0, 0, 0)), P("box", (0.018, 0.035, 0.018), (0.027, 0.053, 0))], (0.30, 0.45, 0.80)),
    "t_block": ("T block", [P("box", (0.050, 0.016, 0.018)), P("box", (0.016, 0.040, 0.018), (0, -0.056, 0))], (0.75, 0.25, 0.30)),
    "mug": ("Mug", [P("cylinder", (0.034, 0.042)), P("capsule", (0.008, 0.022), (0.046, 0, 0))], (0.92, 0.92, 0.90)),
    "bottle": ("Bottle", [P("cylinder", (0.030, 0.050)), P("cylinder", (0.012, 0.020), (0, 0, 0.068)), P("cylinder", (0.014, 0.006), (0, 0, 0.092))], (0.30, 0.65, 0.45)),
    "dumbbell": ("Dumbbell", [P("capsule", (0.012, 0.040), (0, 0, 0), (0, 90, 0)), P("sphere", (0.026,), (-0.058, 0, 0)), P("sphere", (0.026,), (0.058, 0, 0))], (0.35, 0.35, 0.40)),
    "hammer": ("Mallet", [P("capsule", (0.011, 0.060), (0, 0, 0), (0, 90, 0)), P("box", (0.018, 0.035, 0.022), (0.075, 0, 0.004))], (0.62, 0.45, 0.30)),
    "cross": ("Cross", [P("box", (0.050, 0.014, 0.016)), P("box", (0.014, 0.050, 0.016))], (0.25, 0.60, 0.60)),
}
TEST = {
    "banana": ("Banana", [P("capsule", (0.016, 0.030), (0, 0, 0), (0, 90, 0)), P("capsule", (0.015, 0.026), (0.050, 0.016, 0), (0, 90, 35)), P("capsule", (0.015, 0.026), (-0.050, 0.016, 0), (0, 90, -35))], (0.98, 0.85, 0.25)),
    "cone": ("Stepped cone", [P("cylinder", (0.036, 0.014), (0, 0, -0.028)), P("cylinder", (0.026, 0.014), (0, 0, 0)), P("cylinder", (0.015, 0.014), (0, 0, 0.028))], (0.95, 0.45, 0.10)),
    "drill": ("Drill", [P("box", (0.048, 0.020, 0.024), (0, 0, 0.026)), P("box", (0.016, 0.018, 0.040), (-0.024, 0, -0.030)), P("cylinder", (0.010, 0.016), (0.062, 0, 0.026), (0, 90, 0))], (0.20, 0.55, 0.30)),
    "pear": ("Pear", [P("sphere", (0.034,), (0, 0, -0.012)), P("sphere", (0.022,), (0, 0, 0.032))], (0.65, 0.80, 0.30)),
    "spray": ("Spray can", [P("cylinder", (0.026, 0.065)), P("cylinder", (0.010, 0.010), (0, 0, 0.073))], (0.85, 0.20, 0.25)),
    "u_block": ("U block", [P("box", (0.050, 0.014, 0.018)), P("box", (0.014, 0.032, 0.018), (-0.036, 0.042, 0)), P("box", (0.014, 0.032, 0.018), (0.036, 0.042, 0))], (0.55, 0.30, 0.70)),
    "plank": ("Plank", [P("box", (0.085, 0.020, 0.010))], (0.70, 0.55, 0.35)),
    "big_egg": ("Flat egg", [P("ellipsoid", (0.042, 0.030, 0.024))], (0.85, 0.75, 0.60)),
    "duck": ("Duck", [P("ellipsoid", (0.040, 0.030, 0.026)), P("sphere", (0.019,), (0.028, 0, 0.034)), P("capsule", (0.007, 0.010), (0.050, 0, 0.032), (0, 90, 0))], (0.98, 0.82, 0.10)),
    "torch": ("Torch", [P("cylinder", (0.016, 0.055)), P("cylinder", (0.026, 0.018), (0, 0, 0.068))], (0.25, 0.25, 0.30)),
    "z_block": ("Z block", [P("box", (0.014, 0.040, 0.016)), P("box", (0.030, 0.014, 0.016), (0.040, 0.026, 0)), P("box", (0.030, 0.014, 0.016), (-0.040, -0.026, 0))], (0.30, 0.70, 0.55)),
    "pill": ("Pill bottle", [P("cylinder", (0.022, 0.035)), P("cylinder", (0.026, 0.010), (0, 0, 0.044))], (0.92, 0.55, 0.20)),
}


def assign_slots(parts):
    """parts -> the geom slots they use: list of (slot index, part)"""
    used = {t: 0 for t in N_SLOT}
    out = []
    for prt in parts:
        t = prt[0]
        k = used[t]
        if k >= N_SLOT[t]:
            raise ValueError("too many %s parts" % t)
        used[t] += 1
        out.append(([i for i, s in enumerate(SLOT_ORDER) if s == t][k], prt))
    return out


# ── the model (MJCF) ────────────────────────────────────────────────────────────────────────────────────────────────
def hand_xml():
    """the five-fingered hand, on six stiff servos (x, y, z, then yaw, pitch, roll): its pose is kinematic, its
    fingers are real joints with real contacts"""
    px, py, pz = PALM_HALF
    hg = 'contype="2" conaffinity="1" condim="4" friction="1.2 0.02 0.001" solref="0.004 1"'
    s = []
    s.append('<body name="hand_x" pos="0 0 0"><joint name="hx" type="slide" axis="1 0 0" range="-1.5 1.5" armature="0.5"/><inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>')
    s.append('<body name="hand_y"><joint name="hy" type="slide" axis="0 1 0" range="-1.5 1.5" armature="0.5"/><inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>')
    s.append('<body name="hand_z"><joint name="hz" type="slide" axis="0 0 1" range="-0.5 1.5" armature="0.5"/><inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>')
    s.append('<body name="hand_yaw"><joint name="hyaw" type="hinge" axis="0 0 1" range="-6.3 6.3" armature="0.02"/><inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>')
    s.append('<body name="hand_pitch"><joint name="hpitch" type="hinge" axis="0 1 0" range="-1.6 1.6" armature="0.02"/><inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>')
    s.append('<body name="hand"><joint name="hroll" type="hinge" axis="1 0 0" range="-1.6 1.6" armature="0.02"/>')
    s.append('<geom name="palm" type="box" size="%s" mass="0.35" rgba="0.82 0.84 0.88 1" %s/>' % (_f(PALM_HALF), hg))
    s.append('<geom name="wrist" type="cylinder" size="0.030 0.022" pos="%s" euler="0 1.5708 0" mass="0.05" rgba="0.25 0.27 0.30 1" contype="0" conaffinity="0"/>' % _f((-px - 0.020, 0, 0.002)))
    s.append('<site name="grasp" pos="%s" size="0.006" rgba="1 0 0 0"/>' % _f(GRASP_PT))
    for k, (cx, cy) in enumerate([(0.022, 0.020), (0.022, -0.020), (-0.022, 0.020), (-0.022, -0.020)]):
        s.append('<site name="palm_%d" type="box" size="0.023 0.021 0.007" pos="%s" rgba="0 0 0 0"/>' % (k + 1, _f((cx, cy, -pz))))
    # the four fingers: proximal, middle and distal phalanges, flexing about the hand's y axis (towards -z)
    for name, y0, lens, r in FINGERS:
        L1, L2, L3 = lens
        s.append('<body name="%s_1" pos="%s">' % (name, _f((px, y0, -0.002))))
        s.append('<joint name="%s_j1" type="hinge" axis="0 1 0" range="-0.15 %g" damping="0.02" armature="0.002"/>' % (name, CURL_MAX[0]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.012" rgba="0.70 0.72 0.76 1" %s/>' % (L1, r, hg))
        s.append('<body name="%s_2" pos="%g 0 0"><joint name="%s_j2" type="hinge" axis="0 1 0" range="0 %g" damping="0.015" armature="0.002"/>' % (name, L1, name, CURL_MAX[1]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.008" rgba="0.74 0.76 0.80 1" %s/>' % (L2, r * 0.95, hg))
        s.append('<body name="%s_3" pos="%g 0 0"><joint name="%s_j3" type="hinge" axis="0 1 0" range="0 %g" damping="0.01" armature="0.002"/>' % (name, L2, name, CURL_MAX[2]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.006" rgba="0.30 0.32 0.36 1" %s/>' % (L3, r * 0.9, hg))
        s.append('<site name="%s_tip" pos="%g 0 -0.002" size="%g" rgba="0 0 0 0"/>' % (name, L3 * 0.75, r * 1.45))
        s.append('</body></body></body>')
    # the thumb, opposing the fingers: its base under the wrist end of the palm, pointing down and forward; its three
    # joints flex in one plane, curling its tip forward towards the fingertips (the first one, at the base, also swings it
    # back and out of the way to open the hand)
    tb, (T1, T2, T3), tr = THUMB["base"], THUMB["len"], THUMB["r"]
    u, a = THUMB_U, THUMB_A
    s.append('<body name="thumb_0" pos="%s">' % _f(tb))
    s.append('<joint name="thumb_rot" type="hinge" axis="%s" range="-2.1 0.9" damping="0.02" armature="0.002"/>' % _f(a))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.014" rgba="0.70 0.72 0.76 1" %s/>' % (_f(u * T1), tr, hg))
    s.append('<body name="thumb_1" pos="%s"><joint name="thumb_j1" type="hinge" axis="%s" range="-0.1 1.2" damping="0.015" armature="0.002"/>' % (_f(u * T1), _f(a)))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.009" rgba="0.74 0.76 0.80 1" %s/>' % (_f(u * T2), tr * 0.95, hg))
    s.append('<body name="thumb_2" pos="%s"><joint name="thumb_j2" type="hinge" axis="%s" range="0 1.3" damping="0.01" armature="0.002"/>' % (_f(u * T2), _f(a)))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.007" rgba="0.30 0.32 0.36 1" %s/>' % (_f(u * T3), tr * 0.9, hg))
    s.append('<site name="thumb_tip" pos="%s" size="%g" rgba="0 0 0 0"/>' % (_f(u * T3 * 0.75), tr * 1.45))
    s.append('</body></body></body>')
    s.append('</body></body></body></body></body></body>')
    return "\n".join(s)


HAND_JOINTS = ["hx", "hy", "hz", "hyaw", "hpitch", "hroll"]
FINGER_JOINTS = ["thumb_rot", "thumb_j1", "thumb_j2"] + ["%s_j%d" % (f[0], k) for f in FINGERS for k in (1, 2, 3)]


def actuators_xml():
    a = []
    for j in ("hx", "hy", "hz"):
        a.append('<position name="a_%s" joint="%s" kp="4000" kv="160" ctrlrange="-1.5 1.5"/>' % (j, j))
    for j in ("hyaw", "hpitch", "hroll"):
        a.append('<position name="a_%s" joint="%s" kp="60" kv="1.2" ctrlrange="-6.3 6.3"/>' % (j, j))
    for j in FINGER_JOINTS:
        a.append('<position name="a_%s" joint="%s" kp="1.6" kv="0.03" forcerange="-0.9 0.9" ctrlrange="%s"/>' % (j, j, "-2.1 0.9" if j == "thumb_rot" else "-0.2 1.8"))
    a.append('<velocity name="a_belt" joint="belt" kv="400" ctrlrange="-1 1"/>')
    return "\n".join(a)


def object_xml():
    """one object body with geom slots of every kind; each world sizes and places them for its own object"""
    og = 'contype="1" conaffinity="3" condim="6" friction="0.9 0.01 0.002" solref="0.004 1"'
    s = ['<body name="obj" pos="0.6 %g 0.05"><freejoint name="obj"/>' % BELT_Y]
    for k, t in enumerate(SLOT_ORDER):
        size = {"box": "0.02 0.02 0.02", "cylinder": "0.02 0.02", "sphere": "0.02", "capsule": "0.02 0.02", "ellipsoid": "0.02 0.02 0.02"}[t]
        s.append('<geom name="slot%d" type="%s" size="%s" density="500" rgba="0.8 0.4 0.2 1" %s/>' % (k, t, size, og))
    s.append('</body>')
    return "\n".join(s)


def scene_xml(obj_parts=None, rgba=None):
    """the whole cell; with obj_parts, the object's slots are sized for that object (to compile its masses and bounds)"""
    bx, by = BIN_HALF
    cam_xy = look_xyaxes(CAM_POS, CAM_LOOK)
    x = ['<mujoco model="belt_grasp">',
         '<compiler angle="radian" autolimits="true"/>',
         '<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>',
         '<visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>',
         '<default><geom solimp="0.95 0.99 0.001 0.5 2"/></default>',
         '<asset><texture name="belt" type="2d" builtin="checker" rgb1="0.12 0.12 0.13" rgb2="0.16 0.16 0.17" width="64" height="64"/>',
         '<material name="belt" texture="belt" texrepeat="40 2"/></asset>',
         '<worldbody>',
         '<light pos="0.2 0.3 1.6" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="false"/>',
         '<light pos="-0.6 -0.4 1.2" dir="0.4 0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>',
         '<geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" rgba="0.55 0.56 0.58 1" contype="0" conaffinity="0"/>',
         '<geom name="pedestal" type="cylinder" size="0.09 0.375" pos="0 0 -0.375" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0"/>',
         '<geom name="belt_frame" type="box" size="%s" pos="%s" rgba="0.35 0.37 0.40 1" contype="0" conaffinity="0"/>' % (_f((1.25, BELT_W / 2 + 0.03, 0.36)), _f((0.2, BELT_Y, -0.39))),
         # the belt: a long slab on a slide, driven at the belt speed; objects ride on it by friction
         '<body name="belt" pos="%s"><joint name="belt" type="slide" axis="1 0 0" damping="5"/>' % _f((BELT_X0, BELT_Y, 0)),
         '<geom name="belt" type="box" size="%s" pos="0 0 -0.012" mass="40" material="belt" contype="1" conaffinity="2" condim="3" friction="1.0 0.01 0.001"/>' % _f((BELT_LEN / 2, BELT_W / 2, 0.012)),
         '</body>',
         # side rails, so round things stay on the belt
         '<geom name="rail_l" type="box" size="%s" pos="%s" rgba="0.55 0.58 0.62 1" contype="1" conaffinity="2" friction="0.3"/>' % (_f((1.25, 0.008, 0.016)), _f((0.2, BELT_Y + BELT_W / 2 + 0.008, 0.008))),
         '<geom name="rail_r" type="box" size="%s" pos="%s" rgba="0.55 0.58 0.62 1" contype="1" conaffinity="2" friction="0.3"/>' % (_f((1.25, 0.008, 0.016)), _f((0.2, BELT_Y - BELT_W / 2 - 0.008, 0.008))),
         # the bin
         '<body name="bin" pos="%s">' % _f(BIN_C),
         '<geom name="bin_floor" type="box" size="%s" pos="0 0 -0.006" rgba="0.20 0.42 0.70 1" contype="1" conaffinity="2"/>' % _f((bx + BIN_WALL, by + BIN_WALL, 0.006)),
         '<geom name="bin_w1" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((BIN_WALL / 2, by + BIN_WALL, BIN_H / 2)), _f((bx + BIN_WALL / 2, 0, BIN_H / 2))),
         '<geom name="bin_w2" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((BIN_WALL / 2, by + BIN_WALL, BIN_H / 2)), _f((-bx - BIN_WALL / 2, 0, BIN_H / 2))),
         '<geom name="bin_w3" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((bx, BIN_WALL / 2, BIN_H / 2)), _f((0, by + BIN_WALL / 2, BIN_H / 2))),
         '<geom name="bin_w4" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((bx, BIN_WALL / 2, BIN_H / 2)), _f((0, -by - BIN_WALL / 2, BIN_H / 2))),
         '<geom name="bin_stand" type="box" size="%s" pos="0 0 -0.38" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0"/>' % _f((bx, by, 0.37)),
         '</body>',
         # the camera on its pole beside the robot
         '<geom name="cam_pole" type="cylinder" size="0.012 %g" pos="%s" rgba="0.25 0.25 0.28 1" contype="0" conaffinity="0"/>' % ((CAM_POS[2] + 0.75) / 2, _f((CAM_POS[0], CAM_POS[1] - 0.06, (CAM_POS[2] - 0.75) / 2))),
         '<camera name="cam" pos="%s" xyaxes="%s" fovy="%g"/>' % (_f(CAM_POS), _f(cam_xy), CAM_FOVY),
         '<camera name="overview" pos="%s" xyaxes="%s" fovy="45"/>' % (_f((0.95, -0.75, 0.75)), _f(look_xyaxes((0.95, -0.75, 0.75), (0.05, 0.25, 0.02)))),
         '<camera name="closeup" pos="%s" xyaxes="%s" fovy="40"/>' % (_f((0.45, -0.05, 0.42)), _f(look_xyaxes((0.45, -0.05, 0.42), (0.12, 0.30, 0.12)))),
         '<body name="hand_mount" pos="0 0 0">', hand_xml(), '</body>',
         object_xml(),
         '</worldbody>',
         '<actuator>', actuators_xml(), '</actuator>',
         '<sensor>'] + ['<touch name="t_%s" site="%s"/>' % (n, n) for n in TOUCH_NAMES] + ['</sensor>', '</mujoco>']
    xml = "\n".join(x)
    if obj_parts is not None:
        xml = sized_object(xml, obj_parts, rgba)
    return xml


def sized_object(xml, parts, rgba=None):
    """the template's object slots sized and placed for one object (unused slots: a speck inside its first part)"""
    used = dict(assign_slots(parts))
    anchor = parts[0][2]
    for k, t in enumerate(SLOT_ORDER):
        old_size = {"box": "0.02 0.02 0.02", "cylinder": "0.02 0.02", "sphere": "0.02", "capsule": "0.02 0.02", "ellipsoid": "0.02 0.02 0.02"}[t]
        old = '<geom name="slot%d" type="%s" size="%s" density="500" rgba="0.8 0.4 0.2 1"' % (k, t, old_size)
        if k in used:
            tt, size, pos, eul = used[k]
            new = '<geom name="slot%d" type="%s" size="%s" pos="%s" quat="%s" density="500" rgba="%s 1"' % (k, t, _f(size), _f(pos), _f(part_quat(eul)), _f(rgba or (0.8, 0.4, 0.2)))
        else:
            spk = {"box": "0.001 0.001 0.001", "cylinder": "0.001 0.001", "sphere": "0.001", "capsule": "0.001 0.001", "ellipsoid": "0.001 0.001 0.001"}[t]
            new = '<geom name="slot%d" type="%s" size="%s" pos="%s" density="0" rgba="0 0 0 0"' % (k, t, spk, _f(anchor))
        assert old in xml
        xml = xml.replace(old, new)
    return xml
'''
_SOURCES['cell'] = r'''# ── the object library compiled: per object, the slot geoms' sizes, places and colours, and the body's mass ─────────
# Every world has the same model; its object's slots are sized and placed per world (MuJoCo Warp keeps these model
# fields per world). The values come from compiling the cell once per object.
import math

import numpy as np

import scene as S

OBJ_FIELDS_G = ("geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_aabb", "geom_rbound")
OBJ_FIELDS_B = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass")


def library():
    """[(key, label, parts, rgba, test)] train objects first"""
    out = [(k, v[0], v[1], v[2], False) for k, v in S.TRAIN.items()]
    out += [(k, v[0], v[1], v[2], True) for k, v in S.TEST.items()]
    return out


def surface_points(parts, n=192, rng=None):
    """points on an object's surface (its own frame): for the camera's ground truth (where it is, how big, which way)"""
    rng = rng or np.random.default_rng(0)
    pts = []
    vols = []
    for t, size, pos, eul in parts:
        if t == "box":
            v = 8 * size[0] * size[1] * size[2]
        elif t in ("cylinder", "capsule"):
            v = math.pi * size[0] ** 2 * 2 * size[1] + (4 / 3 * math.pi * size[0] ** 3 if t == "capsule" else 0)
        elif t == "sphere":
            v = 4 / 3 * math.pi * size[0] ** 3
        else:
            v = 4 / 3 * math.pi * size[0] * size[1] * size[2]
        vols.append(v ** (2 / 3))
    w = np.array(vols) / sum(vols)
    for (t, size, pos, eul), wi in zip(parts, w):
        k = max(8, int(round(n * wi)))
        u = rng.normal(size=(k, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        if t == "box":
            p = u / np.abs(u).max(1, keepdims=True) * np.array(size)
        elif t == "sphere":
            p = u * size[0]
        elif t == "ellipsoid":
            p = u * np.array(size)
        elif t == "cylinder":
            a = rng.uniform(0, 2 * math.pi, k)
            z = rng.uniform(-1, 1, k)
            cap = rng.uniform(0, 1, k) < 0.35
            r = np.where(cap, np.sqrt(rng.uniform(0, 1, k)), 1.0) * size[0]
            zz = np.where(cap, np.sign(z) * size[1], z * size[1])
            p = np.stack([r * np.cos(a), r * np.sin(a), zz], 1)
        else:  # capsule
            a = rng.uniform(0, 2 * math.pi, k)
            z = rng.uniform(-1, 1, k) * size[1]
            p = np.stack([size[0] * np.cos(a), size[0] * np.sin(a), z], 1)
            ends = rng.uniform(0, 1, k) < 0.3
            p[ends] = u[ends] * size[0] + np.array([0, 0, 1]) * np.sign(u[ends, 2:3]) * size[1]
        R = euler_mat(*[math.radians(e) for e in eul])     # (R = Rz Ry Rx, as scene.part_quat)
        pts.append(p @ R.T + np.array(pos))
    P = np.concatenate(pts)
    idx = rng.choice(len(P), n, replace=len(P) < n)
    return P[idx].astype(np.float32)


def euler_mat(rx, ry, rz):
    """R = Rz * Ry * Rx: a part turned about x, then y, then z (the object's axes)"""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def compile_library(mujoco):
    """the template model, and per object: its slot fields, its surface points, its natural resting poses"""
    tmpl = mujoco.MjModel.from_xml_string(S.scene_xml())
    ob = tmpl.body("obj").id
    gs = [tmpl.geom("slot%d" % k).id for k in range(len(S.SLOT_ORDER))]
    lib = library()
    data = {f: [] for f in OBJ_FIELDS_G + OBJ_FIELDS_B}
    pts, info, models = [], [], []
    for key, label, parts, rgba, test in lib:
        mm = mujoco.MjModel.from_xml_string(S.scene_xml(parts, rgba))
        models.append(mm)
        for f in OBJ_FIELDS_G:
            data[f].append(np.array(getattr(mm, f)[gs], dtype=np.float32))
        for f in OBJ_FIELDS_B:
            data[f].append(np.array(getattr(mm, f)[[ob]], dtype=np.float32))
        pts.append(surface_points(parts))
        info.append({"key": key, "label": label, "test": test, "rgba": list(rgba), "mass": float(mm.body_mass[ob]),
                     "parts": [{"type": t, "size": list(sz), "pos": list(p), "euler": list(e)} for t, sz, p, e in parts]})
    arrays = {f: np.stack(v) for f, v in data.items()}
    return tmpl, models, gs, ob, arrays, np.stack(pts), info
'''
_SOURCES['sims'] = r'''# ── the simulators: MuJoCo Warp (thousands of worlds on a GPU) and MuJoCo on the CPU (threads), same interface ───────
import math
import os
import time

import numpy as np

import scene as S

DEPTH_MAX = 4.0           # the camera's range (m)
N_SUB = 20                 # physics steps (2 ms) per control step (40 ms)
CTRL_DT = 0.04


class WarpSim:
    """n worlds on the GPU; each world's object is its own (sizes, places, masses of the slot geoms per world)"""

    def __init__(self, mjm, lib, n, device="cuda:0", log=None, cam=True):
        import warp as wp
        import mujoco_warp as mjw
        import torch
        wp.config.quiet = True
        wp.init()
        self.wp, self.mjw, self.torch, self.dev, self.n = wp, mjw, torch, device, n
        self.mjm = mjm
        self.lib = lib
        self.cuda = str(device).startswith("cuda")
        tmpl, models, gs, ob, arrays, pts, info = lib
        self.gs, self.ob = gs, ob
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            for f in ("warn_overflow",):
                try:
                    setattr(self.m.opt, f, 0)
                except Exception:
                    pass
            self.d = mjw.make_data(mjm, nworld=n, nconmax=80, njmax=520)
            # the object's model fields, one copy per world
            self.batched = {}
            for f in ("geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_aabb", "geom_rbound", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass"):
                a = getattr(self.m, f)
                host = a.numpy()
                if host.shape[0] != n:
                    host = np.repeat(host[:1], n, axis=0)
                arr = wp.array(host, dtype=a.dtype, device=device)
                setattr(self.m, f, arr)
                self.batched[f] = wp.to_torch(arr)
        self.tdev = wp.device_to_torch(device) if hasattr(wp, "device_to_torch") else (torch.device("cuda:0") if self.cuda else torch.device("cpu"))
        self.lib_t = {f: torch.as_tensor(v, device=self.tdev) for f, v in arrays.items()}
        self.gs_t = torch.as_tensor(gs, device=self.tdev, dtype=torch.long)
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.sens = wp.to_torch(self.d.sensordata)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)
        self.xpos = wp.to_torch(self.d.xpos)
        self.xquat = wp.to_torch(self.d.xquat)
        self.site_xpos = wp.to_torch(self.d.site_xpos)
        self.site_xmat = wp.to_torch(self.d.site_xmat)
        self._qacc = wp.to_torch(self.d.qacc_warmstart) if hasattr(self.d, "qacc_warmstart") else None
        self.graph = None
        t = time.time()
        with wp.ScopedDevice(device):
            mjw.forward(self.m, self.d)
            for _ in range(2):
                mjw.step(self.m, self.d)
            wp.synchronize()
            if self.cuda:
                with wp.ScopedCapture() as cap:
                    for _ in range(N_SUB):
                        mjw.step(self.m, self.d)
                self.graph = cap.graph
        self.rc = None
        if cam:
            self._make_camera()
        if log:
            log("MuJoCo Warp ready on %s: %d worlds (kernels built in %.0f s)." % (device, n, time.time() - t))

    def _make_camera(self):
        wp, mjw = self.wp, self.mjw
        cid = self.mjm.camera("cam").id
        with wp.ScopedDevice(self.dev):
            self.rc = mjw.create_render_context(self.mjm, nworld=self.n, cam_res=[(S.CAM_W, S.CAM_H)], render_rgb=False, render_depth=True,
                                                render_seg=True, use_textures=False, cam_active=[i == cid for i in range(self.mjm.ncam)])
            self._depth = wp.zeros((self.n, S.CAM_H, S.CAM_W), dtype=float)
            self._seg = wp.zeros((self.n, S.CAM_H, S.CAM_W), dtype=wp.vec2i)
        self.depth_t = wp.to_torch(self._depth)
        self.seg_t = wp.to_torch(self._seg)

    def set_objects(self, idx, obj):
        """worlds idx get objects obj (library indices)"""
        g, b = self.gs_t, self.ob
        for f in ("geom_size", "geom_pos", "geom_quat", "geom_rgba", "geom_aabb", "geom_rbound"):
            dst = self.batched[f]
            v = self.lib_t[f][obj]
            dst[idx[:, None], g[None, :]] = v.reshape(v.shape[0], v.shape[1], *dst.shape[2:]).to(dst.dtype)
        for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_subtreemass"):
            dst = self.batched[f]
            dst[idx, b] = self.lib_t[f][obj][:, 0].to(dst.dtype)

    def step(self):
        wp = self.wp
        if self.cuda:
            self.torch.cuda.synchronize()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            with wp.ScopedDevice(self.dev):
                for _ in range(N_SUB):
                    self.mjw.step(self.m, self.d)
        if self.cuda:
            wp.synchronize_device(self.dev)

    def set_rows(self, idx, qpos, qvel, ctrl):
        self.qpos[idx] = qpos.to(self.qpos.dtype)
        self.qvel[idx] = qvel.to(self.qvel.dtype)
        self.ctrl[idx] = ctrl.to(self.ctrl.dtype)
        if self._qacc is not None:
            self._qacc[idx] = 0.0

    def refresh(self):
        if self.cuda:
            self.torch.cuda.synchronize()
        with self.wp.ScopedDevice(self.dev):
            self.mjw.forward(self.m, self.d)
        if self.cuda:
            self.wp.synchronize_device(self.dev)

    def render(self):
        """the camera of every world: depth (m, along the ray) and the geom seen at each pixel (-1: nothing)"""
        wp, mjw = self.wp, self.mjw
        if self.cuda:
            self.torch.cuda.synchronize()
        with wp.ScopedDevice(self.dev):
            mjw.refit_bvh(self.m, self.d, self.rc)
            mjw.render(self.m, self.d, self.rc)
            mjw.get_depth(self.rc, 0, DEPTH_MAX, self._depth)
            mjw.get_segmentation(self.rc, 0, self._seg)
        if self.cuda:
            wp.synchronize_device(self.dev)
        return self.depth_t * DEPTH_MAX, self.seg_t[..., 0]

    def sync(self):
        if self.cuda:
            self.wp.synchronize()


class CpuSim:
    """MuJoCo on the CPU: one simulation per world over threads (MuJoCo releases the GIL); each world uses the
    compiled model of its own object; the camera through MuJoCo's ray casting"""

    def __init__(self, mjm, lib, n, threads=None, log=None, cam=True):
        import mujoco
        import torch
        from concurrent.futures import ThreadPoolExecutor
        self.mj, self.torch = mujoco, torch
        self.mjm = mjm
        tmpl, models, gs, ob, arrays, pts, info = lib
        self.models = models
        self.gs, self.ob = gs, ob
        self.n = n
        self.obj = np.zeros(n, dtype=np.int64)
        self.datas = [mujoco.MjData(models[0]) for _ in range(n)]
        nt = max(1, min(n, threads or (os.cpu_count() or 2)))
        self.ex = ThreadPoolExecutor(nt)
        self.parts = [list(range(i * n // nt, (i + 1) * n // nt)) for i in range(nt)]
        z = lambda *s: np.zeros(s)   # noqa: E731
        self.qpos_np, self.qvel_np, self.ctrl_np = z(n, mjm.nq), z(n, mjm.nv), z(n, mjm.nu)
        self.sens_np, self.xfrc_np = z(n, mjm.nsensordata), z(n, mjm.nbody, 6)
        self.xpos_np, self.xquat_np = z(n, mjm.nbody, 3), z(n, mjm.nbody, 4)
        self.site_xpos_np, self.site_xmat_np = z(n, mjm.nsite, 3), z(n, mjm.nsite, 9)
        T = lambda a: torch.zeros(*a.shape)    # noqa: E731
        self.qpos, self.qvel, self.ctrl, self.sens = T(self.qpos_np), T(self.qvel_np), T(self.ctrl_np), T(self.sens_np)
        self.xfrc = T(self.xfrc_np)
        self.xpos, self.xquat, self.site_xpos, self.site_xmat = T(self.xpos_np), T(self.xquat_np), T(self.site_xpos_np), T(self.site_xmat_np)
        self.tdev = torch.device("cpu")
        self.cam = None
        if cam:
            c = mjm.camera("cam").id
            self.cam_id = c
            self.depth_np = np.zeros((n, S.CAM_H, S.CAM_W), np.float32)
            self.seg_np = -np.ones((n, S.CAM_H, S.CAM_W), np.int32)
            self.depth_t = torch.from_numpy(self.depth_np)
            self.seg_t = torch.from_numpy(self.seg_np)
        if log:
            log("MuJoCo on the CPU: %d worlds over %d threads." % (n, nt))

    def set_objects(self, idx, obj):
        # (a world's data is made for its object's own model: MuJoCo keeps per-model values in it)
        for i, o in zip(idx.tolist(), obj.tolist()):
            if self.obj[i] != int(o) or self.datas[i] is None:
                self.obj[i] = int(o)
                self.datas[i] = self.mj.MjData(self.models[int(o)])

    def _work(self, part):
        mj = self.mj
        for i in part:
            m, d = self.models[self.obj[i]], self.datas[i]
            d.ctrl[:] = self.ctrl_np[i]
            d.xfrc_applied[:] = self.xfrc_np[i]
            for _ in range(N_SUB):
                mj.mj_step(m, d)
            self._pull_one(i)

    def _pull_one(self, i):
        d = self.datas[i]
        self.qpos_np[i] = d.qpos
        self.qvel_np[i] = d.qvel
        self.sens_np[i] = d.sensordata
        self.xpos_np[i] = d.xpos
        self.xquat_np[i] = d.xquat
        self.site_xpos_np[i] = d.site_xpos
        self.site_xmat_np[i] = d.site_xmat

    def step(self):
        self.ctrl_np[:] = self.ctrl.numpy()
        self.xfrc_np[:] = self.xfrc.numpy()
        list(self.ex.map(self._work, self.parts))
        self._pull()

    def _pull(self):
        for a, b in ((self.qpos, self.qpos_np), (self.qvel, self.qvel_np), (self.sens, self.sens_np), (self.xpos, self.xpos_np),
                     (self.xquat, self.xquat_np), (self.site_xpos, self.site_xpos_np), (self.site_xmat, self.site_xmat_np)):
            a.copy_(self.torch.from_numpy(b))

    def set_rows(self, idx, qpos, qvel, ctrl):
        qp, qv, cu = qpos.double().numpy(), qvel.double().numpy(), ctrl.double().numpy()
        for k, i in enumerate(idx.tolist()):
            m, d = self.models[self.obj[i]], self.datas[i]
            self.mj.mj_resetData(m, d)
            d.qpos[:] = qp[k]
            d.qvel[:] = qv[k]
            d.ctrl[:] = cu[k]
            self.mj.mj_forward(m, d)
            self.ctrl_np[i] = cu[k]
            self._pull_one(i)
        self.ctrl[idx] = ctrl.float()
        self._pull()

    def refresh(self):
        pass

    def _ray_one(self, part):
        mj = self.mj
        for i in part:
            m, d = self.models[self.obj[i]], self.datas[i]
            pnt = d.cam_xpos[self.cam_id].copy()
            R = d.cam_xmat[self.cam_id].reshape(3, 3)
            vec = (self._dirs @ R.T).reshape(-1).copy()
            dist = np.zeros(self._nr)
            gid = np.zeros(self._nr, np.int32)
            mj.mj_multiRay(m, d, pnt, vec, None, 1, -1, gid, dist, None, self._nr, 10.0)
            self.depth_np[i] = np.where(gid >= 0, np.minimum(dist * self._pl, DEPTH_MAX), DEPTH_MAX).reshape(S.CAM_H, S.CAM_W)   # planar, as the GPU's
            self.seg_np[i] = gid.reshape(S.CAM_H, S.CAM_W)

    def render(self):
        if not hasattr(self, "_dirs"):
            self._dirs = camera_dirs()
            self._nr = len(self._dirs)
            self._pl = -self._dirs[:, 2]
        list(self.ex.map(self._ray_one, self.parts))
        return self.depth_t, self.seg_t

    def sync(self):
        pass


def camera_dirs():
    """each pixel's ray in the camera's frame (x right, y up, looking along -z), as MuJoCo's cameras"""
    fy = 0.5 * S.CAM_H / math.tan(math.radians(S.CAM_FOVY) / 2)
    xs = (np.arange(S.CAM_W) + 0.5 - S.CAM_W / 2) / fy
    ys = -(np.arange(S.CAM_H) + 0.5 - S.CAM_H / 2) / fy
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, Y, -np.ones_like(X)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


def make_sim(mjm, lib, n, want, log=None, cam=True):
    """(sim, label): the GPU through MuJoCo Warp when it can, else the CPU"""
    import torch
    w = str(want or "auto").lower()
    if w.startswith("nvidia") or w == "auto":
        err = None
        if torch.cuda.is_available():
            try:
                sim = WarpSim(mjm, lib, n, "cuda:0", log, cam)
                return sim, "%s (MuJoCo Warp)" % torch.cuda.get_device_name(0)
            except ImportError as e:
                err = "MuJoCo Warp is not installed on this worker (pip install mujoco-warp): %s" % e
            except Exception as e:      # noqa: BLE001
                err = "MuJoCo Warp could not start on the GPU: %s" % e
        else:
            err = "this PyTorch cannot use an NVIDIA GPU (install the CUDA build of PyTorch)"
        if w.startswith("nvidia"):
            raise RuntimeError("NVIDIA GPU asked, but " + err)
        if log:
            log("On the CPU instead: " + err)
    n = min(n, 256)
    return CpuSim(mjm, lib, n, log=log, cam=cam), "CPU (MuJoCo, %d worlds)" % n
'''
_SOURCES['grasp_env'] = r'''# ── the task: see the object coming on the belt, reach it, grasp it well, put it in the bin ─────────────────────────
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
        self.lifted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ep_ret = z(n)
        self.max_steps = int(EP_SECONDS[stage] / CTRL_DT)
        self.k = 0
        self.assist = {"reach": 0.0, "grasp": 0.5, "place": 0.3}[stage]   # a share of episodes start with the hand over the object

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
        r_reach = 1 - torch.tanh(8 * d)
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
        self.hv_avg = 0.75 * self.hv_avg + 0.25 * hv
        match = (self.hv_avg[:, 0] + self.belt_v).abs() + self.hv_avg[:, 1].abs()
        early = self.steps < 12                                            # (the object settling on the belt)
        self.z_rest = torch.where(self.steps == 12, op[:, 2], self.z_rest)
        r = 0.6 * r_reach + 0.25 * r_align + 0.1 * level.clamp(min=0) - 0.05 * rate - 1.0 * knock - 0.5 * crush - low
        success = torch.zeros_like(held)
        fail = (op[:, 0] < S.EXIT_X) | (op[:, 2] < -0.08)
        in_bin = ((op[:, 0] - S.BIN_C[0]).abs() < S.BIN_HALF[0]) & ((op[:, 1] - S.BIN_C[1]).abs() < S.BIN_HALF[1]) & (op[:, 2] < S.BIN_H) & (op[:, 2] > -0.02)
        if st == "reach":
            ok = (dxy < 0.025) & (dz < 0.025) & (level > 0.85) & ((elong < 0.3) | (align > 0.7)) & (match < 0.08) & ~early
            self.ok_cnt = torch.where(ok, self.ok_cnt + 1, torch.zeros_like(self.ok_cnt))
            success = self.ok_cnt >= 3
            r = r + 1.0 * r_reach * (dxy < 0.04).float() * (1 - torch.tanh(15 * dz)) + 10.0 * success.float()
        else:
            near = (d < 0.05).float()
            r = r + 0.6 * pinch + 0.6 * wrap + 0.3 * near * self.curl[:, 1:].mean(1) * (self.opp > 0.5).float()
            r = r + torch.where(held, 10.0 * lift.clamp(0, 0.14), torch.zeros_like(lift))
            self.hold_cnt = torch.where(lifted, self.hold_cnt + 1, torch.zeros_like(self.hold_cnt))
            r = r + 0.3 * lifted.float()
            grasped = self.hold_cnt >= HOLD_STEPS
            if st == "grasp":
                success = grasped
                r = r + 20.0 * success.float()
            else:
                first = grasped & ~self.lifted_ever
                self.lifted_ever = self.lifted_ever | grasped
                r = r + 5.0 * first.float()
                carrying = held & (lift > 0.03) & self.lifted_ever
                dd = (op - torch.as_tensor(DROP, dtype=torch.float32, device=self.dev)).norm(dim=1)
                r = r + torch.where(carrying, 1.5 * (1 - torch.tanh(4 * dd)), torch.zeros_like(dd))
                over = ((op[:, 0] - S.BIN_C[0]).abs() < S.BIN_HALF[0] - 0.02) & ((op[:, 1] - S.BIN_C[1]).abs() < S.BIN_HALF[1] - 0.02)
                r = r + torch.where(over & self.lifted_ever & (op[:, 2] < 0.25), 0.5 * (1 - self.curl.mean(1)), torch.zeros_like(dd))
                resting = in_bin & (f.max(1).values < 0.2) & (ov.norm(dim=1) < 0.05)
                self.bin_cnt = torch.where(resting, self.bin_cnt + 1, torch.zeros_like(self.bin_cnt))
                success = self.bin_cnt >= 8
                r = r + 30.0 * success.float()
                fail = fail | (self.lifted_ever & ~in_bin & (op[:, 2] < self.z_rest - 0.03))
        r = torch.where(fail, r - 5.0, r)
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
        return obs, priv
'''
_SOURCES['ppo_part'] = r'''import math


class Norm:
    """running mean and variance of what goes into a network"""

    def __init__(self, torch, n, dev):
        self.t = torch
        self.mean = torch.zeros(n, device=dev)
        self.var = torch.ones(n, device=dev)
        self.count = torch.full((n,), 1e-4, device=dev)          # (one count per input: new inputs learn their scale fast)

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-2)).clamp(-5.0, 5.0)     # (what never changes stays near 0)

    def state(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count.cpu()}

    def load(self, st, dev):
        torch = self.t
        n = self.mean.shape[0]
        m, v, c = st["mean"].to(dev), st["var"].to(dev), st["count"]
        c = c.to(dev) if torch.is_tensor(c) else torch.full((m.shape[0],), float(c), device=dev)
        k = m.shape[0]               # (a policy with fewer inputs: the new ones start unscaled, with no count)
        self.mean[:k], self.var[:k], self.count[:k] = m, v, c
        if k < n:
            self.mean[k:], self.var[k:], self.count[k:] = 0.0, 1.0, 1e-4


def mlp(nn, i, sizes, o):
    layers, d = [], i
    for h in sizes:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def make_ac(torch, obs, priv, act, hidden=(512, 256, 128)):
    import torch.nn as nn

    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(nn, obs, hidden, act)
            self.critic = mlp(nn, priv, hidden, 1)
            self.log_std = nn.Parameter(torch.full((act,), math.log(0.35)))    # exploration: the hand's and fingers' moves jitter by a third of their step at first
            with torch.no_grad():     # a new policy starts by keeping still (its last layer near zero)
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()
    return AC()


class PPO:
    def __init__(self, torch, dev, n_env, obs, priv, act, horizon=24, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5,
                 minibatches=4, entropy=0.001, desired_kl=0.01, schedule="fixed"):
        self.t, self.dev = torch, dev
        self.ac = make_ac(torch, obs, priv, act).to(dev)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.lr = lr
        self.on = Norm(torch, obs, dev)
        self.pn = Norm(torch, priv, dev)
        self.n, self.T, self.obs_dim, self.priv_dim, self.act_dim = n_env, horizon, obs, priv, act
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent, self.kl_want = gamma, lam, clip, epochs, minibatches, entropy, desired_kl
        self.schedule = schedule
        T, n = horizon, n_env
        z = lambda *s: torch.zeros(*s, device=dev)   # noqa: E731
        self.b_obs, self.b_priv, self.b_act = z(T, n, obs), z(T, n, priv), z(T, n, act)
        self.b_lp, self.b_val, self.b_rew, self.b_done, self.b_mu = z(T, n), z(T, n), z(T, n), z(T, n), z(T, n, act)

    def act(self, obs, priv, t=None, deterministic=False):
        torch = self.t
        with torch.no_grad():
            x = self.on(obs)
            mu = self.ac.actor(x)
            if deterministic:
                return mu
            std = self.ac.log_std.exp()
            a = mu + std * torch.randn_like(mu)
            if t is not None:
                v = self.ac.critic(self.pn(priv)).squeeze(-1)
                lp = (-0.5 * ((a - mu) / std) ** 2 - self.ac.log_std - 0.9189385).sum(-1)
                self.b_obs[t], self.b_priv[t], self.b_act[t], self.b_lp[t], self.b_val[t], self.b_mu[t] = obs, priv, a, lp, v, mu
            return a

    def record(self, t, rew, done, timeout, priv_next):
        """the reward of step t; an episode cut by the clock is worth what the critic says of where it stopped"""
        torch = self.t
        with torch.no_grad():
            if bool(timeout.any()):
                v_next = self.ac.critic(self.pn(priv_next)).squeeze(-1)
                rew = rew + self.gamma * v_next * timeout.float()
            self.b_rew[t], self.b_done[t] = rew, done.float()

    def update(self, priv_last, critic_only=False):
        """one learning step on the batch gathered (critic_only: the policy holds still while the critic learns new ground)"""
        torch = self.t
        T, n = self.T, self.n
        with torch.no_grad():
            last_v = self.ac.critic(self.pn(priv_last)).squeeze(-1)
            adv = torch.zeros_like(self.b_rew)
            gae = torch.zeros(n, device=self.dev)
            for t in reversed(range(T)):
                nv = last_v if t == T - 1 else self.b_val[t + 1]
                nonterm = 1.0 - self.b_done[t]
                delta = self.b_rew[t] + self.gamma * nv * nonterm - self.b_val[t]
                gae = delta + self.gamma * self.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.b_val
        N = T * n
        obs, priv, act = self.b_obs.reshape(N, -1), self.b_priv.reshape(N, -1), self.b_act.reshape(N, -1)
        lp_old, val_old, mu_old = self.b_lp.reshape(N), self.b_val.reshape(N), self.b_mu.reshape(N, -1)
        adv, ret = adv.reshape(N), ret.reshape(N)
        xo, xp = self.on(obs), self.pn(priv)
        size = N // self.mb
        stats = {"kl": 0.0, "vloss": 0.0, "ploss": 0.0}
        std_old = self.ac.log_std.detach().exp()
        stop, stop_ok = False, True
        for _ in range(self.epochs):
            if stop:
                break
            perm = torch.randperm(N, device=self.dev)
            for i in range(self.mb):
                idx = perm[i * size:(i + 1) * size]
                mu = self.ac.actor(xo[idx])
                ls = self.ac.log_std
                std = ls.exp()
                v = self.ac.critic(xp[idx]).squeeze(-1)
                # how far the policy has moved from the one that gathered this batch (KL)
                with torch.no_grad():
                    kl = (torch.log(std / std_old) + (std_old ** 2 + (mu_old[idx] - mu) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                    if self.schedule == "adaptive":          # the step size follows it
                        if kl > 2 * self.kl_want:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0 < kl < self.kl_want / 2:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                if kl > 4 * self.kl_want and stop_ok and not critic_only:
                    stop = True                                 # moved far enough on this batch: no more steps on it
                    break
                a_ = adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                nlp = (-0.5 * ((act[idx] - mu) / std) ** 2 - ls - 0.9189385).sum(-1)
                ratio = (nlp - lp_old[idx]).exp()
                pl = -torch.min(ratio * a_, ratio.clamp(1 - self.clip, 1 + self.clip) * a_).mean()
                vc = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                ent = (ls + 1.4189385).sum()
                loss = vl if critic_only else pl + 1.0 * vl - self.ent * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
                stats["kl"], stats["vloss"], stats["ploss"] = float(kl), float(vl.detach()), float(pl.detach())
        with torch.no_grad():     # (the input scaling follows what it sees, after the update that used the old one)
            self.on.update(obs)
            self.pn.update(priv)
        stats["std"] = float(self.ac.log_std.detach().exp().mean())
        stats["lr"] = self.lr
        return stats

    def state(self):
        return {"ac": self.ac.state_dict(), "obs_norm": self.on.state(), "priv_norm": self.pn.state(), "lr": self.lr, "opt": self.opt.state_dict()}

    def load(self, st):
        sd = st["ac"]
        same = sd["actor.0.weight"].shape[1] == self.obs_dim and sd["critic.0.weight"].shape[1] == self.priv_dim
        self.ac.load_state_dict(widen(self.t, sd, self.obs_dim, self.priv_dim))
        self.on.load(st["obs_norm"], self.dev)
        self.pn.load(st["priv_norm"], self.dev)
        self.lr = float(st.get("lr") or self.lr)
        # the optimizer's running moments too, when the network is the same shape: training carries on where it was
        if same and st.get("opt"):
            try:
                self.opt.load_state_dict(st["opt"])
            except Exception:      # noqa: BLE001
                pass


def widen(torch, sd, obs, priv):
    """a policy's weights for more inputs than it had: the new inputs' weights start at zero (it acts as before)"""
    sd = dict(sd)
    for key, want in (("actor.0.weight", obs), ("critic.0.weight", priv)):
        w = sd[key]
        if w.shape[1] < want:
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], want - w.shape[1], dtype=w.dtype)], 1)
    return sd


def chunks(rows, budget=40000):
    """the frames in pieces small enough for a telemetry item (at most 60 KB each)"""
    import json
    per = max(1, len(json.dumps(rows[0]))) * 1.15 if rows else 1
    size = max(10, int(budget / per))
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size], i + size >= len(rows)


def policy_stamp(path, name):
    """when a policy was made: the date in its name (grasp_reach_20261119_1926_64.pt), else its file's time"""
    import os
    import re
    import time
    m = re.search(r"(\d{8})_(\d{4})", str(name))
    if m:
        try:
            return time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M"))
        except ValueError:
            pass
    return os.path.getmtime(path)
'''
_SOURCES['train'] = r'''# ── the modes: learn to see, learn to reach / grasp / place, and the test with new objects ─────────────────────────
import base64
import collections
import csv
import glob
import json
import math
import os
import shutil
import time

import numpy as np

import scene as S
from sims import CTRL_DT, make_sim
from grasp_env import ACT_DIM, OBS_DIM, PRIV_DIM, STAGES, DROP, GraspEnv, make_seenet, see_loss
from ppo_part import PPO, chunks, make_ac, Norm, policy_stamp

FORMAT = "esasim.grasp-policy/1"
SEE_FORMAT = "esasim.grasp-see/2"          # (2: the network finds the object's pixels, then its box)
VIEW = "esasim.grasp/1"
STAGE_NAME = {"reach": "Reach", "grasp": "Grasp", "place": "Place"}


# ── what the 3D view needs to draw the cell: every body (tree, joint) and every geom but the object's slots ──────────
def cell_item(mjm, lib):
    import mujoco
    gs = set(int(g) for g in lib[2])
    bodies = []
    for b in range(mjm.nbody):
        jn = None
        if mjm.body_jntnum[b] == 1:
            j = mjm.body_jntadr[b]
            t = int(mjm.jnt_type[j])
            if t in (2, 3):
                jn = {"type": "slide" if t == 2 else "hinge", "axis": [round(float(x), 5) for x in mjm.jnt_axis[j]],
                      "q": int(mjm.jnt_qposadr[j]), "name": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)}
            elif t == 0:
                jn = {"type": "free", "q": int(mjm.jnt_qposadr[j])}
        bodies.append({"name": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b) or "world", "parent": int(mjm.body_parentid[b]),
                       "pos": [round(float(x), 5) for x in mjm.body_pos[b]], "quat": [round(float(x), 6) for x in mjm.body_quat[b]], "joint": jn})
    TYPES = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 0: "plane"}
    geoms = []
    for g in range(mjm.ngeom):
        if g in gs or mjm.geom_rgba[g][3] < 0.02:
            continue
        t = TYPES.get(int(mjm.geom_type[g]))
        if t is None:
            continue
        geoms.append({"b": int(mjm.geom_bodyid[g]), "t": t, "s": [round(float(x), 5) for x in mjm.geom_size[g]],
                      "p": [round(float(x), 5) for x in mjm.geom_pos[g]], "q": [round(float(x), 6) for x in mjm.geom_quat[g]],
                      "c": [round(float(x), 3) for x in mjm.geom_rgba[g]], "n": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or ""})
    cam = mjm.camera("cam")
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    touch = [{"name": n, "body": int(mjm.site_bodyid[mjm.site(n).id]), "p": [round(float(x), 5) for x in mjm.site_pos[mjm.site(n).id]]} for n in S.TOUCH_NAMES]
    return {"format": VIEW, "kind": "cell", "bodies": bodies, "geoms": geoms, "nq": int(mjm.nq),
            "camera": {"pos": [round(float(x), 4) for x in d.cam_xpos[cam.id]], "mat": [round(float(x), 5) for x in d.cam_xmat[cam.id]],
                       "fovy": S.CAM_FOVY, "w": S.CAM_W, "h": S.CAM_H},
            "hand": {"mount": "hand_mount", "grasp_pt": list(S.GRASP_PT)}, "touch": touch,
            "belt": {"y": S.BELT_Y, "w": S.BELT_W, "x0": S.BELT_X0, "len": S.BELT_LEN},
            "bin": {"c": [float(x) for x in S.BIN_C], "half": list(S.BIN_HALF), "h": S.BIN_H, "wall": S.BIN_WALL},
            "obj_body": int(lib[3])}


def object_item(env, i):
    o = env.info[int(env.obj[i])]
    parts = []
    for pt in o["parts"]:
        t, size, pos, eul = (pt["type"], pt["size"], pt["pos"], pt["euler"]) if isinstance(pt, dict) else pt
        parts.append({"t": t, "s": [round(float(x), 5) for x in size], "p": [round(float(x), 5) for x in pos],
                      "q": [round(float(x), 6) for x in S.part_quat(eul)]})
    return {"key": o["key"], "label": o["label"], "test": bool(o["test"]), "rgba": [round(float(x), 3) for x in o["rgba"]], "parts": parts}


def frame_row(env, i, t):
    """[t, qpos..., touch (9, N), the camera's box: centre (3), half sizes (3), yaw]"""
    q = env.sim.qpos[i].float().cpu().numpy()
    f = env.touches()[i].cpu().numpy()
    ec, ee, ey = env.est_c[i].cpu().numpy(), env.est_e[i].cpu().numpy(), env.est_y[i].cpu().numpy()
    psi = 0.5 * math.atan2(float(ey[1]), float(ey[0]))
    return ([round(t, 2)] + [round(float(x), 4) for x in q] + [round(float(x), 2) for x in f]
            + [round(float(x), 3) for x in ec] + [round(float(x), 3) for x in ee] + [round(psi, 3)])


def camera_item(env, i, box=True):
    """the camera's picture as the network sees it (its colours, shaded), and the box it found, for the view"""
    img = env.last_img[i]
    rgb = (img[3:6].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    seg = env.last_seg[i].cpu().numpy()
    bg = np.array([28, 30, 34], np.uint8)
    rgb = np.where((seg < 0)[..., None], bg, rgb)
    item = {"format": VIEW, "kind": "camera", "w": S.CAM_W, "h": S.CAM_H, "rgb": base64.b64encode(rgb.tobytes()).decode()}
    return item


def episode_result(info, i, stage):
    return {"success": bool(info["success"][i]), "fail": bool(info["fail"][i])}


def results_text(stage, ok, fail):
    if ok:
        return {"reach": "reached it", "grasp": "grasped it and held it", "place": "put it in the bin"}[stage]
    return "dropped it or let it go by" if fail else "ran out of time"


# ── checkpoints to carry on from ────────────────────────────────────────────────────────────────────────────────
def find_files(ctx, port, checkpoint, fmt):
    import torch
    out = []

    def consider(path, name, where):
        try:
            st = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:     # noqa: BLE001
            ctx.log("Skipped %s: %s" % (name, e))
            return
        if st.get("format") != fmt:
            return
        out.append({"st": st, "name": str(name), "where": where, "path": path, "stamp": policy_stamp(path, name),
                    "steps": int(st.get("steps") or 0), "stage": st.get("stage")})
    for f in (ctx.inputs.get(port) or []):
        if str(f.name).endswith(".pt"):
            consider(f.path, f.name, "dataset")
    here = os.getcwd()
    jobs = os.path.dirname(os.path.dirname(here))
    if os.path.basename(jobs).lower() == "jobs":
        for pth in sorted(glob.glob(os.path.join(jobs, "*", "*", checkpoint)), key=os.path.getmtime, reverse=True)[:40]:
            if os.path.dirname(os.path.abspath(pth)) == os.path.abspath(here):
                continue
            consider(pth, "the one run #%s left on this machine" % os.path.basename(os.path.dirname(os.path.dirname(pth))), "machine")
    return out


def load_see(ctx, dev, need=False):
    import torch
    found = find_files(ctx, "see", "see_checkpoint.pt", SEE_FORMAT)
    found += find_files(ctx, "policy", "see_checkpoint.pt", SEE_FORMAT)
    if not found:
        if need:
            raise RuntimeError("No camera network yet: run Learn to see first.")
        return None, None
    found.sort(key=lambda x: (x["stamp"]), reverse=True)
    pick = found[0]
    net = make_seenet(torch).to(dev)
    net.load_state_dict(pick["st"]["net"])
    net.eval()
    return net, pick


def time_budget(ctx, minutes):
    left = ctx.time_left() if hasattr(ctx, "time_left") else None
    if left is not None and left / 60.0 - 3.0 < minutes:
        cut = max(1.0, left / 60.0 - 3.0)
        ctx.log("The run stops after %.0f minutes (Stop after): learning ends at %.0f minutes, so what it learned is kept." % (left / 60.0, cut))
        return cut
    return minutes


def finish_asked(ctx, ch):
    for m in ctx.messages(ch):
        d = m.get("data") if isinstance(m, dict) else None
        if isinstance(d, dict) and str(d.get("control", "")).lower() in ("finish now", "finish"):
            return True
    return False


def build(ctx, n, stage, cam=True, seed=1, test_objects=False):
    import mujoco
    import cell
    ctx.log("Building the cell: the belt, the bin, the camera, the hand and %d objects (%d to learn with, %d kept for the test)." % (len(S.TRAIN) + len(S.TEST), len(S.TRAIN), len(S.TEST)))
    lib = cell.compile_library(mujoco)
    sim, label = make_sim(lib[0], lib, n, ctx.params.get("accelerator") or "auto", ctx.log, cam=cam)
    env = GraspEnv(__import__("torch"), sim, lib, stage=stage, seed=seed, test_objects=test_objects)
    return lib, sim, label, env


# ── Learn to see ────────────────────────────────────────────────────────────────────────────────────────────────
def see_run(ctx):
    import torch
    p = ctx.params
    n = int(p.get("worlds") or 1024)
    minutes = time_budget(ctx, float(p.get("minutes") or 30))
    seed = int(p.get("seed") or 1)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    lib, sim, label, env = build(ctx, n, "reach", seed=seed)
    n, dev = sim.n, sim.tdev
    net = make_seenet(torch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    prev = None if str(p.get("training") or "").lower().startswith("start") else load_see(ctx, dev)[1]
    if prev:
        net.load_state_dict(prev["st"]["net"])
        ctx.log("Continuing from %s." % prev["name"])
    ctx.log("Learning to see on %s: %d cells at once, for up to %.0f minutes. The camera (%dx%d, depth and colour) looks at the belt; "
            "the network learns where the object is, how big, and which way it lies." % (label, n, minutes, S.CAM_W, S.CAM_H))
    ctx.telemetry(ch_bot, cell_item(lib[0], lib))
    cap = min(16384, 12 * n)
    B_img = torch.zeros(cap, 6, S.CAM_H, S.CAM_W, dtype=torch.float16, device=dev)
    B_c, B_e, B_y, B_s = torch.zeros(cap, 3, device=dev), torch.zeros(cap, 3, device=dev), torch.zeros(cap, 2, device=dev), torch.zeros(cap, dtype=torch.bool, device=dev)
    B_m = torch.zeros(cap, S.CAM_H, S.CAM_W, dtype=torch.bool, device=dev)       # which pixels are the object (the camera's segmentation)
    fill, head = 0, 0
    goal = torch.zeros(n, 3, device=dev)
    env.reset_all()
    curve = open(ctx.output("curve", "see.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "images", "centre_cm", "size_cm", "yaw_deg", "test_centre_cm", "test_size_cm", "test_yaw_deg", "seen_pct"])
    t0, it, last_tel, best = time.time(), 0, 0.0, 1e9
    test_m = None
    out_path = os.path.join(os.getcwd(), "see_checkpoint.pt")
    seen_total = 0

    def wander():
        """the hand wanders about the object (hiding it, now and then), the fingers open and close"""
        nonlocal goal
        new = (env.steps % 25 == 0)
        if bool(new.any()):
            _, _, c, _, _ = env.truth()
            gg = c + torch.stack([(torch.rand(n, device=dev) - 0.5) * 0.3, (torch.rand(n, device=dev) - 0.5) * 0.3, torch.rand(n, device=dev) * 0.3], -1)
            far = torch.rand(n, device=dev) < 0.3
            gg = torch.where(far[:, None], torch.as_tensor(S.HOME, dtype=torch.float32, device=dev).expand(n, 3), gg)
            goal = torch.where(new[:, None], gg, goal)
        a = torch.zeros(n, ACT_DIM, device=dev)
        a[:, :3] = ((goal - env.tgt_p) / 0.018).clamp(-1, 1)
        a[:, 3:] = torch.randn(n, ACT_DIM - 3, device=dev) * 0.5
        return a

    def evaluate(img, c, e, y, s):
        with torch.no_grad():
            out = []
            for k in range(0, img.shape[0], 256):
                out.append(net(img[k:k + 256].float()))
            cc = torch.cat([o[0] for o in out]); ee = torch.cat([o[1] for o in out]); yy = torch.cat([o[2] for o in out]); ll = torch.cat([o[3] for o in out])
        m = s.float()
        el = ((e[:, 0] - e[:, 1]) / e[:, 0].clamp(min=1e-3)) > 0.3
        ce = float(((cc - c).norm(dim=1) * m).sum() / m.sum().clamp(min=1) * 100)
        se = float(((ee - e).abs().mean(1) * m).sum() / m.sum().clamp(min=1) * 100)
        pa = 0.5 * torch.atan2(yy[:, 1], yy[:, 0]); pb = 0.5 * torch.atan2(y[:, 1], y[:, 0])
        dy = torch.remainder(pa - pb + math.pi / 2, math.pi) - math.pi / 2
        mm = m * el.float()
        ye = float((dy.abs() * mm).sum() / mm.sum().clamp(min=1) * 180 / math.pi)
        acc = float(((torch.sigmoid(ll) > 0.5) == s).float().mean() * 100)
        return ce, se, ye, acc, (cc, ee, yy)

    def save(note, stats):
        torch.save({"format": SEE_FORMAT, "net": net.state_dict(), "note": note, "images": seen_total, "stats": stats,
                    "cam": {"w": S.CAM_W, "h": S.CAM_H, "fovy": S.CAM_FOVY}}, out_path)

    stop = "time"
    stats = {}
    while time.time() - t0 < minutes * 60:
        if finish_asked(ctx, ch_ctl):
            stop = "asked"
            break
        for _ in range(3):
            env.step(wander())
            dn = env.steps >= 120
            idx = dn.nonzero().flatten()
            if len(idx):
                env.reset(idx)
                sim.refresh()
        img, objpix = env.perceive()
        _, _, c, e, y = env.truth()
        s = objpix >= 6
        k = img.shape[0]
        ids = (torch.arange(k, device=dev) + head) % cap
        B_img[ids] = img.half(); B_c[ids] = c; B_e[ids] = e; B_y[ids] = y; B_s[ids] = s
        B_m[ids] = torch.isin(env.last_seg, torch.as_tensor(env.gs, device=dev))
        head = (head + k) % cap
        fill = min(cap, fill + k)
        seen_total += k
        net.train()
        for _ in range(max(2, min(16, k // 128))):
            j = torch.randint(fill, (256,), device=dev)
            loss, st_ = see_loss(torch, net(B_img[j].float(), mask_out=True), B_c[j], B_e[j], B_y[j], B_s[j], B_m[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        it += 1
        if it % 25 == 0 or time.time() - last_tel > 5:
            ce, se, ye, acc, _ = evaluate(img, c, e, y, s)
            # the objects it never learned on: a look at each of them every so often
            if test_m is None or it % 100 == 0:
                env.pool = env.test_ids
                env.reset_all()
                for _ in range(15):
                    env.step(wander())
                timg, _ = env.perceive()
                _, _, tc, te, ty = env.truth()
                ts = torch.isin(env.last_seg, torch.as_tensor(env.gs, device=dev)).sum((1, 2)) >= 6
                tce, tse, tye, tacc, pred = evaluate(timg, tc, te, ty, ts)
                test_m = (tce, tse, tye)
                # the camera picture of one cell, with what the network made of it
                env.est_c[0], env.est_e[0], env.est_y[0] = pred[0][0], pred[1][0], pred[2][0]
                ci = camera_item(env, 0)
                ci["label"] = "Unseen object: %s · centre off by %.1f cm" % (env.info[int(env.obj[0])]["label"], float((pred[0][0] - tc[0]).norm()) * 100)
                ci["box"] = {"c": [round(float(x), 4) for x in pred[0][0]], "e": [round(float(x), 4) for x in pred[1][0]], "yaw": round(0.5 * math.atan2(float(pred[2][0][1]), float(pred[2][0][0])), 3)}
                ctx.telemetry(ch_bot, ci)
                env.pool = env.train_ids
                env.reset_all()
            last_tel = time.time()
            el = (time.time() - t0) / 60
            stats = {"centre_cm": round(ce, 2), "size_cm": round(se, 2), "yaw_deg": round(ye, 1), "test_centre_cm": round(test_m[0], 2),
                     "test_size_cm": round(test_m[1], 2), "test_yaw_deg": round(test_m[2], 1), "seen_pct": round(acc, 1)}
            row = {"iteration": it, "minutes": round(el, 2), "images_k": round(seen_total / 1000, 1), **stats}
            ctx.telemetry(ch_train, row)
            cw.writerow([row["minutes"], it, seen_total, ce, se, ye, test_m[0], test_m[1], test_m[2], acc])
            curve.flush()
            ctx.progress(min(99, 100 * (time.time() - t0) / (minutes * 60)), "centre off by %.1f cm (unseen objects %.1f cm) · %.0f k images" % (ce, test_m[0], seen_total / 1000))
            score = test_m[0] + 0.5 * test_m[1] + 0.05 * test_m[2]
            if score < best and it > 50:
                best = score
                save("best so far", stats)
    curve.close()
    if not os.path.exists(out_path):
        save("final", stats)
    name = "grasp_see_%s.pt" % time.strftime("%Y%m%d_%H%M")
    shutil.copyfile(out_path, ctx.output("policy", name))
    ctx.log("Camera network written: %s. On objects it never saw: centre off by %.1f cm, size by %.1f cm, long axis by %.0f°." % (name, *(test_m or (0, 0, 0))))
    ctx.log("Done (%s): %d iterations, %.0f thousand images." % ("time is up" if stop == "time" else "finished on request", it, seen_total / 1000))
    return {"test_centre_cm": round(test_m[0], 2) if test_m else None, "images_k": round(seen_total / 1000)}


# ── Learn to reach / grasp / place ──────────────────────────────────────────────────────────────────────────────
def learn_run(ctx, stage):
    import torch
    p = ctx.params
    n = int(p.get("worlds") or 2048)
    minutes = time_budget(ctx, float(p.get("minutes") or 60))
    seed = int(p.get("seed") or 1)
    replay_every = float(p.get("replay_every") or 20)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    lib, sim, label, env = build(ctx, n, stage, seed=seed)
    n, dev = sim.n, sim.tdev
    see, sp = load_see(ctx, dev)
    if see is not None:
        env.see = see
        ctx.log("It sees through the camera network (%s)." % sp["name"])
    else:
        ctx.log("No camera network yet (Learn to see): it learns with the object's true box, blurred as a camera would. Run Learn to see first for the real thing.")
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, ACT_DIM, horizon=32, lr=3e-4, entropy=0.002, schedule="adaptive")
    steps_done, warm = 0, 0
    order = {s: k for k, s in enumerate(STAGES)}
    found = [] if str(p.get("training") or "").lower().startswith("start") else find_files(ctx, "policy", "policy_checkpoint.pt", FORMAT)
    found = [f for f in found if order.get(f["stage"], 9) <= order[stage]]
    found.sort(key=lambda f: (order.get(f["stage"], 0), f["steps"], f["stamp"]), reverse=True)
    if found:
        pick = found[0]
        st = pick["st"]
        algo.load(st)
        steps_done = int(st.get("steps") or 0)
        if pick["stage"] != stage:
            warm = 20
            with torch.no_grad():
                algo.ac.log_std.fill_(math.log(0.3))
            ctx.log("Starting from the %s policy (%s, %.0f%% success there): its critic first learns what this stage is worth." % (STAGE_NAME[pick["stage"]], pick["name"], float(st.get("score") or 0)))
        else:
            ctx.log("Continuing from %s (%.0f%% success, %.0f million steps)." % (pick["name"], float(st.get("score") or 0), steps_done / 1e6))
    else:
        ctx.log("No policy to carry on from: a new one learns from nothing.")
    ctx.log("Learning to %s on %s: %d cells at once, for up to %.0f minutes." % (stage, label, n, minutes))
    ctx.telemetry(ch_bot, cell_item(lib[0], lib))
    env.reset_all()
    env.perceive()
    a_obs, a_priv = env.obs()
    curve = open(ctx.output("curve", "%s.csv" % stage), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "success_pct", "fail_pct", "episode_s", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(300, min(3000, n)))
    per_obj = collections.defaultdict(lambda: collections.deque(maxlen=60))
    t0, last_tel, last_replay, it, best, stop = time.time(), 0.0, -1e9, 0, -1.0, "time"
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")
    rec, rec_ep, rec_obj = [], 0, object_item(env, 0)
    dist_ = collections.deque(maxlen=3000)          # how far the grasp point ended from where it should be (cm)

    def score():
        return 100.0 * sum(h[0] for h in hist) / len(hist) if len(hist) >= 50 else 0.0

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "stage": stage, "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": ACT_DIM, "score": score(), "note": note,
                   "steps": steps_done, "device": label, "minutes": (time.time() - t0) / 60, "ctrl_dt": CTRL_DT,
                   "per_object": {k: round(100 * sum(v) / max(1, len(v))) for k, v in per_obj.items()}})
        torch.save(st, out_path)

    try:
        while time.time() - t0 < minutes * 60:
            if finish_asked(ctx, ch_ctl):
                stop = "asked"
                break
            tt = time.time()
            for t in range(algo.T):
                act = algo.act(a_obs, a_priv, t)
                rew, done, info = env.step(act)
                o2, p2 = env.obs()
                algo.record(t, rew, done, info["timeout"], p2)
                if int(env.steps[0]) % 2 == 0:
                    rec.append(frame_row(env, 0, float(env.steps[0]) * CTRL_DT))
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    sc_, fl_, ln_ = info["success"][idx].tolist(), info["fail"][idx].tolist(), env.steps[idx].tolist()
                    dist_.extend((info["d"][idx] * 100).tolist())
                    rt_, ob_ = env.ep_ret[idx].tolist(), env.obj[idx].tolist()
                    for a_, b_, c_, d_, o_ in zip(sc_, fl_, ln_, rt_, ob_):
                        hist.append((1.0 if a_ else 0.0, 1.0 if b_ else 0.0, c_ * CTRL_DT, d_))
                        per_obj[env.info[o_]["key"]].append(1.0 if a_ else 0.0)
                    if bool(done[0]):
                        if rec and time.time() - last_replay > replay_every and len(rec) > 10:
                            last_replay = time.time()
                            rec_ep += 1
                            ok, fl = bool(info["success"][0]), bool(info["fail"][0])
                            lab = "Episode %d (%s): %s, %s after %.1f s" % (rec_ep, STAGE_NAME[stage], rec_obj["label"], results_text(stage, ok, fl), float(env.steps[0]) * CTRL_DT)
                            for k, part, last in chunks(rec):
                                ctx.telemetry(ch_bot, {"format": VIEW, "kind": "training", "ep": rec_ep, "seq": k, "last": last, "frames": part,
                                                       "object": rec_obj if k == 0 else None, "label": lab, "result": {"success": ok, "fail": fl}})
                        rec = []
                    env.reset(idx)
                    sim.refresh()
                    if bool(done[0]):
                        rec_obj = object_item(env, 0)
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
            stats = algo.update(a_priv, critic_only=it < warm)
            it += 1
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - tt)
            sc = score()
            if sc > best and len(hist) >= 50 and it > warm + 10:
                best = sc
                save("best so far")
            if time.time() - last_tel > 3.0 or it == 1:
                last_tel = time.time()
                k_ = max(1, len(hist))
                fail = 100.0 * sum(h[1] for h in hist) / k_
                eps = sum(h[2] for h in hist) / k_
                ret = sum(h[3] for h in hist) / k_
                row = {"iteration": it, "minutes": round((time.time() - t0) / 60, 2), "steps_m": round(steps_done / 1e6, 2), "success": round(sc, 1),
                       "failed": round(fail, 1), "episode_s": round(eps, 1), "reward": round(ret, 1), "steps_per_s": round(sps),
                       "std": round(stats["std"], 3), "kl": round(stats["kl"], 4), "end_dist_cm": round(sum(dist_) / max(1, len(dist_)), 1)}
                ctx.telemetry(ch_train, row)
                cw.writerow([row["minutes"], it, steps_done, row["success"], row["failed"], row["episode_s"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * (time.time() - t0) / (minutes * 60 + 1)), "%s: %.0f%% success · ends %.1f cm from the object · %.1f M steps" % (STAGE_NAME[stage], sc, row["end_dist_cm"], steps_done / 1e6))
    finally:
        curve.close()
    if best < 0 or score() >= best:
        save("final")
    final = torch.load(out_path, map_location="cpu", weights_only=False)
    name = "grasp_%s_%s_%02d.pt" % (stage, time.strftime("%Y%m%d_%H%M"), int(round(max(0.0, float(final.get("score") or 0)))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    if sp:      # (and the camera network it saw with, so that the next stage has both wherever it runs)
        shutil.copyfile(sp["path"], ctx.output("policy", "grasp_see_%s.pt" % time.strftime("%Y%m%d_%H%M", time.localtime(sp["stamp"]))))
    po = final.get("per_object") or {}
    if po:
        ctx.log("Success per object: " + ", ".join("%s %d%%" % (S.TRAIN[k][0] if k in S.TRAIN else k, v) for k, v in sorted(po.items(), key=lambda x: -x[1])))
    ctx.log("Policy written: %s (%s success %.0f%%)." % (name, STAGE_NAME[stage], float(final.get("score") or 0)))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ("time is up" if stop == "time" else "finished on request", it, steps_done / 1e6, (time.time() - t0) / 60))
    return {"success_pct": round(float(final.get("score") or 0), 1), "stage": stage, "steps_m": round(steps_done / 1e6, 1)}


# ── the test: objects it never learned on, chosen in the view ──────────────────────────────────────────────────
POSES = {"upright": 0, "on its side": 1, "side": 1, "any": 2, "random": 2, "tumbled": 2}


def test_run(ctx):
    import torch
    p = ctx.params
    minutes = time_budget(ctx, float(p.get("minutes") or 30))
    ch_bot, ch_ev, ch_ctl = p.get("robot_channel") or "robot", p.get("events_channel") or "events", p.get("control_channel") or "control"
    seed = int(p.get("seed") or 1)
    torch.manual_seed(seed)
    pace = float(p.get("pace") or 1.0)
    auto = str(p.get("autoplay") or "Yes").lower().startswith("y")
    lib, sim, label, env = build(ctx, 1, "place", seed=seed, test_objects=True)
    dev = sim.tdev
    env.assist = 0.0
    see, sp = load_see(ctx, dev, need=True)
    env.see = see
    pols = find_files(ctx, "policy", "policy_checkpoint.pt", FORMAT)
    order = {s: k for k, s in enumerate(STAGES)}
    pols.sort(key=lambda f: (order.get(f["stage"], 0), f["stamp"]), reverse=True)
    if not pols:
        raise RuntimeError("No policy yet: run Learn to reach, Learn to grasp and Learn to place first.")
    pol = pols[0]
    st = pol["st"]
    env.stage = pol["stage"]
    env.max_steps = int(16.0 / CTRL_DT)
    ac = make_ac(torch, st["obs_dim"], st["priv_dim"], st["act_dim"]).to(dev)
    ac.load_state_dict(st["ac"])
    ac.eval()
    on = Norm(torch, st["obs_dim"], dev)
    on.load(st["obs_norm"], dev)
    ctx.log("The test: %s policy (%s, %.0f%% in training), camera network %s, on %s." % (STAGE_NAME[pol["stage"]], pol["name"], float(st.get("score") or 0), sp["name"], label))
    objs = [o for o in env.info if o["test"]]
    if str(p.get("objects") or "").lower().startswith("all"):
        objs = env.info
    cell = cell_item(lib[0], lib)
    cell["catalog"] = [{"key": o["key"], "label": o["label"], "test": bool(o["test"]), "rgba": [round(float(x), 3) for x in o["rgba"]]} for o in env.info]
    ctx.telemetry(ch_bot, cell)
    ctx.telemetry(ch_ev, {"text": "Ready: choose an object and how it lies on the belt (or let it pick them)."})
    queue = collections.deque()
    rows = []
    out_csv = open(ctx.output("curve", "test.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(out_csv)
    cw.writerow(["trial", "object", "seen_in_training", "pose", "yaw_deg", "belt_speed", "grasped", "placed", "result", "seconds", "max_force_N"])
    t0, trial, cyc = time.time(), 0, 0
    idle_since = time.time()
    keys = {o["key"]: k for k, o in enumerate(env.info)}

    def read_controls():
        for m in ctx.messages(ch_ctl):
            d = m.get("data") if isinstance(m, dict) else None
            if not isinstance(d, dict):
                continue
            c = str(d.get("control", "")).lower()
            if c in ("finish now", "finish"):
                return "finish"
            if c in ("drop", "pick", "test", "object"):
                v = d.get("value") if isinstance(d.get("value"), dict) else d
                k = keys.get(str(v.get("object") or ""))
                if k is None:
                    ctx.telemetry(ch_ev, {"text": "Unknown object: %s" % v.get("object")})
                    continue
                queue.append({"obj": k, "pose": POSES.get(str(v.get("pose") or "any").lower(), 2),
                              "yaw": float(v.get("yaw")) if v.get("yaw") not in (None, "") else None,
                              "speed": float(v.get("speed")) if v.get("speed") not in (None, "") else None})
        return None

    stop = "time"
    while time.time() - t0 < minutes * 60:
        if read_controls() == "finish":
            stop = "asked"
            break
        if not queue:
            if auto and time.time() - idle_since > float(p.get("idle") or 8):
                o = objs[cyc % len(objs)]
                cyc += 1
                queue.append({"obj": keys[o["key"]], "pose": int(np.random.randint(3)), "yaw": None, "speed": None})
            else:
                time.sleep(0.3)
                continue
        req = queue.popleft()
        trial += 1
        idx = torch.zeros(1, dtype=torch.long, device=dev)
        yaw = None if req["yaw"] is None else math.radians(req["yaw"])
        env.reset(idx, obj=torch.tensor([req["obj"]], device=dev), mode=req["pose"], yaw=yaw, speed=req["speed"])
        sim.refresh()
        env.perceive()
        o = env.info[req["obj"]]
        pose_txt = ["upright", "on its side", "tumbled"][req["pose"]]
        ctx.telemetry(ch_ev, {"text": "Trial %d: %s%s, %s, belt at %.0f cm/s" % (trial, o["label"], "" if o["test"] else " (seen in training)", pose_txt, float(env.belt_v[0]) * 100)})
        ctx.telemetry(ch_bot, {"format": VIEW, "kind": "object", "trial": trial, "object": object_item(env, 0)})
        frames, last_send, wall0 = [], time.time(), time.time()
        grasped, maxf, result = False, 0.0, "ran out of time"
        cam_k = 0
        while True:
            obs, _ = env.obs()
            with torch.no_grad():
                a = ac.actor(on(obs))
            r, done, info = env.step(a)
            grasped = grasped or bool(env.lifted_ever[0]) or bool(env.hold_cnt[0] >= 25)
            maxf = max(maxf, float(env.touches()[0].max()))
            ts = float(env.steps[0]) * CTRL_DT
            frames.append(frame_row(env, 0, ts))
            if env.k % 4 == 0:
                cam_k += 1
                if cam_k % 2 == 0:
                    ci = camera_item(env, 0)
                    ci["box"] = {"c": [round(float(x), 4) for x in env.est_c[0]], "e": [round(float(x), 4) for x in env.est_e[0]],
                                 "yaw": round(0.5 * math.atan2(float(env.est_y[0][1]), float(env.est_y[0][0])), 3)}
                    ctx.telemetry(ch_bot, ci)
            if time.time() - last_send > 0.25:
                ctx.telemetry(ch_bot, {"format": VIEW, "kind": "live", "trial": trial, "frames": frames, "label": "Trial %d: %s, %s" % (trial, o["label"], pose_txt)})
                frames, last_send = [], time.time()
            if pace > 0:
                lag = wall0 + ts / pace - time.time()
                if lag > 0:
                    time.sleep(lag)
            if bool(done[0]):
                if bool(info["success"][0]):
                    result = "in the bin" if env.stage == "place" else "grasped and held"
                elif bool(info["fail"][0]):
                    result = "dropped" if grasped else "missed it"
                break
            if read_controls() == "finish":
                stop = "asked"
                break
        if frames:
            ctx.telemetry(ch_bot, {"format": VIEW, "kind": "live", "trial": trial, "frames": frames, "label": "Trial %d: %s, %s" % (trial, o["label"], pose_txt)})
        placed = result == "in the bin"
        ctx.telemetry(ch_ev, {"text": "Trial %d, %s: %s%s (%.1f s, grip up to %.1f N)" % (trial, o["label"], result, "" if placed or not grasped else ", it had grasped it", ts, maxf),
                              "trial": trial, "object": o["key"], "result": result, "grasped": grasped, "placed": placed})
        rows.append((o["key"], grasped, placed))
        cw.writerow([trial, o["label"], not o["test"], pose_txt, req["yaw"], round(float(env.belt_v[0]), 3), grasped, placed, result, round(ts, 1), round(maxf, 1)])
        out_csv.flush()
        ctx.progress(min(99, 100 * (time.time() - t0) / (minutes * 60)), "%d trials: %d grasped, %d in the bin" % (len(rows), sum(r[1] for r in rows), sum(r[2] for r in rows)))
        idle_since = time.time()
        if stop == "asked":
            break
    out_csv.close()
    g_, p_ = sum(r[1] for r in rows), sum(r[2] for r in rows)
    ctx.log("Test done: %d trials, %d grasped, %d put in the bin." % (len(rows), g_, p_))
    return {"trials": len(rows), "grasped": g_, "placed": p_}


def run(ctx):
    what = str(ctx.params.get("mode") or "Learn to reach").lower()
    if "see" in what:
        return see_run(ctx)
    if "test" in what:
        return test_run(ctx)
    for s in STAGES:
        if s in what:
            return learn_run(ctx, s)
    return learn_run(ctx, "reach")
'''


def _load():
    for name in ['scene', 'cell', 'sims', 'grasp_env', 'ppo_part', 'train']:
        if name in _sys.modules and getattr(_sys.modules[name], "_grasp_block", False):
            continue
        mod = _types.ModuleType(name)
        mod.__file__ = "<grasp:%s>" % name
        mod._grasp_block = True
        _sys.modules[name] = mod
        exec(compile(_SOURCES[name], mod.__file__, "exec"), mod.__dict__)


def _gpu_torch(ctx):
    """This worker's Python may have PyTorch built for the CPU only, though it has an NVIDIA GPU: then a CUDA build of
    the same PyTorch goes into the worker's data folder once (cache/torch-cuda, about 2.5 GB), used by this block only
    (the worker's own Python is left as it is)."""
    import os
    import shutil
    import subprocess
    want = str(ctx.params.get("accelerator") or "auto").lower()
    if want.startswith("cpu") or "torch" in _sys.modules:
        return
    if not shutil.which("nvidia-smi"):
        return
    here = os.path.abspath(os.getcwd())
    jobs = os.path.dirname(os.path.dirname(here))
    base = os.path.dirname(jobs) if os.path.basename(jobs).lower() == "jobs" else os.path.dirname(here)
    target = os.path.join(base, "cache", "torch-cuda")
    marker = target + ".failed"
    probe = "import torch; x = torch.ones(2, device='cuda') * 2; print(torch.__version__, float(x.sum()))"

    why = [""]

    def works(path):
        env = dict(os.environ)
        if path:
            env["PYTHONPATH"] = path + os.pathsep + env.get("PYTHONPATH", "")
        try:
            r = subprocess.run([_sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, env=env)
            why[0] = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
            why[0] = why[0][0]
            return r.returncode == 0 and "4.0" in r.stdout
        except Exception as e:      # noqa: BLE001
            why[0] = str(e)
            return False
    if os.path.isdir(os.path.join(target, "torch")) and works(target):
        _sys.path.insert(0, target)
        return
    if works(None):
        return
    try:      # (a GPU this worker's runs cannot reach: nothing to install for)
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60)
        if smi.returncode != 0 or "GPU" not in smi.stdout:
            ctx.log("This worker's runs cannot reach its NVIDIA GPU (nvidia-smi: %s): on the CPU. (Windows: the worker's "
                    "service account needs access to the GPU.)" % (smi.stdout or smi.stderr).strip()[:200])
            return
    except Exception:          # noqa: BLE001
        pass
    try:      # (tried in the last day and the GPU could not be used: not again, the run goes on the CPU)
        import time as _t
        if os.path.exists(marker) and _t.time() - os.path.getmtime(marker) < 86400:
            ctx.log("This worker's NVIDIA GPU cannot be used by its runs (%s): on the CPU." % open(marker).read().strip())
            return
    except Exception:          # noqa: BLE001
        pass
    try:
        ver = subprocess.run([_sys.executable, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, timeout=300).stdout.strip()
    except Exception:          # noqa: BLE001
        ver = ""
    ver = ver.split("+")[0]
    ctx.log("This worker's PyTorch (%s) is built for the CPU only: installing its CUDA build once for this block, in %s "
            "(about 2.5 GB; the worker's own Python is left as it is)." % (ver or "?", target))
    tries = [("torch==%s" % ver, cu) for cu in ("cu126", "cu124", "cu128")] if ver else []
    tries.append(("torch", "cu126"))
    for spec, cu in tries:
        cmd = [_sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "--target", target, spec,
               "--index-url", "https://download.pytorch.org/whl/%s" % cu]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and works(target):
            ctx.log("CUDA PyTorch ready (%s %s)." % (spec, cu))
            _sys.path.insert(0, target)
            return
        ctx.log("PyTorch %s for %s did not work here: %s" % (spec, cu, why[0] if r.returncode == 0 else (r.stderr or r.stdout or "")[-300:].strip()))
        if r.returncode == 0 and "No CUDA GPUs" in why[0]:
            break       # (installed, but the GPU is not visible to this worker's runs: no other build will see it)
    shutil.rmtree(target, ignore_errors=True)
    try:
        with open(marker, "w") as f:
            f.write(why[0][:300])
    except Exception:          # noqa: BLE001
        pass
    ctx.log("The GPU cannot be used on this worker (%s): this run uses the CPU. (Windows: the worker's service account needs access "
            "to the GPU; nvidia-smi run by that account shows it.)" % why[0])


def run(ctx):
    _gpu_torch(ctx)
    _load()
    return _sys.modules["train"].run(ctx)

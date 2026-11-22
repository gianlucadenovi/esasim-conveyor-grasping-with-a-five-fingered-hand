# ── EsaSim block ──────────────────────────────────────────────────────
# Conveyor grasping (xArm + 5-finger hand) (ws-conveyor-grasping-xarm-5-finger-hand) · version 63 · worker block, Python: def run(ctx)
# A five-fingered hand on an xArm (in the physics, with collisions) learns to see objects dropped on a moving conveyor, reach, grasp with real contact forces and place them in a bin
# Workspace: Gianluca De Novi's workspace · Author: Gianluca De Novi
# Saved: 2026-11-22
#
# Inputs   ctx.inputs["id"] is a list of files: .path .name .size .read_text() .read_json()
#   policy      Files (pt, py)            Policies · several · optional · Camera networks (grasp_see_*.pt) and policies (grasp_reach/grasp/place_*.pt) to carry on from, test or combine; those left on the machine by earlier runs are found by themselves (not in a round, which
#   conveyor    Files (glb)               Conveyor model · optional · Make the 3D models: a conveyor 3D file (.glb), fitted to the simulated belt and drawn in place of the boxes
# Outputs  ctx.output("id", "name.ext") is the path to write; ctx.write_json / ctx.write_text
#   policy      Files (pt)                Network · grasp_see_<date>.pt (Learn to see) or grasp_<stage>_<date>_<success>.pt with the camera network it used; Combine a round: the combined policy
#   curve       Table (csv)               Learning curve · see.csv, reach.csv, grasp.csv, place.csv, or test.csv (one row per trial)
#   models      Files (gz)                3D models · Make the 3D models: xarm_hand.robot.json.gz and belt_objects.robot.json.gz (with the conveyor's 3D file when one is wired in), for the project's assets
#   telemetry   Live telemetry            ctx.telemetry("channel", data) while it runs
# Settings ctx.params["key"]
#   mode        select    What to do = "Learn to reach" of Learn to see | Learn to reach | Learn to grasp | Learn to place | …
#   accelerator select    Run on = "auto" of auto | NVIDIA GPU | CPU
#   round_task  number    Task in a round = 0
#   worlds      number    Cells learning at once = 1024
#   minutes     number    Minutes = 60
#   training    select    Training = "Continue" of Continue | Start over
#   live_cell   select    Live cell while learning = "No" of No | Yes
#   replay_every number    Replay every (s) = 20
#   pace        number    Test pace (1 = real time) = 1
#   drop_cm     number    Test: drop height (cm) = 25
#   autoplay    select    Test: pick objects when idle = "Yes" of Yes | No
#   idle        number    Test: idle before it picks (s) = 8
#   objects     select    Test objects = "New objects only" of New objects only | All objects
#   belt_drawing select    Belt drawn = "Flat colours (3D file)" of Its own materials (3D file) | Flat colours (3D file) | Box…
#   belt_texture text      Belt texture = ""
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
from the one before; in parallel rounds over several workers, GPU and CPU, their policies averaged), Test with new
objects, Make the 3D models. On the NVIDIA GPU (MuJoCo Warp, thousands of cells at once, with its batch renderer for
the cameras) or on the CPU (MuJoCo). The xArm 6 and the hand are simulated bodies: every contact is a force."""
import sys as _sys
import types as _types

_SOURCES = {}
_SOURCES['scene'] = r'''# ── the cell: belt, bin, camera, the xArm with the hand on its flange, and the objects ───────────────────────────────
# World frame: the xArm's base at the origin, z up, metres. The belt runs along x in front of the robot (objects come
# from +x and travel towards -x); the bin is next to the belt, downstream; the camera is on a pole beside the robot.
import math

import numpy as np

# the conveyor is the project's 3D file (models/simple_rubber_conveyor.glb) at its own proportions: in the file it is
# 2.00 m long, 0.94 m wide overall, its rubber belt 0.65 m wide with its top 0.714 m above the floor; it is scaled
# alike in every direction (CONV_S) so that the belt's top is level with the top of the xArm's 75 cm pedestal
CONV_S = 0.75 / 0.714
BELT_TOP = 0.0                                   # the belt's top surface (the world's z = 0: the pedestal's top)
BELT_CY = 0.63                                   # the conveyor's centre line (y)
BELT_W = 0.650 * CONV_S                          # the rubber belt's width (0.68 m)
BELT_Y = 0.42                                    # the lane the objects are put in (y), on the robot's side of the belt
RAIL_NEAR = (-0.486 * CONV_S, -0.329 * CONV_S)   # the side frames beside the belt (y, from the centre line), as in the file:
RAIL_FAR = (0.329 * CONV_S, 0.457 * CONV_S)      # their tops 3.7 cm above the belt
RAIL_TOP = 0.035 * CONV_S
BELT_X0 = 0.2                                    # belt centre (x): the conveyor runs from BELT_START_X to BELT_END_X
BELT_HALF_LEN = 1.0 * CONV_S
BELT_START_X, BELT_END_X = BELT_X0 + BELT_HALF_LEN, BELT_X0 - BELT_HALF_LEN
SPAWN_X = (0.28, 0.93)                           # (training) where objects are put on the belt, upstream of the robot
TEST_SPAWN_X = BELT_START_X - 0.10               # (the test) dropped near the start of the belt: it travels the whole way
EXIT_X = BELT_END_X - 0.10                       # past here an object has gone off the end of the belt (and falls)
BELT_V = (0.05, 0.10)                            # the belt's speed (m/s) in training
BASE_POS = np.array([0.0, -0.10, 0.0])           # the xArm's base on its pedestal: 13 cm back from the conveyor's frame, so
                                                 # that the object's lane (52 cm out) is in front of the arm, not under it
BIN_C = np.array([-0.42, -0.02, 0.0]) + BASE_POS # bin floor centre: beside the robot, downstream, clear of the conveyor
BIN_HALF = (0.12, 0.09)                          # inner half sizes
BIN_WALL, BIN_H = 0.010, 0.10
CAM_POS = np.array([0.30, 0.02, 0.80])           # the camera on its pole
CAM_LOOK = np.array([0.24, 0.42, 0.0])
CAM_FOVY = 58.0
CAM_W, CAM_H = 96, 72
BASE_YAW = math.pi / 2                           # the xArm's base turned a quarter: its front (+x) towards the belt, its
                                                 # cable connector (at its back) away from it
HOME = np.array([0.12, 0.22, 0.30]) + BASE_POS   # the hand's rest pose (palm down, fingers along +x)
# the UFACTORY xArm 6 (its kinematics, mm and rad: each joint's place and turn in the frame of the one before; every
# joint turns about its own z), standing on its pedestal at the origin; the hand is screwed onto its flange
XARM6 = [((0.0, 0.0, 267.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)), ((53.5, -284.5, 0.0), (0.0, 0.0, 0.0)),
         ((77.5, 342.5, 0.0), (-1.5708, 0.0, 0.0)), ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)), ((76.0, 97.0, 0.0), (-1.5708, 0.0, 0.0))]
XARM6_LIM = [(-178.2, 178.2), (-118.0, 120.0), (-178.2, 11.0), (-178.2, 178.2), (-97.0, 178.2), (-178.2, 178.2)]   # degrees
ARM_MASS = (2.38, 2.30, 1.98, 1.32, 1.19, 0.17)                  # kg (UFACTORY's figures, rounded)
ARM_TORQUE = (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)                # N m: what each joint's motor can give
ARM_R = (0.062, 0.056, 0.052, 0.046, 0.044, 0.038)               # the links' collision capsules (m)
R_FH = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])   # the hand in the flange's frame
P_FH = np.array([-0.002, 0.0, 0.118])
GRASP_PT = (0.072, 0.022, -0.060)                # between the thumb and the fingertips, in the hand's frame
THUMB_OPEN, THUMB_CLOSED = 0.0, 0.71            # the thumb's opposition (rad): beside the palm, and turned under it facing the fingers

# the hand (metres, in its own frame: palm centre at the origin, fingers along +x, palm facing -z, thumb on +y), sized and
# jointed as an adult's: fingers flex at the knuckle (MCP) 0-90 deg (10 deg back), the middle joint (PIP) 0-100 deg and
# the end joint (DIP) 0-70 deg, together as tendons move them; the thumb turns under the palm at its base (CMC, 0-40 deg
# of opposition), and flexes at its knuckle (MCP, 0-55 deg) and its end joint (IP, 0-80 deg)
PALM_HALF = (0.046, 0.042, 0.013)
FINGERS = [  # name, base y, phalanx lengths, radius, splay (rad: the fingers fan out a little)
    ("index", 0.028, (0.043, 0.025, 0.020), 0.0085, 0.06),
    ("middle", 0.0095, (0.047, 0.028, 0.021), 0.0088, 0.0),
    ("ring", -0.009, (0.044, 0.027, 0.020), 0.0085, -0.05),
    ("little", -0.027, (0.035, 0.020, 0.018), 0.0078, -0.12),
]
THUMB = {"base": (-0.024, 0.036, -0.010), "len": (0.048, 0.033, 0.028), "r": 0.0105}
THUMB_U = np.array([0.62, 0.72, -0.30]); THUMB_U = THUMB_U / np.linalg.norm(THUMB_U)      # the thumb's direction, open (beside the palm)
THUMB_OPP = np.array([-0.403, -0.034, -0.915]); THUMB_OPP = THUMB_OPP / np.linalg.norm(THUMB_OPP)   # the axis it turns about to oppose
THUMB_FLEX = np.array([0.0, 0.0, 1.0])            # (set below) its knuckle's and end joint's axis: curls the tip across the palm
# what the policy moves: where the hand should be (the arm's six servos take it there) and 6 finger motors: each finger's curl (its three
# joints together, as tendons do) and the thumb's opposition; the thumb's curl is the 5th
CURL_MAX = (1.57, 1.75, 1.22)                    # MCP, PIP, DIP at a full fist (rad)
THUMB_MAX = (0.95, 1.40)                         # the thumb's MCP and IP at full flexion (rad)
TOUCH_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip", "palm_1", "palm_2", "palm_3", "palm_4"]
# and along each digit, its middle and its first phalanx (the thumb: its proximal phalanx and its metacarpal): a hand wrapped
# round an object holds it with the whole of its fingers, not only their tips (these came after the first nine, so the
# networks that learned with nine sensors carry on: the new ones are added at the end of what they are given)
DIGITS = ["thumb", "index", "middle", "ring", "little"]
TOUCH_NAMES += [d + seg for d in DIGITS for seg in ("_mid", "_prox")]
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
def rpy_quat(r, p, y):
    """w x y z of R = Rz(y) Ry(p) Rx(r)"""
    return quat_from_euler(r, p, y)


def mat_quat(R):
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


# the xArm's own link meshes for the camera, {name: (vertices (m), faces)}: set by the block (viz.camera_arm_meshes) from
# UFACTORY's files before the cell is built; without them the camera sees the contact capsules instead
ARM_MESHES = None


def arm_meshes_xml():
    """the xArm's link meshes, inline"""
    out = []
    for name, (v, f) in (ARM_MESHES or {}).items():
        out.append('<mesh name="xarm_%s" inertia="shell" vertex="%s" face="%s"/>' % (name, " ".join("%.4f" % x for x in v.reshape(-1)), " ".join(str(int(x)) for x in f.reshape(-1))))
    return "\n".join(out)


def arm_open_xml():
    """the xArm 6: six links on its six joints (position servos, stiff, with the torques its motors have, gravity
    compensated as its controller does), each with collision capsules from its joint to the next; then the hand on
    the flange. The arm and the hand collide with the belt, its rails, the bin, the pedestal and the object."""
    # (its contacts: capsules, in geom group 3, which the camera does not see; the camera sees the arm's own meshes)
    meshes = bool(ARM_MESHES)
    ag = 'contype="2" conaffinity="1" condim="3" friction="0.6 0.01 0.001" group="%d"' % (3 if meshes else 0)
    vis = 'contype="0" conaffinity="0" group="1" density="0" rgba="0.93 0.94 0.95 1"'
    s = []
    for i, ((xyz, rpy), lim) in enumerate(zip(XARM6, XARM6_LIM)):
        p = np.array(xyz) / 1000.0
        s.append('<body name="link%d" pos="%s" quat="%s" gravcomp="1">' % (i + 1, _f(p), _f(rpy_quat(*rpy))))
        s.append('<joint name="j%d" type="hinge" axis="0 0 1" range="%g %g" armature="0.25" damping="2"/>' % (i + 1, math.radians(lim[0]), math.radians(lim[1])))
        s.append('<inertial pos="0 0 0" mass="%g" diaginertia="0.008 0.008 0.006"/>' % ARM_MASS[i])
        if meshes:
            s.append('<geom name="arm%d_vis" type="mesh" mesh="xarm_link%d" %s/>' % (i + 1, i + 1, vis))
        r = ARM_R[i]
        s.append('<geom name="arm%d_j" type="sphere" size="%g" rgba="0.94 0.95 0.96 1" %s/>' % (i + 1, r * 1.05, ag))
        if i + 1 < len(XARM6):
            nx = np.array(XARM6[i + 1][0]) / 1000.0
            pts = [np.zeros(3), np.array([nx[0], 0, 0]), nx] if nx[0] and nx[1] else [np.zeros(3), nx]
            for k, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                if np.linalg.norm(b - a) > 1e-4:
                    s.append('<geom name="arm%d_%d" type="capsule" fromto="%s %s" size="%g" rgba="0.94 0.95 0.96 1" %s/>' % (i + 1, k, _f(a), _f(b), r, ag))
        if i == 0:      # the shoulder housing, down to the base
            s.append('<geom name="arm1_col" type="capsule" fromto="0 0 -0.07 0 0 -0.02" size="%g" rgba="0.94 0.95 0.96 1" %s/>' % (r, ag))
    s.append('<body name="hand" pos="%s" quat="%s">' % (_f(P_FH), _f(mat_quat(R_FH))))
    return "\n".join(s)


def arm_close_xml():
    return "</body>" + "</body>" * len(XARM6)


def thumb_flex_axis():
    """the thumb's flexion axis (MCP and IP): across the thumb, so that flexing moves its tip towards the palm's middle
    and its palm side (-z, -y) as a thumb does"""
    want = np.array([0.0, -0.5, -0.75])                 # where flexion takes the tip, with the thumb beside the palm
    d = want - (want @ THUMB_U) * THUMB_U
    d = d / np.linalg.norm(d)
    a = np.cross(THUMB_U, d)                            # (a rotation about a moves THUMB_U towards d)
    return a / np.linalg.norm(a)


def hand_xml():
    """the xArm, and on its flange the five-fingered hand: its fingers are real joints with real contacts"""
    px, py, pz = PALM_HALF
    hg = 'contype="2" conaffinity="1" condim="%d" friction="1.2 0.02 0.001" solref="%s" solimp="%s"' % (HAND_CONDIM, HAND_SOLREF, HAND_SOLIMP)
    s = [arm_open_xml()]
    s.append('<geom name="palm" type="box" size="%s" mass="0.35" rgba="0.82 0.84 0.88 1" %s/>' % (_f(PALM_HALF), hg))
    s.append('<geom name="wrist" type="cylinder" size="0.030 0.022" pos="%s" euler="0 1.5708 0" mass="0.05" rgba="0.25 0.27 0.30 1" contype="0" conaffinity="0"/>' % _f((-px - 0.020, 0, 0.002)))
    s.append('<site name="grasp" pos="%s" size="0.006" rgba="1 0 0 0"/>' % _f(GRASP_PT))
    for k, (cx, cy) in enumerate([(0.022, 0.020), (0.022, -0.020), (-0.022, 0.020), (-0.022, -0.020)]):
        s.append('<site name="palm_%d" type="box" size="0.023 0.021 0.007" pos="%s" rgba="0 0 0 0"/>' % (k + 1, _f((cx, cy, -pz))))
    # the four fingers: proximal, middle and distal phalanges, flexing about the hand's y axis (towards -z)
    for name, y0, lens, r, splay in FINGERS:
        L1, L2, L3 = lens
        s.append('<body name="%s_1" pos="%s" euler="0 0 %g">' % (name, _f((px, y0, -0.002)), splay))
        s.append('<joint name="%s_j1" type="hinge" axis="0 1 0" range="-0.17 %g" damping="0.02" armature="0.002"/>' % (name, CURL_MAX[0]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.012" rgba="0.70 0.72 0.76 1" %s/>' % (L1, r, hg))
        s.append('<site name="%s_prox" type="capsule" fromto="%g 0 0 %g 0 0" size="%g" rgba="0 0 0 0"/>' % (name, L1 * 0.15, L1 * 0.85, r * 1.45))
        s.append('<body name="%s_2" pos="%g 0 0"><joint name="%s_j2" type="hinge" axis="0 1 0" range="0 %g" damping="0.015" armature="0.002"/>' % (name, L1, name, CURL_MAX[1]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.008" rgba="0.74 0.76 0.80 1" %s/>' % (L2, r * 0.95, hg))
        s.append('<site name="%s_mid" type="capsule" fromto="%g 0 0 %g 0 0" size="%g" rgba="0 0 0 0"/>' % (name, L2 * 0.15, L2 * 0.85, r * 1.4))
        s.append('<body name="%s_3" pos="%g 0 0"><joint name="%s_j3" type="hinge" axis="0 1 0" range="0 %g" damping="0.01" armature="0.002"/>' % (name, L2, name, CURL_MAX[2]))
        s.append('<geom type="capsule" fromto="0 0 0 %g 0 0" size="%g" mass="0.006" rgba="0.30 0.32 0.36 1" %s/>' % (L3, r * 0.9, hg))
        s.append('<site name="%s_tip" pos="%g 0 -0.002" size="%g" rgba="0 0 0 0"/>' % (name, L3 * 0.75, r * 1.45))
        s.append('</body></body></body>')
    # the thumb: its metacarpal from the base of the palm on the thumb side (CMC), turning under the palm to oppose the
    # fingers; then its proximal and distal phalanges, flexing about one axis (MCP and IP) to curl the tip across the palm
    tb, (T1, T2, T3), tr = THUMB["base"], THUMB["len"], THUMB["r"]
    u, o, a = THUMB_U, THUMB_OPP, thumb_flex_axis()
    s.append('<body name="thumb_0" pos="%s">' % _f(tb))
    s.append('<joint name="thumb_rot" type="hinge" axis="%s" range="-0.1 0.8" damping="0.02" armature="0.002"/>' % _f(o))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.016" rgba="0.70 0.72 0.76 1" %s/>' % (_f(u * T1), tr * 1.15, hg))
    s.append('<site name="thumb_prox" type="capsule" fromto="%s %s" size="%g" rgba="0 0 0 0"/>' % (_f(u * T1 * 0.2), _f(u * T1 * 0.85), tr * 1.6))
    s.append('<body name="thumb_1" pos="%s"><joint name="thumb_j1" type="hinge" axis="%s" range="-0.1 %g" damping="0.015" armature="0.002"/>' % (_f(u * T1), _f(a), THUMB_MAX[0]))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.009" rgba="0.74 0.76 0.80 1" %s/>' % (_f(u * T2), tr * 0.95, hg))
    s.append('<site name="thumb_mid" type="capsule" fromto="%s %s" size="%g" rgba="0 0 0 0"/>' % (_f(u * T2 * 0.15), _f(u * T2 * 0.85), tr * 1.4))
    s.append('<body name="thumb_2" pos="%s"><joint name="thumb_j2" type="hinge" axis="%s" range="-0.15 %g" damping="0.01" armature="0.002"/>' % (_f(u * T2), _f(a), THUMB_MAX[1]))
    s.append('<geom type="capsule" fromto="0 0 0 %s" size="%g" mass="0.007" rgba="0.30 0.32 0.36 1" %s/>' % (_f(u * T3), tr * 0.9, hg))
    s.append('<site name="thumb_tip" pos="%s" size="%g" rgba="0 0 0 0"/>' % (_f(u * T3 * 0.75), tr * 1.45))
    s.append('</body></body></body>')
    s.append(arm_close_xml())
    return "\n".join(s)


HAND_JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6"]           # (the arm's joints: they place the hand)
FINGER_JOINTS = ["thumb_rot", "thumb_j1", "thumb_j2"] + ["%s_j%d" % (f[0], k) for f in FINGERS for k in (1, 2, 3)]


# the fingers' strength and the feel of their contacts (see the README: what a human hand does)
# a human finger presses with some 10 N at its tip at most in a firm grip (0.35 N m about a knuckle 3-5 cm back); the
# earlier 0.9 N m let the fingertips squeeze 20-60 N and squirt light objects out of the hand
FINGER_KP, FINGER_KV, FINGER_FORCE = 0.8, 0.06, 0.35       # each finger joint: stiffness (N m/rad), damping (a finger closing fast on an object squeezes it out), most torque (N m)
HAND_CONDIM, HAND_SOLREF, HAND_SOLIMP = 4, "0.004 1", "0.95 0.99 0.001 0.5 2"
OBJ_SOLREF = "0.004 1"
ARM_KP, ARM_KV = (1500, 500), (60, 15)                    # the shoulder and elbow joints, the wrist joints
IMPRATIO = 5
TIMESTEP = 0.002                                         # (s) the physics step: CTRL_DT / TIMESTEP of them per control step


def actuators_xml():
    a = []
    for i, j in enumerate(HAND_JOINTS):
        lo, hi = XARM6_LIM[i]
        a.append('<position name="a_%s" joint="%s" kp="%g" kv="%g" forcerange="-%g %g" ctrlrange="%g %g"/>'
                 % (j, j, ARM_KP[0] if i < 3 else ARM_KP[1], ARM_KV[0] if i < 3 else ARM_KV[1], ARM_TORQUE[i], ARM_TORQUE[i], math.radians(lo), math.radians(hi)))
    for j in FINGER_JOINTS:
        a.append('<position name="a_%s" joint="%s" kp="%g" kv="%g" forcerange="-%g %g" ctrlrange="%s"/>' % (j, j, FINGER_KP, FINGER_KV, FINGER_FORCE, FINGER_FORCE, "-0.1 0.8" if j == "thumb_rot" else "-0.2 1.8"))
    a.append('<velocity name="a_belt" joint="belt" kv="400" ctrlrange="-1 1"/>')
    return "\n".join(a)


def object_xml():
    """one object body with geom slots of every kind; each world sizes and places them for its own object"""
    og = 'contype="1" conaffinity="3" condim="6" friction="0.9 0.01 0.002" solref="%s"' % OBJ_SOLREF
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
         '<option timestep="%g" integrator="implicitfast" cone="elliptic" impratio="%g"/>' % (TIMESTEP, IMPRATIO),
         '<visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>',
         '<default><geom solimp="0.95 0.99 0.001 0.5 2"/></default>',
         '<asset><texture name="belt" type="2d" builtin="checker" rgb1="0.12 0.12 0.13" rgb2="0.16 0.16 0.17" width="64" height="64"/>',
         '<material name="belt" texture="belt" texrepeat="40 2"/>', arm_meshes_xml(), '</asset>',
         '<worldbody>',
         '<light pos="0.2 0.3 1.6" dir="0 0 -1" diffuse="0.9 0.9 0.9" castshadow="false"/>',
         '<light pos="-0.6 -0.4 1.2" dir="0.4 0.4 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>',
         '<geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.75" rgba="0.55 0.56 0.58 1" contype="1" conaffinity="1"/>',
         '<geom name="pedestal" type="box" size="0.09 0.09 0.375" pos="%s" rgba="0.3 0.3 0.33 1" contype="1" conaffinity="2"/>' % _f((BASE_POS[0], BASE_POS[1], -0.375)),
         # the conveyor's frame under the belt (its legs stand on the floor below it)
         '<geom name="belt_frame" type="box" size="%s" pos="%s" rgba="0.35 0.37 0.40 1" contype="1" conaffinity="2"/>' % (_f((BELT_HALF_LEN, (RAIL_FAR[1] - RAIL_NEAR[0]) / 2, 0.07)), _f((BELT_X0, BELT_CY + (RAIL_FAR[1] + RAIL_NEAR[0]) / 2, -0.024 - 0.07))),
         # the belt: a slab as long as the conveyor, moving at the belt's speed (the task puts it back in its place every
         # step, keeping its speed): what lies on it is carried by friction, and what goes past its end falls to the floor
         '<body name="belt" pos="%s"><joint name="belt" type="slide" axis="1 0 0" damping="5"/>' % _f((BELT_X0, BELT_CY, 0)),
         '<geom name="belt" type="box" size="%s" pos="0 0 -0.012" mass="40" material="belt" contype="1" conaffinity="2" condim="3" friction="1.0 0.01 0.001"/>' % _f((BELT_HALF_LEN, BELT_W / 2, 0.012)),
         '</body>',
         # side rails, so round things stay on the belt
         '<geom name="rail_l" type="box" size="%s" pos="%s" rgba="0.55 0.58 0.62 1" contype="1" conaffinity="2" friction="0.3"/>' % (_f((BELT_HALF_LEN, (RAIL_FAR[1] - RAIL_FAR[0]) / 2, (RAIL_TOP + 0.03) / 2)), _f((BELT_X0, BELT_CY + (RAIL_FAR[0] + RAIL_FAR[1]) / 2, (RAIL_TOP - 0.03) / 2))),
         '<geom name="rail_r" type="box" size="%s" pos="%s" rgba="0.55 0.58 0.62 1" contype="1" conaffinity="2" friction="0.3"/>' % (_f((BELT_HALF_LEN, (RAIL_NEAR[1] - RAIL_NEAR[0]) / 2, (RAIL_TOP + 0.03) / 2)), _f((BELT_X0, BELT_CY + (RAIL_NEAR[0] + RAIL_NEAR[1]) / 2, (RAIL_TOP - 0.03) / 2))),
         # the bin
         '<body name="bin" pos="%s">' % _f(BIN_C),
         '<geom name="bin_floor" type="box" size="%s" pos="0 0 -0.006" rgba="0.20 0.42 0.70 1" contype="1" conaffinity="2"/>' % _f((bx + BIN_WALL, by + BIN_WALL, 0.006)),
         '<geom name="bin_w1" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((BIN_WALL / 2, by + BIN_WALL, BIN_H / 2)), _f((bx + BIN_WALL / 2, 0, BIN_H / 2))),
         '<geom name="bin_w2" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((BIN_WALL / 2, by + BIN_WALL, BIN_H / 2)), _f((-bx - BIN_WALL / 2, 0, BIN_H / 2))),
         '<geom name="bin_w3" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((bx, BIN_WALL / 2, BIN_H / 2)), _f((0, by + BIN_WALL / 2, BIN_H / 2))),
         '<geom name="bin_w4" type="box" size="%s" pos="%s" rgba="0.22 0.46 0.76 1" contype="1" conaffinity="2"/>' % (_f((bx, BIN_WALL / 2, BIN_H / 2)), _f((0, -by - BIN_WALL / 2, BIN_H / 2))),
         '<geom name="bin_stand" type="box" size="%s" pos="0 0 -0.38" rgba="0.3 0.3 0.33 1" contype="1" conaffinity="2"/>' % _f((bx, by, 0.37)),
         '</body>',
         # the camera on its pole beside the robot
         '<geom name="cam_pole" type="cylinder" size="0.012 %g" pos="%s" rgba="0.25 0.25 0.28 1" contype="0" conaffinity="0"/>' % ((CAM_POS[2] + 0.75) / 2, _f((CAM_POS[0], CAM_POS[1] - 0.06, (CAM_POS[2] - 0.75) / 2))),
         '<camera name="cam" pos="%s" xyaxes="%s" fovy="%g"/>' % (_f(CAM_POS), _f(cam_xy), CAM_FOVY),
         '<camera name="overview" pos="%s" xyaxes="%s" fovy="45"/>' % (_f((0.95, -0.75, 0.75)), _f(look_xyaxes((0.95, -0.75, 0.75), (0.05, 0.25, 0.02)))),
         '<camera name="closeup" pos="%s" xyaxes="%s" fovy="40"/>' % (_f((0.45, -0.05, 0.42)), _f(look_xyaxes((0.45, -0.05, 0.42), (0.12, 0.30, 0.12)))),
         '<body name="arm_mount" pos="%s" euler="0 0 %g">' % (_f(BASE_POS), BASE_YAW),
         '<geom name="arm_base" type="cylinder" size="0.066 0.06" pos="0 0 0.06" rgba="0.17 0.18 0.2 1" contype="1" conaffinity="2" group="%d"/>' % (3 if ARM_MESHES else 0),
         '<geom name="arm_base_vis" type="mesh" mesh="xarm_link_base" contype="0" conaffinity="0" group="1" density="0" rgba="0.93 0.94 0.95 1"/>' if ARM_MESHES else '',
         hand_xml(), '</body>',
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
N_SUB = 20                 # physics steps (2 ms) per control step (40 ms): CTRL_DT / scene.TIMESTEP
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
            self.d = mjw.make_data(mjm, nworld=n, nconmax=128, njmax=900)
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

    def hold_belt(self, qa, da, v):
        """the belt slab back at its place, moving at speed v (per world)"""
        self.qpos[:, qa] = 0
        self.qvel[:, da] = v.to(self.qvel.dtype)

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
        nt = max(1, min(n, threads or cpu_threads()))
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

    def hold_belt(self, qa, da, v):
        self._hold = (int(qa), int(da), v.double().cpu().numpy().copy())

    def _work(self, part):
        mj = self.mj
        hold = getattr(self, "_hold", None)
        for i in part:
            m, d = self.models[self.obj[i]], self.datas[i]
            if hold is not None:
                d.qpos[hold[0]] = 0.0
                d.qvel[hold[1]] = hold[2][i]
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
            mj.mj_multiRay(m, d, pnt, vec, CAM_GROUPS, 1, -1, gid, dist, None, self._nr, 10.0)
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


CAM_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)     # (the geom groups the camera sees, as the GPU's renderer: 0-2)


def camera_dirs():
    """each pixel's ray in the camera's frame (x right, y up, looking along -z), as MuJoCo's cameras"""
    fy = 0.5 * S.CAM_H / math.tan(math.radians(S.CAM_FOVY) / 2)
    xs = (np.arange(S.CAM_W) + 0.5 - S.CAM_W / 2) / fy
    ys = -(np.arange(S.CAM_H) + 0.5 - S.CAM_H / 2) / fy
    X, Y = np.meshgrid(xs, ys)
    d = np.stack([X, Y, -np.ones_like(X)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.reshape(-1, 3)


def cpu_threads():
    """the threads the simulation may use on the CPU: all but two, so that the machine (and its EsaSim worker, which
    must keep reporting) is never starved"""
    c = os.cpu_count() or 2
    return max(1, c - 2 if c > 4 else c - 1)


_NEIGHBOUR = {"done": False}


def be_a_good_neighbour(log=None):
    """(learning on the CPU) the run's process below normal priority, and PyTorch's own threads few: the simulation's
    threads take the machine's cores, and PyTorch's would only fight them for the same cores"""
    if _NEIGHBOUR["done"]:
        return
    _NEIGHBOUR["done"] = True
    import torch
    torch.set_num_threads(max(1, min(4, cpu_threads() // 8)))
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)      # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:      # noqa: BLE001
        pass
    if log:
        log("On the CPU: %d threads for the simulation, %d for the networks, below normal priority (the machine stays responsive)." % (cpu_threads(), torch.get_num_threads()))


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
    # (on the CPU each thread steps its share of the cells: about 32 cells a thread keep every thread busy and spread
    # Python's own time per step over enough of them; more only make each update slower)
    if n > 1:              # (not the one live cell a GPU run plays beside its learning)
        be_a_good_neighbour(log)
    n = min(n, max(64, min(768, 32 * cpu_threads())))
    return CpuSim(mjm, lib, n, log=log, cam=cam), "CPU (MuJoCo, %d worlds)" % n
'''
_SOURCES['grasp_env'] = r'''# ── the task: see the object coming on the belt, reach it, grasp it well, put it in the bin ─────────────────────────
# What the policy senses (as the robot itself could): its hand's pose and speed, its fingers' angles and commands, the
# 19 touch sensors (each digit's tip, middle and first phalanx, four zones of the palm), the conveyor's speed (it is known),
# where the bin is, and what the camera network says of the object (where it is, how big, which way it lies), carried
# along at the belt speed between camera frames. What it does: says where its hand should go (x, y, z, yaw, pitch,
# roll), which the xArm's six servos do through inverse kinematics, and moves its fingers (each finger's curl and the
# thumb's swing under the palm). The arm, the hand, the fingers and the object are all simulated bodies: every contact
# between them, the belt, its rails, the bin and the pedestal is a force MuJoCo computes.
import math

import numpy as np

import scene as S
from sims import CTRL_DT

ACT_DIM = 12
OBS_DIM = 89
PRIV_DIM = 132
CAM_EVERY = 4                      # a camera frame every 4 control steps (6 per second)
STAGES = ("reach", "grasp", "place")
EP_SECONDS = {"reach": 12.0, "grasp": 15.0, "place": 20.0}
LIFT_OK, HOLD_STEPS = 0.08, 25     # a grasp: lifted 8 cm and held for 1 s (pushed about in the last half)
WORK_LO = np.array([-0.62, -0.08 + S.BASE_POS[1], 0.005])   # where the grasp point may go (the arm's reach around the cell; its
                                                               # near side behind the bin, which stands beside the arm's base)
WORK_HI = np.array([0.62, 0.62, 0.45])
DROP = np.array([S.BIN_C[0], S.BIN_C[1], 0.20])
TRAIN_DROP = 0.15                  # in training, half the objects fall onto the belt from up to 15 cm
YAW_MAX = 3.0                      # the hand's yaw (rad) either way: nearly every way round, so the arm can choose the pose it reaches
CURL_RATE = 0.05          # finger closing per control step (fraction of full curl): full closing in 0.8 s; faster, the
                          # converging fingers pinch the object and squeeze it out at a metre a second
SETTLE = 22                        # steps (0.9 s) for an object to land and settle before it counts as resting


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


class ArmKin:
    """the xArm 6's kinematics in torch, for thousands of cells at once: where the hand is for given joint angles, and
    the joint angles that put the hand where the policy wants it (damped least squares, warm-started from the angles
    of the step before, within the joints' limits)"""

    def __init__(self, torch, dev):
        self.t = torch
        J = []
        for xyz, rpy in S.XARM6:
            T = np.eye(4)
            T[:3, :3] = _rpy(*rpy)
            T[:3, 3] = np.array(xyz) / 1000.0
            J.append(T)
        self.J = torch.as_tensor(np.stack(J), dtype=torch.float32, device=dev)          # (6, 4, 4)
        B = np.eye(4)                                                                    # (the base, turned on its pedestal)
        B[:3, :3] = _rpy(0.0, 0.0, S.BASE_YAW)
        B[:3, 3] = S.BASE_POS
        self.base = torch.as_tensor(S.BASE_POS, dtype=torch.float32, device=dev)
        self.B = torch.as_tensor(B, dtype=torch.float32, device=dev)
        FH = np.eye(4)
        FH[:3, :3] = S.R_FH
        FH[:3, 3] = S.P_FH
        self.FH = torch.as_tensor(FH, dtype=torch.float32, device=dev)
        # the range the controller keeps each joint in: inside the arm's own limits, and for the shoulder and the
        # elbow the part where the arm works over the belt (not folded down under itself)
        self.lo = torch.as_tensor(np.radians([-175.0, -100.0, -170.0, -175.0, -90.0, -175.0]), dtype=torch.float32, device=dev)
        self.hi = torch.as_tensor(np.radians([175.0, 90.0, 0.0, 175.0, 170.0, 175.0]), dtype=torch.float32, device=dev)
        # the posture it prefers when the hand's pose leaves it a choice (elbow up, wrist over the work), and how much
        self.q_pref = torch.as_tensor([0.0, -0.35, -1.25, 0.0, 1.5, 0.0], dtype=torch.float32, device=dev)
        self.w_pref = torch.as_tensor([0.0, 1.0, 1.0, 0.6, 0.6, 0.0], dtype=torch.float32, device=dev)

    def fk(self, q):
        """q (n, 6) -> the hand's frame (n, 4, 4), and each joint's axis and origin (n, 6, 3) in the world"""
        torch = self.t
        n = q.shape[0]
        T = self.B.expand(n, 4, 4)
        zs, ps = [], []
        c, s = torch.cos(q), torch.sin(q)
        for i in range(6):
            T = T @ self.J[i]
            zs.append(T[:, :3, 2])
            ps.append(T[:, :3, 3])
            Rz = torch.zeros(n, 4, 4, device=q.device)
            Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c[:, i], -s[:, i], s[:, i], c[:, i]
            Rz[:, 2, 2] = 1
            Rz[:, 3, 3] = 1
            T = T @ Rz
        return T @ self.FH, torch.stack(zs, 1), torch.stack(ps, 1)

    def residual(self, q, p_t, R_t, clearance=False):
        H, z, o = self.fk(q)
        r = (p_t - H[:, :3, 3]).norm(dim=1) + 0.1 * rot_log(self.t, R_t @ H[:, :3, :3].transpose(1, 2)).norm(dim=1)
        if clearance:      # an arm pose through its own base, the pedestal or the belt is no pose at all
            pts = self.t.cat([o[:, 2:], 0.5 * (o[:, 2:3] + o[:, 3:4]), H[:, None, :3, 3]], 1)      # elbow, forearm, wrist, hand
            near_base = (pts[..., :2] - self.base[:2]).norm(dim=-1) < 0.22
            low = (0.20 - pts[..., 2]).clamp(min=0) * near_base.float() + (0.03 - pts[..., 2]).clamp(min=0)
            r = r + 10.0 * low.sum(1)
        return r

    def solve_seeds(self, p_t, R_t, iters=60):
        """(at a reset) the best of several starting poses of the arm"""
        torch = self.t
        n = p_t.shape[0]
        best_q, best_r = None, None
        for sd in ARM_SEEDS:
            sd = list(sd)
            sd[0] = (sd[0] - S.BASE_YAW + math.pi) % (2 * math.pi) - math.pi          # (the seeds are in the world's turn)
            q = self.solve(torch.tensor(sd, device=p_t.device, dtype=torch.float32).expand(n, 6).clone(), p_t, R_t, iters=iters)
            r = self.residual(q, p_t, R_t, clearance=True)
            if best_q is None:
                best_q, best_r = q, r
            else:
                better = r < best_r
                best_q = torch.where(better[:, None], q, best_q)
                best_r = torch.where(better, r, best_r)
        return best_q

    def solve(self, q, p_t, R_t, iters=3, lam=4e-3, wr=0.4):
        """joint angles from q towards the hand at p_t (n, 3) turned R_t (n, 3, 3); what the hand's pose leaves free
        goes towards the preferred posture"""
        torch = self.t
        eye6 = torch.eye(6, device=q.device)
        for _ in range(iters):
            H, z, o = self.fk(q)
            ep = p_t - H[:, :3, 3]
            er = rot_log(torch, R_t @ H[:, :3, :3].transpose(1, 2))
            e = torch.cat([ep, wr * er], -1)                                         # (n, 6)
            Jv = torch.cross(z, H[:, None, :3, 3] - o, dim=-1)                       # (n, 6, 3)
            Jm = torch.cat([Jv, wr * z], -1).transpose(1, 2)                         # (n, 6 rows, 6 joints)
            A = Jm @ Jm.transpose(1, 2) + lam * eye6
            Jp = Jm.transpose(1, 2) @ torch.linalg.inv(A)                            # (damped) pseudo-inverse
            dq = (Jp @ e[..., None])[..., 0]
            pref = self.q_pref.clone().expand_as(q).clone()
            pref[:, 0] = torch.remainder(torch.atan2(p_t[:, 1] - self.base[1], p_t[:, 0] - self.base[0]) - S.BASE_YAW + math.pi, 2 * math.pi) - math.pi   # (towards the hand)
            dn = self.w_pref * 0.3 * (pref - q)
            dq = dq + ((eye6 - Jp @ Jm) @ dn[..., None])[..., 0]
            q = torch.max(torch.min(q + dq.clamp(-0.2, 0.2), self.hi), self.lo)
        return q


ARM_SEEDS = ([0.6, -0.4, -1.2, 0.0, 1.4, 0.0], [1.2, -0.3, -1.3, 0.0, 1.4, 0.0], [0.0, -0.5, -1.0, 0.0, 1.2, 0.0],
             [2.2, 0.0, -1.0, 0.0, 1.0, 0.0], [-1.0, -0.3, -1.2, 0.0, 1.3, 0.0], [0.6, -0.4, -1.2, 0.0, 1.4, 3.0],
             [0.6, -0.4, -1.2, 0.0, 1.4, -3.0], [1.6, 0.2, -1.6, 0.0, 1.2, 1.5])


def rot_log(torch, Re):
    """the rotation vector (axis times angle) of rotation matrices (n, 3, 3), right up to half a turn"""
    v = 0.5 * torch.stack([Re[:, 2, 1] - Re[:, 1, 2], Re[:, 0, 2] - Re[:, 2, 0], Re[:, 1, 0] - Re[:, 0, 1]], -1)
    c = ((Re[:, 0, 0] + Re[:, 1, 1] + Re[:, 2, 2] - 1) / 2).clamp(-1, 1)
    ang = torch.acos(c)
    s = v.norm(dim=1)
    w = v / s.clamp(min=1e-9)[:, None] * ang[:, None]
    # near half a turn: about the axis the rotation leaves alone (its largest diagonal column)
    k = torch.argmax(torch.diagonal(Re, dim1=1, dim2=2), dim=1)
    ax = Re[torch.arange(Re.shape[0], device=Re.device), :, k] + torch.nn.functional.one_hot(k, 3).float()
    ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9) * math.pi
    return torch.where(((s < 1e-4) & (c < 0))[:, None], ax, torch.where((s < 1e-9)[:, None], v, w))


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


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
        self.belt_travel = torch.zeros(n, device=dev)     # how far the belt has run (for its slats in the 3D view)
        self.rgba = torch.as_tensor(np.array(tmpl.geom_rgba), dtype=torch.float32, device=dev)[None].repeat(n, 1, 1)
        J = lambda nm: mjm.joint(nm).qposadr[0]      # noqa: E731
        Dv = lambda nm: mjm.joint(nm).dofadr[0]      # noqa: E731
        self.qh = torch.tensor([J(j) for j in S.HAND_JOINTS], device=dev)         # the arm's six joints
        self.dh = torch.tensor([Dv(j) for j in S.HAND_JOINTS], device=dev)
        self.kin = ArmKin(torch, dev)
        self.hbody = mjm.body("hand").id
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
        self.assisted = torch.zeros(n, dtype=torch.bool, device=dev)    # (episodes that started with the hand over the object)
        self.hv_avg = z(n, 2)                # the hand's speed over the belt, smoothed (exploration jitters it step to step)
        self.hv = z(n, 6)                    # the hand's speed (linear, angular), measured from its pose step to step
        self.g_prev, self.R_prev = z(n, 3), torch.eye(3, device=dev).repeat(n, 1, 1)
        self.q_cmd = z(n, 6)                 # the arm's joint targets
        # the arm's angles with the hand at rest over the belt (from a seed the arm likes: elbow up, wrist down)
        R0 = self._euler_R(torch.zeros(1, 3, device=dev))
        p0 = torch.as_tensor(S.HOME, dtype=torch.float32, device=dev)[None] - R0[:, :, 0] * 0 - torch.einsum("kij,j->ki", R0, torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=dev))
        qs = torch.tensor([[0.6 - S.BASE_YAW, -0.4, -1.2, 0.0, 1.4, 0.0]], device=dev)
        self.q_home = self.kin.solve(qs, p0, R0, iters=200)
        self.prev_d = z(n) - 1               # the grasp point's distance a step ago (-1: just reset)
        self.prev_dd = z(n) - 1              # (place) the object's distance from over the bin a step ago
        self.lifted_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ep_ret = z(n)
        self.max_steps = int(EP_SECONDS[stage] / CTRL_DT)
        self.k = 0
        self.n_bad = 0
        self.assist = {"reach": 0.0, "grasp": 0.5, "place": 0.3}[stage]   # a share of episodes start with the hand over the object
        if stage in ("grasp", "place") and n > 1:
            install_imitation(self)

    # ── resets ────────────────────────────────────────────────────────────────────────────────────────────────────
    def reset_all(self):
        self.reset(self.t.arange(self.n, device=self.dev))
        self.sim.refresh()

    def reset(self, idx, obj=None, mode=None, yaw=None, speed=None, x0=None, assist=None, drop=None):
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
        z0 = -zlow + 0.006
        # dropped from above the belt, turning a little: where and how it lands is up to the physics (the test drops
        # every object; in training half of them fall from up to TRAIN_DROP)
        if drop:
            dh = float(drop) * (0.6 + 0.4 * torch.rand(k, generator=g, device=dev))
        else:
            dh = torch.where(torch.rand(k, generator=g, device=dev) < 0.5, TRAIN_DROP * torch.rand(k, generator=g, device=dev), torch.zeros(k, device=dev))
        falls = dh > 0.01
        z_rest0, q_rest0 = z0.clone(), q.clone()
        z0 = z0 + dh
        tilt = random_quat(torch, k, g, dev)
        q = torch.where(falls[:, None], torch.nn.functional.normalize(q + 0.25 * tilt * torch.sign((q * tilt).sum(-1, keepdim=True) + 1e-9), dim=-1), q)
        qpos[:, self.qo:self.qo + 3] = torch.stack([x0, y0, z0], -1)
        qpos[:, self.qo + 3:self.qo + 7] = q
        # the belt's speed
        v = (S.BELT_V[0] + (S.BELT_V[1] - S.BELT_V[0]) * torch.rand(k, generator=g, device=dev)) if speed is None else torch.full((k,), float(speed), device=dev)
        self.belt_v[idx] = v
        qvel = torch.zeros(k, self.mjm.nv, device=dev)
        qvel[:, self.db] = -v
        spin = 4.0 * (torch.rand(k, 3, generator=g, device=dev) * 2 - 1)            # spinning as it falls
        slide = 0.15 * (torch.rand(k, 2, generator=g, device=dev) * 2 - 1)
        qvel[:, self.do_ + 3:self.do_ + 6] = torch.where(falls[:, None], spin, torch.zeros_like(spin))
        qvel[:, self.do_:self.do_ + 2] = torch.where(falls[:, None], slide, torch.zeros_like(slide))
        # the hand: at rest, palm down, open (a little off each time)
        hp = torch.as_tensor(S.HOME, dtype=torch.float32, device=dev)[None] + 0.04 * (torch.rand(k, 3, generator=g, device=dev) * 2 - 1)
        hy = 0.3 * (torch.rand(k, generator=g, device=dev) * 2 - 1)
        qpos[:, self.qf[0]] = S.THUMB_OPEN
        # a share of the grasp and place episodes start with the hand already over the object, turned across it
        # (the reach it learned before), so that it finds out early what closing the hand on things does
        pa = self.assist if assist is None else float(assist)
        self.assisted[idx] = False
        if pa > 0:
            cw, ew, yw2 = obb_of(torch, p + torch.stack([x0, y0, -zlow + 0.006], -1)[:, None])
            over = torch.rand(k, generator=g, device=dev) < pa
            hp_a = torch.stack([cw[:, 0] - 0.04 * v, cw[:, 1], cw[:, 2] + ew[:, 2] + 0.03 + 0.03 * torch.rand(k, generator=g, device=dev)], -1)
            hy_a = 0.5 * torch.atan2(yw2[:, 1], yw2[:, 0]) + math.pi / 2
            hy_a = torch.remainder(hy_a + math.pi / 2, math.pi) - math.pi / 2       # (a turn of pi is the same grasp)
            hp = torch.where(over[:, None], hp_a, hp)
            hy = torch.where(over, hy_a, hy)
            self.assisted[idx] = over
            # (with the hand already over it, the object rests on the belt: it is not dropped into the hand)
            qpos[:, self.qo + 2] = torch.where(over, z_rest0, qpos[:, self.qo + 2])
            qpos[:, self.qo + 3:self.qo + 7] = torch.where(over[:, None], q_rest0, qpos[:, self.qo + 3:self.qo + 7])
            qvel[:, self.do_:self.do_ + 6] = torch.where(over[:, None], torch.zeros_like(qvel[:, self.do_:self.do_ + 6]), qvel[:, self.do_:self.do_ + 6])
        e0 = torch.stack([hy, torch.zeros_like(hy), torch.zeros_like(hy)], -1)
        R0 = self._euler_R(e0)
        root = hp - torch.einsum("kij,j->ki", R0, torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=dev))
        q0 = self.kin.solve_seeds(root, R0)
        qpos[:, self.qh] = q0
        self.q_cmd[idx] = q0
        self.tgt_p[idx] = hp
        self.tgt_e[idx] = e0
        self.curl[idx] = 0.0
        self.opp[idx] = 0.0
        ctrl = torch.zeros(k, self.mjm.nu, device=dev)
        self._fill_ctrl(ctrl, self.tgt_p[idx], self.tgt_e[idx], self.curl[idx], self.opp[idx], idx=idx)
        H, _, _ = self.kin.fk(q0)
        self.R_prev[idx] = H[:, :3, :3]
        self.g_prev[idx] = H[:, :3, 3] + torch.einsum("kij,j->ki", H[:, :3, :3], torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=dev))
        self.hv[idx] = 0
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

    def restart_clock(self, idx):
        """(after a dropped object has landed) the episode starts now: its clock and counters from zero"""
        self.steps[idx] = 0
        for a in (self.ok_cnt, self.hold_cnt, self.bin_cnt, self.ep_ret):
            a[idx] = 0
        self.prev_d[idx] = -1
        self.prev_dd[idx] = -1
        self.lifted_ever[idx] = False

    def _fill_ctrl(self, ctrl, p, e, curl, opp, idx=None):
        """the arm's servos from where the grasp point should be (palm down, turned by yaw, pitch and roll): the joint
        angles that put the hand there (or as near as the arm can); the fingers' joints from their curls; the thumb's
        base from its swing"""
        torch = self.t
        R = self._euler_R(e)
        gp = torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=self.dev)
        root = p - torch.einsum("kij,j->ki", R, gp)
        if idx is None:
            q = self.kin.solve(self.q_cmd, root, R, iters=6)
            self.q_cmd = q
            ctrl[:, 0:6] = self.q_cmd
        else:
            ctrl[:, 0:6] = self.q_cmd[idx]
        cm = torch.as_tensor(S.CURL_MAX, dtype=torch.float32, device=self.dev)
        ctrl[:, 6] = S.THUMB_OPEN + opp * (S.THUMB_CLOSED - S.THUMB_OPEN)
        ctrl[:, 7] = curl[:, 0] * S.THUMB_MAX[0]
        ctrl[:, 8] = curl[:, 0] * S.THUMB_MAX[1]
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
        self.tgt_e[:, 0] = self.tgt_e[:, 0].clamp(-YAW_MAX, YAW_MAX)
        self.tgt_e[:, 1:] = self.tgt_e[:, 1:].clamp(-0.7, 0.7)
        # the fingers and the thumb as a servo hand moves them: the action is where each one is to be (0 open, 1 closed;
        # below 0 open too) and it goes there no faster than CURL_RATE a step. (As a speed added up step by step, the
        # spread of the actions while it learns cancels out and the fingers almost never close on the object at the
        # right moment: at a realistic speed it never found out what closing the hand does.)
        self.curl = self.curl + (a[:, 6:11].clamp(0, 1) - self.curl).clamp(-CURL_RATE, CURL_RATE)
        self.opp = self.opp + (a[:, 11].clamp(0, 1) - self.opp).clamp(-CURL_RATE, CURL_RATE)
        ctrl = sim.ctrl.clone().float()
        self._fill_ctrl(ctrl, self.tgt_p, self.tgt_e, self.curl, self.opp)
        ctrl[:, self.a_belt] = -self.belt_v
        sim.ctrl[:] = ctrl.to(sim.ctrl.dtype)
        # the grasp test: once it holds the object up, it is pushed about (1.2 N, a new way every 0.2 s)
        sim.xfrc[:, self.ob] = 0
        sim.hold_belt(self.qb, self.db, -self.belt_v)          # (the belt slab back in its place, at the belt's speed)
        self.belt_travel = self.belt_travel + self.belt_v * CTRL_DT
        if self.stage in ("grasp", "place"):
            push = (self.hold_cnt >= HOLD_STEPS // 2) & (self.hold_cnt < HOLD_STEPS)
            if bool(push.any()):
                if self.k % 5 == 0 or not hasattr(self, "_push_dir"):
                    d_ = torch.randn(self.n, 3, generator=self.gen, device=self.dev)
                    self._push_dir = d_ / d_.norm(dim=1, keepdim=True)
                sim.xfrc[:, self.ob, :3] = (1.2 * self._push_dir * push[:, None].float()).to(sim.xfrc.dtype)
        sim.step()
        g_, R_ = self.hand_state()
        Rd = R_ @ self.R_prev.transpose(1, 2)
        w_ = 0.5 * torch.stack([Rd[:, 2, 1] - Rd[:, 1, 2], Rd[:, 0, 2] - Rd[:, 2, 0], Rd[:, 1, 0] - Rd[:, 0, 1]], -1)
        self.hv = torch.cat([(g_ - self.g_prev) / CTRL_DT, w_ / CTRL_DT], -1)
        self.g_prev, self.R_prev = g_.clone(), R_.clone()
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
        dig = f[:, 0:5] + f[:, 9::2] + f[:, 10::2]                         # (each digit: its tip, middle and first phalanx)
        thumb, fing, palm = dig[:, 0], dig[:, 1:5], f[:, 5:9]
        tn = torch.log1p(f[:, :5]) / math.log1p(10.0)
        dn = torch.log1p(dig) / math.log1p(10.0)
        pinch = torch.min(tn[:, 0], tn[:, 1:5].max(1).values).clamp(0, 1)
        wrap = (dn[:, 1:5].clamp(0, 1).sum(1) / 4) * (dn[:, 0].clamp(0, 1) + (torch.log1p(palm) / math.log1p(10.0)).max(1).values.clamp(0, 1)).clamp(0, 1)
        held = ((thumb > 0.3) | (palm.max(1).values > 0.3)) & ((fing > 0.3).sum(1) >= 1)
        lift = op[:, 2] - self.z_rest
        lifted = (lift > LIFT_OK) & held
        rel_v = (ov[:, 0] + self.belt_v).abs() + ov[:, 1].abs()
        on_belt = (op[:, 2] < self.z_rest + 0.015)
        knock = torch.where(on_belt & ~held, (rel_v - 0.03).clamp(min=0), torch.zeros_like(rel_v))
        rate = ((a - self.prev_a) ** 2).mean(1)
        crush = (f.max(1).values - 25.0).clamp(min=0) / 25.0
        low = (0.004 - (g[:, 2] - 0.0)).clamp(min=0) * 50
        hv = self.hv[:, :2]
        self.hv_avg = 0.85 * self.hv_avg + 0.15 * hv
        match = (self.hv_avg[:, 0] + self.belt_v).abs() + self.hv_avg[:, 1].abs()
        early = self.steps < SETTLE                                        # (the object falling and settling on the belt)
        self.z_rest = torch.where(self.steps == SETTLE, op[:, 2], self.z_rest)
        r = 0.6 * r_reach + 20.0 * gain + 0.25 * r_align + 0.1 * level.clamp(min=0) - 0.05 * rate - 1.0 * knock - 0.5 * crush - low
        success = torch.zeros_like(held)
        fail = (op[:, 0] < S.EXIT_X) | (op[:, 2] < -0.08)
        in_bin = ((op[:, 0] - S.BIN_C[0]).abs() < S.BIN_HALF[0]) & ((op[:, 1] - S.BIN_C[1]).abs() < S.BIN_HALF[1]) & (op[:, 2] < S.BIN_H) & (op[:, 2] > -0.02)
        if st == "reach":
            # (within 4 cm of the grasp point, palm about level, turned roughly across a long object, keeping roughly
            # with the belt, for two steps: what a grasp can start from. The earlier test, 2.5 cm and 12 cm/s for three
            # steps under the exploration's jitter, was never met in any run, so the success bonus never paid)
            ok = (dxy < 0.04) & (dz < 0.04) & (level > 0.8) & ((elong < 0.3) | (align > 0.5)) & (match < 0.2) & ~early
            self.ok_cnt = torch.where(ok, self.ok_cnt + 1, torch.zeros_like(self.ok_cnt))
            success = self.ok_cnt >= 2
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
        hv = self.hv
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
            tn[:, :9],                                               # 9
            (T - g) * 5, ec, ee * 10, ey, torch.stack([torch.cos(dy), torch.sin(dy)], -1),    # 3 + 3 + 3 + 2 + 2
            (self.est_age.clamp(max=3) / 3)[:, None], (self.belt_v * 10)[:, None], (self.est_v * 10)[:, None],   # 3
            (drop - g) * 3,                                           # 3
            self.prev_a,                                              # 12
            tn[:, 9:],                                                # 10: the phalanges' sensors (added last, see S.TOUCH_NAMES)
        ], 1)
        op, oq, c, e, y = self.truth()
        ov = sim.qvel[:, self.do_:self.do_ + 6].float()
        st = torch.zeros(self.n, 3, device=self.dev)
        st[:, STAGES.index(self.stage)] = 1
        priv = torch.cat([obs, (c - g) * 5, oq, ov[:, :3], ov[:, 3:] * 0.1, e * 10, y, ((op[:, 2] - self.z_rest) * 10)[:, None],
                          (f[:, :9] > 0.3).float(), st, (self.steps.float() / self.max_steps)[:, None], self.lifted_ever.float()[:, None],
                          (f[:, 9:] > 0.3).float()], 1)
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
        # of the two ways round (a grasp turned half a turn is the same grasp), the one the arm reaches better
        if self.timer.sum() == 0 or int(self.timer.max()) % 10 == 0:
            alt = torch.where(want > 0, want - math.pi, want + math.pi)
            gp = torch.as_tensor(S.GRASP_PT, dtype=torch.float32, device=dev)
            res = []
            for yv in (want, alt):
                Rw = env._euler_R(torch.stack([yv, torch.zeros_like(yv), torch.zeros_like(yv)], -1))
                root = T - torch.einsum("kij,j->ki", Rw, gp)
                res.append(env.kin.residual(env.kin.solve(env.q_cmd.clone(), root, Rw, iters=12), root, Rw))
            self.flip = res[1] + 0.01 < res[0]
        if getattr(self, "flip", None) is not None:
            want = torch.where(self.flip, torch.where(want > 0, want - math.pi, want + math.pi), want)
        # carrying and over the bin: the fingers pointing away from the arm's base (a pose the arm reaches well there)
        out = torch.atan2(env.tgt_p[:, 1] - S.BASE_POS[1], env.tgt_p[:, 0] - S.BASE_POS[0])
        out = torch.where(out > YAW_MAX, out - 2 * math.pi, torch.where(out < -YAW_MAX, out + 2 * math.pi, out)).clamp(-YAW_MAX, YAW_MAX)
        want = torch.where(phase >= 4, out, want)
        a = torch.zeros(n, ACT_DIM, device=dev)
        tgt = T.clone()
        tgt[:, 0] = torch.where(phase == 0, tgt[:, 0].clamp(max=0.52), tgt[:, 0])   # (an object still out of reach: waited for)
        tgt[:, 2] = torch.where(phase == 0, T[:, 2] + 0.07, T[:, 2])
        lift = torch.stack([env.tgt_p[:, 0], env.tgt_p[:, 1], torch.full((n,), 0.20, device=dev)], -1)
        drop = torch.as_tensor(DROP, dtype=torch.float32, device=dev).expand(n, 3)
        # (closing: the hand keeps going with the object as the belt carries it)
        tgt = torch.where((phase == 3)[:, None], lift, tgt)
        # carrying: round the arm's base at a steady distance from it (an arc, as a robot programmer would), not in
        # a straight line that cuts close past the base where the arm would have to fold up
        b0 = torch.as_tensor(S.BASE_POS[:2], dtype=torch.float32, device=dev)
        rel_now, rel_d = env.tgt_p[:, :2] - b0, drop[:, :2] - b0
        r_now = rel_now.norm(dim=1).clamp(min=1e-3)
        th_now = torch.atan2(rel_now[:, 1], rel_now[:, 0])
        r_d = rel_d.norm(dim=1)
        th_d = torch.atan2(rel_d[:, 1], rel_d[:, 0])
        dth = torch.remainder(th_d - th_now + math.pi, 2 * math.pi) - math.pi
        th_n = th_now + dth.clamp(-0.03 / r_now, 0.03 / r_now)
        r_n = r_now + (r_d - r_now).clamp(-0.012, 0.012)
        arc = torch.stack([b0[0] + r_n * torch.cos(th_n), b0[1] + r_n * torch.sin(th_n), drop[:, 2]], -1)
        arc = torch.where(((env.tgt_p[:, :2] - drop[:, :2]).norm(dim=1) < 0.03)[:, None], drop, arc)
        tgt = torch.where((phase >= 4)[:, None], arc, tgt)
        sp = torch.where(phase >= 3, 0.6, 1.0)
        sp = torch.where((phase == 3) & (timer < 15), 0.25, sp)              # (the first centimetres of the lift slowly: the grip settles)
        a[:, :3] = ((tgt - env.tgt_p) / 0.018).clamp(-1, 1) * sp[:, None]   # (carrying: steadier)
        a[:, 3] = ((want - env.tgt_e[:, 0]) / 0.10).clamp(-1, 1) * torch.where(phase >= 4, 0.35, 1.0)     # (turning gently with it in hand)
        # fingers tilted down to grasp (the arm reaches low on the belt far better so), level to carry
        want_e = torch.stack([torch.where(phase <= 3, 0.4, 0.0), torch.zeros(n, device=dev)], -1)
        a[:, 4:6] = ((want_e - env.tgt_e[:, 1:]) / 0.06).clamp(-1, 1)
        close = (phase >= 2) & (phase < 5)
        a[:, 6:11] = torch.where(close[:, None], torch.ones(n, 5, device=dev), -torch.ones(n, 5, device=dev))
        a[:, 11] = torch.where(close | ((phase == 1) & (timer > 4)), torch.ones(n, device=dev), -torch.ones(n, device=dev))
        d = g - T
        timer += 1
        nxt = (phase == 0) & (d[:, :2].norm(dim=1) < 0.012) & (timer > 8)
        nxt |= (phase == 1) & (d.norm(dim=1) < 0.012) & (timer > 6)
        nxt |= (phase == 2) & (timer > 12) & (((env.curl.mean(1) > 0.95) & (timer > 24)) | (timer > 40))   # (lifted only once the hand has closed)
        nxt |= (phase == 3) & (env.tgt_p[:, 2] > 0.19) & (timer > 30)
        nxt |= (phase == 4) & ((env.tgt_p - drop).norm(dim=1) < 0.01)
        self.phase = torch.where(nxt, phase + 1, phase)
        self.timer = torch.where(nxt, torch.zeros_like(timer), timer)
        return a


def imitate(env, algo, log, steps=700, epochs=3000, rounds=3):
    """the policy learns to act as the scripted controller does. First the controller drives the cells (with a little
    noise) while (what the policy senses, what the controller does) pairs are kept, and the policy's network is fitted
    to them; then, for a few rounds (DAgger), the policy itself drives and the controller says what it would do in the
    places the policy gets to, so that it learns to get back on track from its own mistakes"""
    import time
    torch = env.t
    t0 = time.time()
    prog = getattr(env, "progress", None) or (lambda *a: None)
    actor = algo.ac.actor
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    O, A = [], []

    def collect(r_, drive_policy):
        demo = Demo(env)
        env.reset_all()
        env.perceive()
        ok = tot = 0
        for k in range(steps):
            a = demo.act()
            o, _ = env.obs()
            if k % 2 == 0:            # (kept in the computer's memory, not the GPU's)
                O.append(o.half().cpu())
                A.append(a.half().cpu())
            if drive_policy:
                with torch.no_grad():
                    u = actor(algo.on(o))
                u = u + 0.1 * torch.randn_like(u)
            else:
                u = a + 0.15 * torch.randn_like(a)
            r, done, info = env.step(u.clamp(-1, 1))
            if bool(done.any()):
                idx = done.nonzero().flatten()
                ok += int(info["success"][idx].sum())
                tot += len(idx)
                env.reset(idx)
                env.sim.refresh()
                demo.reset(idx)
            if k % 20 == 0:
                prog(0, "Learning from demonstrations: round %d of %d, step %d of %d" % (r_ + 1, rounds + 1, k, steps))
        return ok, tot

    def fit(n_epochs):
        Oc, Ac = torch.cat(O), torch.cat(A)
        N = len(Oc)
        loss = torch.zeros(())
        for ep in range(n_epochs):
            j = torch.randint(N, (4096,))
            loss = ((actor(algo.on(Oc[j].to(env.dev).float())) - Ac[j].to(env.dev).float()) ** 2).mean()
            if ep % 200 == 0:
                prog(0, "Learning from demonstrations: fitting the policy (%d of %d)" % (ep, n_epochs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return float(loss.item()), N

    ok, tot = collect(0, False)
    log("Demonstrations: %d episodes, %.0f%% succeeded with the scripted controller." % (tot, 100.0 * ok / max(1, tot)))
    Oc = torch.cat(O)
    for k in range(0, len(Oc), 65536):
        algo.on.update(Oc[k:k + 65536].to(env.dev).float())
    del Oc
    err, N = fit(epochs)
    log("The policy imitates it: error %.3f on %d examples (%.0f s)." % (err, N, time.time() - t0))
    for r_ in range(rounds):
        ok, tot = collect(r_ + 1, True)
        err, N = fit(epochs // 2 + 500)
        log("Round %d (the policy drives, the controller corrects): %.0f%% of %d episodes succeeded; error %.3f on %d examples (%.0f s)." % (
            r_ + 1, 100.0 * ok / max(1, tot), tot, err, N, time.time() - t0))
    with torch.no_grad():
        algo.ac.log_std.fill_(math.log(0.12))
    # the demonstrations stay with PPO for a while: a term that keeps the policy near them, fading out
    keep = min(len(O), 600)
    algo.bc = (torch.cat(O[-keep:]), torch.cat(A[-keep:]))
    algo.bc_coef = 1.0
    # (small, capped steps from here: the adaptive step size grows while the demonstrations hold the policy still,
    # and at 1e-2 PPO undid in a few updates what the policy had imitated)
    algo.lr, algo.lr_max = 1e-4, 3e-4
    for g in algo.opt.param_groups:
        g["lr"] = algo.lr
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
            rate = 0.0
            if getattr(env, "policy_stage", None) == env.stage:      # (only a policy of this stage can already do it)
                rate, tot = policy_rate(env, self, env.max_steps + 10)
                env.log("The policy as it is succeeds in %.0f%% of %d episodes." % (rate, tot))
            if rate < 50:
                imitate(env, self, env.log, steps=500, rounds=2)
                env.log("Now PPO, from what it imitated (its critic first learns what the states are worth).")
                state["critic"] = 10
        if state["critic"] > 0:
            state["critic"] -= 1
            critic_only = True
        return orig(self, priv_last, critic_only)
    pp.PPO.update = update
    pp.PPO._imitation = True
'''
_SOURCES['viz'] = r'''# ── what the project's 3D view draws: the xArm carrying the hand, the objects on the belt, the cell around them ─────────
# The 3D view poses robots (esasim.robot/1 files in the project's assets) from live joint angles. Two robots:
#   xarm_hand: the UFACTORY xArm 6 (its own meshes) with the five-fingered hand screwed onto its flange, so the hand
#              is always where the arm puts it; the arm's six angles and the fingers' are the simulated ones
#   belt_objects: every object of the library, each on a free mount (x, y, z, then turns about z, y, x); the one on the
#              belt is posed from the simulation, the others wait out of sight under the floor
# and the cell (belt, rails, bin, pedestal, camera on its pole) as objects of the scene channel.
import gzip
import json
import math
import os
import time

import numpy as np

import scene as S

ARM = "xarm_hand"
OBJS = "belt_objects"
PARK_MM = -3000.0                 # where the objects not in play wait (under the floor)
XARM6 = [((0.0, 0.0, 267.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)), ((53.5, -284.5, 0.0), (0.0, 0.0, 0.0)),
         ((77.5, 342.5, 0.0), (-1.5708, 0.0, 0.0)), ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)), ((76.0, 97.0, 0.0), (-1.5708, 0.0, 0.0))]
XARM6_LIM = [(-178.2, 178.2), (-118.0, 120.0), (-178.2, 11.0), (-178.2, 178.2), (-97.0, 178.2), (-178.2, 178.2)]
MESH_URL = "https://raw.githubusercontent.com/xArm-Developer/xarm_ros/master/xarm_description/meshes/xarm6/visual/%s.stl"
FLANGE_X = -0.118                 # the far face of the adapter on the back of the hand, on the hand's x axis (m)
# the hand in the flange's frame: the flange's z carries on along the hand's x (the fingers), its x is the hand's z
R_FH = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
P_FH = np.array([-0.002, 0.0, -FLANGE_X])


# ── small maths ───────────────────────────────────────────────────────────────────────────────────────────────────
def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rpy_of(R):
    """roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll)"""
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-6:
        return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), p, 0.0


def _near(a, ref):
    """angle a moved by whole turns to be as close as can be to ref"""
    return a + 2 * math.pi * round((ref - a) / (2 * math.pi))


def rpy_near(R, prev):
    """roll, pitch, yaw of R (R = Rz(yaw) Ry(pitch) Rx(roll)) chosen to follow on from prev, the angles sent before: the
    3D view eases between the angles it is sent, so a yaw that wraps from +180 to -180 degrees, or one of the two ways of
    writing the same turn swapped for the other, would make the object whirl round in the view though it lies still"""
    if prev is None:
        return rpy_of(R)
    pr, pp, py = prev
    s = max(-1.0, min(1.0, -R[2, 0]))
    p = math.asin(s)
    if math.cos(p) > 2e-3:
        r, y = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
        cands = [(r, p, y), (r + math.pi, math.pi - p, y + math.pi)]
    else:                       # (standing on end: only yaw minus roll is fixed, so the yaw is kept where it was)
        p = math.copysign(math.pi / 2, s)
        y = py
        c, sn = math.cos(y), math.sin(y)
        Rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        cp, sp = math.cos(p), math.sin(p)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        M = Ry.T @ Rz.T @ R
        cands = [(math.atan2(M[2, 1], M[2, 2]), p, y)]
    best, bd = None, 1e9
    for r, p, y in cands:
        c = (_near(r, pr), _near(p, pp), _near(y, py))
        d = abs(c[0] - pr) + abs(c[1] - pp) + abs(c[2] - py)
        if d < bd:
            best, bd = c, d
    return best


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ── meshes ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _sphere(n=10, m=16):
    v, f = [], []
    for i in range(n + 1):
        th = math.pi * i / n
        for j in range(m):
            ph = 2 * math.pi * j / m
            v.append((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)))
    for i in range(n):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = a + m, b + m
            f += [(a, c, b), (b, c, d)]
    return np.array(v), np.array(f)


def _capsule(r, h, n=8, m=18):
    v, f = [], []
    rings = []                                  # top cap, then bottom cap: a sphere split at its equator, pulled apart
    for i in range(n + 1):
        th = (math.pi / 2) * i / n
        rings.append((r * math.sin(th), h + r * math.cos(th)))
    for i in range(n, -1, -1):
        th = (math.pi / 2) * i / n
        rings.append((r * math.sin(th), -h - r * math.cos(th)))
    for rr, z in rings:
        for j in range(m):
            ph = 2 * math.pi * j / m
            v.append((rr * math.cos(ph), rr * math.sin(ph), z))
    for i in range(len(rings) - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            c, d = a + m, b + m
            f += [(a, c, b), (b, c, d)]
    return np.array(v), np.array(f)


def _cylinder(r, h, m=20):
    v = [(0.0, 0.0, h), (0.0, 0.0, -h)]
    for j in range(m):
        ph = 2 * math.pi * j / m
        v += [(r * math.cos(ph), r * math.sin(ph), h), (r * math.cos(ph), r * math.sin(ph), -h)]
    f = []
    for j in range(m):
        a, b = 2 + 2 * j, 2 + 2 * ((j + 1) % m)
        f += [(0, a, b), (1, b + 1, a + 1), (a, a + 1, b), (b, a + 1, b + 1)]
    return np.array(v), np.array(f)


def _box(sx, sy, sz):
    v = np.array([(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v, np.array(f)


def prim(t, size):
    """vertices (m) and faces of a MuJoCo primitive in its own frame"""
    if t == "box":
        return _box(*size[:3])
    if t == "sphere":
        v, f = _sphere()
        return v * size[0], f
    if t == "ellipsoid":
        v, f = _sphere()
        return v * np.array(size[:3]), f
    if t == "cylinder":
        return _cylinder(size[0], size[1])
    if t == "capsule":
        return _capsule(size[0], size[1])
    return np.zeros((0, 3)), np.zeros((0, 3), int)


class Mesh:
    def __init__(self):
        self.v, self.f, self.n = [], [], 0

    def add(self, v, f, R=None, p=None):
        v = np.asarray(v, float)
        if R is not None:
            v = v @ np.asarray(R).T
        if p is not None:
            v = v + np.asarray(p)
        self.v.append(v)
        self.f.append(np.asarray(f, int) + self.n)
        self.n += len(v)

    def link(self, color, scale_to=10000.0):
        """the link in the file's units (0.1 mm as integers, the file's scale 0.1 making them mm)"""
        if not self.v:
            return {}
        v = np.concatenate(self.v) * scale_to
        f = np.concatenate(self.f)
        return {"v": [int(round(x)) for x in v.reshape(-1)], "f": [int(x) for x in f.reshape(-1)], "color": color}


def hexcol(rgba):
    r, g, b = [max(0, min(255, int(round(255 * float(c))))) for c in rgba[:3]]
    return (r << 16) | (g << 8) | b


# ── the xArm and the hand ─────────────────────────────────────────────────────────────────────────────────────────
def read_stl(data):
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        nums = [float(x) for line in data.decode("ascii", "replace").splitlines() if line.strip().startswith("vertex") for x in line.split()[1:4]]
        return np.array(nums, np.float32).reshape(-1, 3, 3)
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].copy()


def decimate(tris, cell):
    v = tris.reshape(-1, 3).astype(np.float64)
    keys = np.floor(v / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(uniq), 3)); np.add.at(sums, inv, v)
    verts = sums / np.bincount(inv, minlength=len(uniq))[:, None]
    f = inv.reshape(-1, 3)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if len(f):
        _, idx = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
        f = f[np.sort(idx)]
    used = np.unique(f)
    remap = -np.ones(len(verts), np.int64); remap[used] = np.arange(len(used))
    return verts[used], remap[f]


def xarm_links(cache, log):
    """the xArm 6's seven link meshes (m), from UFACTORY's own (xarm_ros, BSD-3), downloaded once into cache; None
    when they cannot be had"""
    import urllib.request
    out = []
    try:
        os.makedirs(cache, exist_ok=True)
        for name in ["link_base"] + ["link%d" % (i + 1) for i in range(6)]:
            path = os.path.join(cache, name + ".stl")
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                with urllib.request.urlopen(MESH_URL % name, timeout=60) as r:
                    data = r.read()
                with open(path + ".part", "wb") as fh:
                    fh.write(data)
                os.replace(path + ".part", path)
            v, f = decimate(read_stl(open(path, "rb").read()) * 1000.0, 3.0)
            out.append((v / 1000.0, f))
        return out
    except Exception as e:      # noqa: BLE001
        log("The xArm's meshes could not be downloaded (%s): it is drawn with simple shapes." % e)
        return None


def camera_arm_meshes(cache, log):
    """the xArm's link meshes for the simulation's camera (simplified to 1 cm: its pictures are 96 x 72), or None"""
    import urllib.request
    out = {}
    try:
        os.makedirs(cache, exist_ok=True)
        for name in ["link_base"] + ["link%d" % (i + 1) for i in range(6)]:
            path = os.path.join(cache, name + ".stl")
            if not os.path.exists(path) or os.path.getsize(path) < 1000:
                with urllib.request.urlopen(MESH_URL % name, timeout=60) as r:
                    data = r.read()
                with open(path + ".part", "wb") as fh:
                    fh.write(data)
                os.replace(path + ".part", path)
            v, f = decimate(read_stl(open(path, "rb").read()) * 1000.0, 10.0)
            out[name] = (np.round(v) / 1000.0, f)
        return out
    except Exception as e:      # noqa: BLE001
        log("The xArm's meshes could not be had (%s): the camera sees the arm as its contact capsules." % e)
        return None


def simple_xarm_links():
    """the xArm 6 drawn from its kinematics: a rounded link from each joint to the next"""
    out = []
    R = [0.075, 0.06, 0.055, 0.05, 0.045, 0.04, 0.035]
    for i in range(7):
        m = Mesh()
        if i == 0:
            v, f = _cylinder(0.08, 0.1)
            m.add(v, f, p=(0, 0, 0.1))
        else:
            v, f = _cylinder(R[i] * 1.1, R[i] * 1.2)
            m.add(v, f)
            if i < 6:
                nx = np.array(XARM6[i][0]) / 1000.0
                pts = [np.zeros(3), np.array([nx[0], 0, 0]), nx] if nx[0] and nx[1] else [np.zeros(3), nx]
                for a, b in zip(pts[:-1], pts[1:]):
                    d = b - a
                    L = np.linalg.norm(d)
                    if L < 1e-4:
                        continue
                    v, f = _capsule(R[i], L / 2)
                    z = d / L
                    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
                    x = x / np.linalg.norm(x)
                    m.add(v, f, R=np.stack([x, np.cross(z, x), z], 1), p=(a + b) / 2)
        out.append((np.concatenate(m.v), np.concatenate(m.f)))
    return out


def arm_model(mjm, links):
    """xarm_hand.robot.json: the xArm's 6 joints, then the hand fixed on the flange, then the hand's 15 finger joints
    (taken from the simulated model: every finger body with its hinge and its capsules)"""
    import mujoco
    joints, out_links = [], []
    lk = [Mesh() for _ in range(7)]
    for i, (v, f) in enumerate(links):
        lk[i].add(v, f)
    white, dark = 0xf0f1f3, 0x2a2d33
    out_links.append(lk[0].link(dark))
    for i, ((xyz, rpy), lim) in enumerate(zip(XARM6, XARM6_LIM)):
        joints.append({"name": "j%d" % (i + 1), "type": "revolute", "xyz": list(xyz), "rpy": list(rpy), "axis": [0, 0, 1],
                       "limits": list(lim)})
        out_links.append(lk[i + 1].link(white if i < 5 else dark))
    # the hand: fixed on the flange (its own geoms, and the adapter between them)
    hb = mjm.body("hand").id
    hand = Mesh()
    for g in range(mjm.ngeom):
        if mjm.geom_bodyid[g] == hb and mjm.geom_rgba[g][3] > 0.02:
            v, f = prim({6: "box", 5: "cylinder", 3: "capsule", 2: "sphere", 4: "ellipsoid"}.get(int(mjm.geom_type[g]), ""), list(mjm.geom_size[g]))
            hand.add(v, f, quat_R(mjm.geom_quat[g]), mjm.geom_pos[g])
    v, f = _cylinder(0.036, 0.015)
    hand.add(v, f, R=rot_rpy(0, math.pi / 2, 0), p=(FLANGE_X + 0.015, 0, 0.002))
    r, p, y = rpy_of(R_FH)
    joints.append({"name": "hand", "type": "fixed", "parent": "j6", "xyz": [float(x) * 1000 for x in P_FH], "rpy": [r, p, y], "axis": [0, 0, 1]})
    out_links.append(hand.link(0xd1d5db))
    finger_joints = []
    names = {}
    for b in range(mjm.nbody):
        par = mjm.body_parentid[b]
        if b == hb or not _under(mjm, b, hb):
            continue
        j = mjm.body_jntadr[b]
        jn = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
        pname = "hand" if par == hb else names[par]
        r, p, y = rpy_of(quat_R(mjm.body_quat[b]))
        joints.append({"name": jn, "type": "revolute", "parent": pname, "xyz": [float(x) * 1000 for x in mjm.body_pos[b]],
                       "rpy": [r, p, y], "axis": [float(x) for x in mjm.jnt_axis[j]]})
        names[b] = jn
        finger_joints.append(int(mjm.jnt_qposadr[j]))
        m = Mesh()
        col = 0xb8bec8
        for g in range(mjm.ngeom):
            if mjm.geom_bodyid[g] == b:
                v, f = prim({6: "box", 5: "cylinder", 3: "capsule", 2: "sphere", 4: "ellipsoid"}.get(int(mjm.geom_type[g]), ""), list(mjm.geom_size[g]))
                m.add(v, f, quat_R(mjm.geom_quat[g]), mjm.geom_pos[g])
                col = hexcol(mjm.geom_rgba[g])
        out_links.append(m.link(col))
    model = {"kind": "esasim.robot/1", "robot": ARM, "units": "mm", "scale": 0.1, "angle_units": "rad", "tool": "hand",
             "joints": joints, "links": out_links,
             "license": "xArm meshes and kinematics: xarm_ros by UFACTORY (github.com/xArm-Developer/xarm_ros), BSD 3-Clause."}
    return model, finger_joints


def _under(mjm, b, top):
    while b > 0:
        b = mjm.body_parentid[b]
        if b == top:
            return True
    return False


# ── a 3D file of the conveyor (glTF binary), baked into the cell model ────────────────────────────────────────────
def glb_part_picture(data, word="belt"):
    """the picture of the first part of a .glb whose name has word in it (its material's colour texture, a data URL),
    with the size one picture covers on it, (along its length, across it) in m, from its texture coordinates; or None"""
    import base64
    import struct
    try:
        n = struct.unpack("<I", data[12:16])[0]
        js = json.loads(data[20:20 + n])
        off = 20 + n
        bl = struct.unpack("<I", data[off:off + 4])[0]
        binc = data[off + 8:off + 8 + bl]

        def acc(i):
            a = js["accessors"][i]
            bv = js["bufferViews"][a["bufferView"]]
            k = {"VEC2": 2, "VEC3": 3}[a["type"]]
            st = bv.get("byteStride") or 4 * k
            o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            return np.frombuffer(binc[o:o + st * a["count"]], dtype=np.float32).reshape(a["count"], st // 4)[:, :k].astype(float)
        for m in js.get("meshes", []):
            if word not in str(m.get("name", "")).lower():
                continue
            pr = m["primitives"][0]
            mat = js["materials"][pr["material"]]
            t = (mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
                 or mat.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness", {}).get("diffuseTexture"))
            if not t or "TEXCOORD_0" not in pr["attributes"]:
                return None
            im = js["images"][js["textures"][t["index"]]["source"]]
            bv = js["bufferViews"][im["bufferView"]]
            raw = binc[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]
            P, U = acc(pr["attributes"]["POSITION"]), acc(pr["attributes"]["TEXCOORD_0"])
            ext = P.max(0) - P.min(0)
            lng, wid = int(np.argmax(ext)), int(np.argsort(ext)[1])     # (its length, its width: in the file's axes)
            # metres per whole picture, along and across: from how far the coordinates run over the part
            def per(ax):
                c = [abs(np.corrcoef(U[:, j], P[:, ax])[0, 1]) for j in range(2)]
                j = int(np.argmax(c))
                du = U[:, j].max() - U[:, j].min()
                return j, ext[ax] / max(du, 1e-6)
            ja, along = per(lng)
            _, across = per(wid)
            return {"image": "data:%s;base64,%s" % (im.get("mimeType", "image/png"), base64.b64encode(raw).decode()),
                    "along": float(along), "across": float(across), "along_uv": "uv"[ja]}
    except Exception:      # noqa: BLE001
        return None
    return None


def glb_meshes(data):
    """the triangles of a .glb: [(vertices (m, Z up), faces, colour)] with every node's transform applied"""
    import struct
    n = struct.unpack("<I", data[12:16])[0]
    js = json.loads(data[20:20 + n])
    off = 20 + n
    bl = struct.unpack("<I", data[off:off + 4])[0]
    binc = data[off + 8:off + 8 + bl]
    CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        k = NC[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0) or dt.itemsize * k
        raw = np.frombuffer(binc, dtype=np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * k, offset=start)
        rows = np.lib.stride_tricks.as_strided(raw, shape=(a["count"], dt.itemsize * k), strides=(stride, 1))
        return np.ascontiguousarray(rows).view(dt).reshape(a["count"], k).astype(np.float64 if dt == np.float32 else np.int64)

    def node_T(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"], float).reshape(4, 4).T
        T = np.eye(4)
        if "scale" in nd:
            T = np.diag(list(nd["scale"]) + [1.0]) @ T
        if "rotation" in nd:
            x, y, z, w = nd["rotation"]
            R = quat_R((w, x, y, z))
            M = np.eye(4); M[:3, :3] = R
            T = M @ T
        if "translation" in nd:
            M = np.eye(4); M[:3, 3] = nd["translation"]
            T = M @ T
        return T

    cols = []
    for mt in js.get("materials", []):
        c = (mt.get("pbrMetallicRoughness") or {}).get("baseColorFactor")
        cols.append(c)
    out = []

    def walk(i, Tp):
        nd = js["nodes"][i]
        T = Tp @ node_T(nd)
        if "mesh" in nd:
            mesh = js["meshes"][nd["mesh"]]
            for pr in mesh["primitives"]:
                if pr.get("mode", 4) != 4:
                    continue
                v = acc(pr["attributes"]["POSITION"])
                v = v @ T[:3, :3].T + T[:3, 3]
                f = acc(pr["indices"]).reshape(-1, 3) if "indices" in pr else np.arange(len(v)).reshape(-1, 3)
                mi = pr.get("material")
                c = cols[mi] if mi is not None and mi < len(cols) else None
                mname = js["materials"][mi].get("name", "") if mi is not None and mi < len(js.get("materials", [])) else ""
                out.append((v, f, c, " ".join([str(nd.get("name", "")), str(mesh.get("name", "")), str(mname)]).lower()))
        for ch in nd.get("children", []):
            walk(ch, T)
    sc = js.get("scenes", [{"nodes": list(range(len(js.get("nodes", []))))}])[js.get("scene", 0)]
    Yup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])        # (glTF is Y up; the cell is Z up)
    for i in sc["nodes"]:
        walk(i, Yup)
    return out


CONVEYOR = {"mode": "flat", "fit": None, "asset": None}   # mode: textured (the 3D view draws the file), flat (baked in colours), boxes
CONVEYOR_LEN = 2 * S.BELT_HALF_LEN   # (m) the simulated conveyor's length (its 3D file at its own proportions)
FLOOR_Z = -0.75                   # the floor (the xArm's pedestal is 75 cm tall)


def part_colour(c, name):
    """a part's colour: its material's, else one by its name (a textured model has none)"""
    if c is not None:
        return hexcol(c)
    if "belt" in name or "rubber" in name:
        return 0x1d1f23
    if "leg" in name or "stand" in name or "foot" in name or "feet" in name:
        return 0x4a4f57
    return 0x9aa1aa


def conveyor_mesh(path, log):
    """the conveyor's 3D file at its own proportions: scaled alike in every direction so that its belt's top is on the
    simulated belt (its feet then on the floor), its long side along the belt, its belt centred on the simulated one.
    The simulated belt was measured from this file (scene.py): a file of other proportions is said so in the log."""
    data = open(path, "rb").read()
    parts = glb_meshes(data)
    if not parts:
        return None
    pic = glb_part_picture(data)
    allv = np.concatenate([p[0] for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    ext = hi - lo
    turn = ext[1] > ext[0]                         # (its long side along y: turned a quarter so it runs along x)
    if turn:
        Rq = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
        parts = [(v @ Rq.T, f, c, nm) for v, f, c, nm in parts]
        allv = np.concatenate([p[0] for p in parts])
        lo, hi = allv.min(0), allv.max(0)
        ext = hi - lo
    belt = [p for p in parts if "belt" in p[3]]
    if belt:
        bv = np.concatenate([p[0] for p in belt])
        b_lo, b_hi = bv.min(0), bv.max(0)
    else:                                          # (no part named belt: the belt is taken as the whole width, under the top)
        b_lo, b_hi = lo.copy(), hi.copy()
    k = (S.BELT_TOP - FLOOR_Z) / max(b_hi[2] - lo[2], 1e-6)       # (one scale for every direction)
    sc = np.array([k, k, k])
    shift = np.array([S.BELT_X0, S.BELT_CY, S.BELT_TOP]) - np.array([(b_lo[0] + b_hi[0]) / 2, (b_lo[1] + b_hi[1]) / 2, b_hi[2]]) * sc
    blen, bw = (b_hi[0] - b_lo[0]) * k, (b_hi[1] - b_lo[1]) * k
    log("Conveyor model: %d parts, %d triangles; %.2f m long, %.2f m wide, %.2f m high in its file%s, its belt %.2f m wide with its top at %.2f m. "
        "Drawn at its own proportions, scaled %.3f: %.2f m long, its belt %.2f m wide, its top on the belt and its feet on the floor."
        % (len(parts), sum(len(p[1]) for p in parts), ext[0], ext[1], ext[2], ", turned a quarter" if turn else "", b_hi[1] - b_lo[1], b_hi[2] - lo[2],
           k, ext[0] * k, bw))
    if abs(blen - CONVEYOR_LEN) > 0.03 or abs(bw - S.BELT_W) > 0.03:
        log("Note: the simulated belt is %.2f m long and %.2f m wide (scene.py), this file's %.2f m and %.2f m: the objects ride "
            "on the simulated one." % (CONVEYOR_LEN, S.BELT_W, blen, bw))
    CONVEYOR["fit"] = {"kind": "model", "id": "conveyor", "asset": os.path.basename(path), "model_units": "m", "up": "y",
                       "rpy": [0, 0, round(math.pi / 2, 6) if turn else 0], "scale": [round(float(x), 5) for x in sc],
                       "pos": [round(float(x) * 1000, 1) for x in shift]}
    by_col = {}
    for v, f, c, nm in parts:
        key = ("belt",) if "belt" in nm else part_colour(c, nm)     # (the belt apart: it carries the moving texture)
        by_col.setdefault(key, []).append((v * sc + shift, f, part_colour(c, nm)))
    links = []
    for key, items in by_col.items():
        m = Mesh()
        for v, f, _ in items:
            vv, ff = decimate(v[f] * 1000.0, 2.0) if len(f) > 3000 else (v * 1000.0, f)
            m.add(vv / 1000.0, ff)
        tex = None
        if key == ("belt",):
            tex = dict(BELT_TEXTURE)
            if pic:            # (the belt's own picture, from the file, at the file's own scale on the belt)
                # (one picture every BELT_PERIOD along the belt, its proportions kept across: within a few % of the
                # file's own scale, and the belt's position is sent within that period, so it scrolls without a seam)
                across = BELT_PERIOD * pic["across"] / pic["along"]
                tex.update({"pattern": "image", "image": pic["image"], "along_uv": pic["along_uv"],
                            "period": round(BELT_PERIOD * 1000, 1), "across": round(across * 1000, 1)})
                log("The belt's own texture: one picture every %.2f m along the belt (%.2f m in the file), %.2f m across; it "
                    "scrolls with the belt (a 3D view that draws link textures)." % (BELT_PERIOD, pic["along"] * float(sc[0]), across))
        links.append((items[0][2], m, tex))
    return links


# the texture of the belt's surface, for the 3D view: rubber with a fine grain and chevrons pointing the way the belt
# runs (towards -x), one every BELT_PERIOD, laid along x over the belt part and scrolled with the objects model's
# joint 0 (belt_x: where the belt is within a period, in mm)
BELT_PERIOD = 2.0                 # (m) the belt's texture repeats every BELT_PERIOD along it; its position is sent within it
BELT_TEXTURE = {"pattern": "belt", "period": BELT_PERIOD * 1000, "marks": 20, "axis": "x", "direction": -1, "follow_joint": 0,
                "color": 0x1d1f23, "mark_color": 0x3a3f47}


def objects_model(info, conveyor=None):
    """belt_objects.robot.json: the belt's slide (along x: its value, where the belt is within BELT_PERIOD, scrolls the
    texture of the conveyor's belt), each object on its own free mount, then the conveyor (its 3D file, if given, as
    fixed parts; its belt a part of its own, with that texture)"""
    joints, links = [], [{}]
    joints.append({"name": "belt_x", "type": "prismatic", "parent": "base", "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [1, 0, 0]})
    links.append({})       # (no slats: the belt's own surface carries a texture that the 3D view scrolls with this joint)
    for k, o in enumerate(info):
        par = "base"
        for nm, typ, axis in (("x", "prismatic", [1, 0, 0]), ("y", "prismatic", [0, 1, 0]), ("z", "prismatic", [0, 0, 1]),
                              ("a", "revolute", [0, 0, 1]), ("b", "revolute", [0, 1, 0]), ("c", "revolute", [1, 0, 0])):
            jn = "%s_%s" % (o["key"], nm)
            joints.append({"name": jn, "type": typ, "parent": par, "xyz": [0, 0, 0],
                           "rpy": [0, 0, 0], "axis": axis})
            par = jn
            links.append({})
        m = Mesh()
        for pt in o["parts"]:
            t, size, pos, eul = (pt["type"], pt["size"], pt["pos"], pt["euler"]) if isinstance(pt, dict) else pt
            v, f = prim(t, size)
            m.add(v, f, quat_R(S.part_quat(eul)), pos)
        links[-1] = m.link(hexcol(o["rgba"]))
    # the conveyor's parts last: the 3D view gives the joint items' values to the joints in their order, fixed ones too
    for k, (col, m, tex) in enumerate(conveyor or []):
        joints.append({"name": "conveyor_%d" % k, "type": "fixed", "parent": "base", "xyz": [0, 0, 0], "rpy": [0, 0, 0], "axis": [0, 0, 1]})
        lk = m.link(col)
        if tex and lk:
            lk["texture"] = tex
        links.append(lk)
    return {"kind": "esasim.robot/1", "robot": OBJS, "units": "mm", "scale": 0.1, "angle_units": "rad", "joints": joints, "links": links,
            "base": {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}}


def write_models(ctx, port, mjm, info, cache, conveyor_file=None):
    links = xarm_links(cache, ctx.log) or simple_xarm_links()
    arm, _ = arm_model(mjm, links)
    conv = None
    if conveyor_file and CONVEYOR["mode"] == "textured":
        conveyor_mesh(conveyor_file, ctx.log)
        ctx.log("The conveyor is drawn by the 3D view from its own file, with its materials: not baked into the objects model.")
    elif conveyor_file:
        try:
            conv = conveyor_mesh(conveyor_file, ctx.log)
        except Exception as e:      # noqa: BLE001
            ctx.log("The conveyor's 3D file could not be read (%s): the belt is drawn with boxes." % e)
    objs = objects_model(info, conv)
    objs["conveyor"] = bool(conv)
    for name, model in ((ARM, arm), (OBJS, objs)):
        raw = json.dumps(model, separators=(",", ":")).encode()
        with open(ctx.output(port, "%s.robot.json.gz" % name), "wb") as fh:
            fh.write(gzip.compress(raw, 9))
        ctx.log("%s.robot.json.gz: %d joints, %d KB." % (name, len(model["joints"]), len(gzip.compress(raw, 9)) // 1024))


# ── the arm's angles: inverse kinematics on the hand's pose ───────────────────────────────────────────────────────
def fk(q):
    T = np.eye(4)
    for (xyz, rpy), a in zip(XARM6, q):
        J = np.eye(4); J[:3, :3] = rot_rpy(*rpy); J[:3, 3] = np.array(xyz) / 1000.0
        Z = np.eye(4); Z[:3, :3] = rz(a)
        T = T @ J @ Z
    return T


LO = np.radians([l for l, h in XARM6_LIM])
HI = np.radians([h for l, h in XARM6_LIM])
SEEDS = [np.array(s) for s in ([0.6, -0.4, -1.2, 0, 1.4, 0], [1.2, -0.3, -1.3, 0, 1.4, 0], [0.0, -0.5, -1.0, 0, 1.2, 0],
                               [2.2, 0.0, -1.0, 0, 1.0, 0], [-1.0, -0.3, -1.2, 0, 1.3, 0])]


def err6(T, Tt):
    e = Tt[:3, 3] - T[:3, 3]
    Re = Tt[:3, :3] @ T[:3, :3].T
    v = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    c = max(-1.0, min(1.0, (np.trace(Re) - 1) / 2))
    ang = math.acos(c)
    s = np.linalg.norm(v)
    if s > 1e-9:
        w = v / s * ang
    elif c > 0:
        w = v
    else:                                       # (half a turn: about the axis Re leaves alone)
        k = int(np.argmax(np.diag(Re)))
        ax = Re[:, k] + np.eye(3)[k]
        w = ax / np.linalg.norm(ax) * math.pi
    return np.concatenate([e, 0.25 * w])


def ik_from(q, Tt, iters=30):
    q = q.copy()
    for _ in range(iters):
        e = err6(fk(q), Tt)
        if np.linalg.norm(e) < 2e-4:
            break
        J = np.zeros((6, 6))
        for j in range(6):
            q2 = q.copy(); q2[j] += 1e-4
            J[:, j] = (err6(fk(q2), Tt) - e) / -1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + 4e-4 * np.eye(6), e)
        q = np.clip(q + np.clip(dq, -0.3, 0.3), LO, HI)
    e = err6(fk(q), Tt)
    return q, float(np.linalg.norm(e[:3]))


class ArmIK:
    """the xArm's angles for where the simulation has the hand (warm-started from the last ones, so the arm moves
    smoothly)"""

    def __init__(self):
        self.q = SEEDS[0].copy()

    def solve(self, hand_pos, hand_R):
        Tt = np.eye(4)
        Tt[:3, :3] = hand_R @ R_FH.T
        Tt[:3, 3] = hand_pos - Tt[:3, :3] @ P_FH
        q, e = ik_from(self.q, Tt)
        if e > 0.003:
            for s in SEEDS:
                q2, e2 = ik_from(s, Tt, 60)
                if e2 < e:
                    q, e = q2, e2
        self.q = q
        return q, e


# ── the items the 3D view gets ────────────────────────────────────────────────────────────────────────────────────
CONVEYOR_DRAWN = {"value": False}     # (set when the objects model carries the conveyor's 3D file)


def cell_objects(mjm):
    """the cell around the robot (scene channel): belt, frame, rails, bin, pedestal, the camera on its pole"""
    import mujoco
    d = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, d)
    objs = []
    belt_parts = () if CONVEYOR_DRAWN["value"] else (("belt_frame", 0x5a6068), ("rail_l", 0x9aa1aa), ("rail_r", 0x9aa1aa))
    for nm, col in belt_parts + (("bin_floor", 0x2f6db5), ("bin_w1", 0x3a7fd0),
                    ("bin_w2", 0x3a7fd0), ("bin_w3", 0x3a7fd0), ("bin_w4", 0x3a7fd0), ("bin_stand", 0x4a4f57)):
        g = mjm.geom(nm).id
        objs.append({"id": nm, "kind": "box", "center": [round(float(x) * 1000, 1) for x in d.geom_xpos[g]],
                     "size": [round(float(x) * 2000, 1) for x in mjm.geom_size[g]], "color": col})
    if not CONVEYOR_DRAWN["value"]:
        objs.append({"id": "belt", "kind": "box", "center": [round(S.BELT_X0 * 1000, 1), round(S.BELT_CY * 1000, 1), -12.0],
                     "size": [round(CONVEYOR_LEN * 1000, 1), round(S.BELT_W * 1000, 1), 24.0], "color": 0x1d1f23, "roughness": 0.95})
    if CONVEYOR["mode"] == "textured" and CONVEYOR["fit"]:
        fit = dict(CONVEYOR["fit"])
        fit["asset"] = CONVEYOR.get("asset") or fit["asset"]
        objs.append(fit)
    objs.append({"id": "pedestal", "kind": "box", "center": [round(1000 * S.BASE_POS[0], 1), round(1000 * S.BASE_POS[1], 1), -375.0], "size": [180.0, 180.0, 750.0], "color": 0x3a3f47})
    cam = mjm.camera("cam").id
    R = d.cam_xmat[cam].reshape(3, 3)
    objs.append({"id": "camera", "kind": "camera", "pos": [round(float(x) * 1000, 1) for x in d.cam_xpos[cam]],
                 "forward": [round(float(x), 4) for x in -R[:, 2]], "up": [round(float(x), 4) for x in R[:, 1]],
                 "hfov": round(2 * math.degrees(math.atan(math.tan(math.radians(S.CAM_FOVY) / 2) * S.CAM_W / S.CAM_H)), 1),
                 "aspect": round(S.CAM_W / S.CAM_H, 3), "frustum": 250, "stand_to": -750})
    # (the camera's picture is shown in a view of its own, beside the 3D view: not on a sheet in the cell)
    # the two robots where the simulation has them: the xArm's base on its pedestal at the origin, the objects' mounts
    # at the origin too (else the view stands robots without a place 0.8 m apart)
    objs.append({"id": ARM, "kind": "robot", "robot": ARM, "pos": [round(1000 * S.BASE_POS[0], 1), round(1000 * S.BASE_POS[1], 1), 0], "rpy": [0, 0, round(S.BASE_YAW, 6)]})
    objs.append({"id": OBJS, "kind": "robot", "robot": OBJS, "pos": [0, 0, 0], "rpy": [0, 0, 0]})
    return {"units": "mm", "up": "z", "floor": -750, "objects": objs}


class Pose3D:
    """turns a cell of the simulation into joint items for the 3D view"""

    def __init__(self, env):
        mjm = env.mjm
        self.env = env
        self.hand = mjm.body("hand").id
        import mujoco
        hb = self.hand
        self.fq = []
        for b in range(mjm.nbody):
            if b != hb and _under(mjm, b, hb):
                self.fq.append(int(mjm.jnt_qposadr[mjm.body_jntadr[b]]))
        self.fq_t = env.t.as_tensor(self.fq, device=env.dev)
        self.nobj = len(env.info)
        self.err = 0.0
        self.scene = cell_objects(mjm)      # (sent again every few seconds: a view opened later still gets the cell)
        self.n_sent = 0
        self.last_slide = None
        self.last_obj, self.last_eul = None, None
        self.n_q = None

    def snap(self, i=0):
        """what the 3D view needs of cell i, now (kept to be turned into joint items later, off the learning's way)"""
        env, sim = self.env, self.env.sim
        return (sim.qpos[i, env.qh].float().cpu().numpy().astype(float), env.touches()[i].cpu().numpy().astype(float),
                sim.qpos[i, self.fq_t].float().cpu().numpy().astype(float), int(env.obj[i]),
                sim.xpos[i, env.ob].float().cpu().numpy().astype(float), sim.xquat[i, env.ob].float().cpu().numpy().astype(float),
                -float(env.belt_travel[i]))

    def snap_many(self, cells):
        """snap() for several cells, read from the simulation in one go"""
        env, sim = self.env, self.env.sim
        ix = env.t.as_tensor(cells, device=env.dev)
        q = sim.qpos[ix][:, env.qh].float().cpu().numpy().astype(float)
        f = env.touches()[ix].cpu().numpy().astype(float)
        fq = sim.qpos[ix][:, self.fq_t].float().cpu().numpy().astype(float)
        ob = env.obj[ix].tolist()
        op = sim.xpos[ix, env.ob].float().cpu().numpy().astype(float)
        oq = sim.xquat[ix, env.ob].float().cpu().numpy().astype(float)
        bt = (-env.belt_travel[ix].float()).tolist()
        return {c: (q[k], f[k], fq[k], int(ob[k]), op[k], oq[k], float(bt[k])) for k, c in enumerate(cells)}

    def items(self, sn):
        q, _, fq, o, op, oq, belt = sn            # (the arm's angles are the simulated arm's own)
        arm = [round(float(a), 4) for a in q] + [0.0] + [round(float(x), 4) for x in fq]
        if o != self.last_obj:
            self.last_eul = None
        self.last_obj = o
        r, p, y = self.last_eul = rpy_near(quat_R(oq), self.last_eul)
        gap = BELT_PERIOD
        slide = round((belt % gap) * 1000, 1)               # (the texture repeats every gap: where in it the belt is)
        # when the slats wrap round to the start of a gap, the view must not slide them back across it (it eases
        # between items): that item carries no value for them, so they jump, unseen, to where the next one puts them
        wrap = self.last_slide is not None and slide > self.last_slide + 0.5 * gap * 1000
        self.last_slide = slide
        if wrap:
            slide = None
        objs = [slide] + [0.0, 0.0, PARK_MM, 0.0, 0.0, 0.0] * self.nobj      # (every item complete: the objects not in play under the floor)
        objs[1 + 6 * o:7 + 6 * o] = [round(op[0] * 1000, 1), round(op[1] * 1000, 1), round(op[2] * 1000, 1), round(y, 4), round(p, 4), round(r, 4)]
        self.n_q = (len(arm), len(objs))
        return ({"robot": ARM, "units": "rad", "q": arm}, {"robot": OBJS, "units": "rad", "q": objs})

    def cut(self, ctx):
        """a break between two episodes: the 3D view eases from one item to the next, so without it the arm and the
        object would glide from where one episode ended to where the next begins (an object flying across the cell,
        or up from under the floor where the objects not in play wait). An item with no values in it is not eased
        to or from: the view holds the last pose, then jumps to the next episode's first one"""
        self.last_slide, self.last_obj, self.last_eul = None, None, None
        if not self.n_q:
            return
        ctx.telemetry("joints", {"robot": ARM, "units": "rad", "q": [None] * self.n_q[0]})
        time.sleep(0.002)
        ctx.telemetry("joints", {"robot": OBJS, "units": "rad", "q": [None] * self.n_q[1]})
        time.sleep(0.03)

    def send(self, ctx, sn):
        if self.n_sent % 40 == 0:
            ctx.telemetry("scene", self.scene)
        self.n_sent += 1
        a, o = self.items(sn)
        if sn[1] is not None and self.n_sent % 3 == 1:
            ctx.telemetry("touch", touch_row(sn[1]))
        ctx.telemetry("joints", a)
        time.sleep(0.002)       # (the 3D view takes two items of a channel with the same time stamp for one: the objects' would be lost)
        ctx.telemetry("joints", o)


def touch_row(f):
    """the hand's 19 pressure sensors (N) as the gauges and the hand map show them: each one by its name (thumb_tip,
    index_mid, little_prox, palm_1 ...), each digit in all, the palm, the grip in all, and whether it is holding (the
    thumb, or the palm, pressing against at least one finger)"""
    f = [max(0.0, float(x)) for x in f]
    row = {n: round(x, 2) for n, x in zip(S.TOUCH_NAMES, f)}
    by = dict(zip(S.TOUCH_NAMES, f))
    dig = {d: sum(by.get(d + s, 0.0) for s in ("_tip", "_mid", "_prox")) for d in S.DIGITS}
    palm = sum(by.get("palm_%d" % k, 0.0) for k in range(1, 5))
    for d in S.DIGITS:
        row[d + "_n"] = round(dig[d], 2)
    row["palm_n"] = round(palm, 2)
    row["grip_n"] = round(sum(f), 2)
    row["touching"] = sum(1 for x in f if x > 0.3)
    row["holding"] = 1 if (dig["thumb"] > 0.5 or palm > 0.5) and max(dig[d] for d in S.DIGITS[1:]) > 0.5 else 0
    return row


class TelemetryBudget:
    """EsaSim keeps the first 20 000 live items of a run and drops the rest without a word: sent as fast as the cell
    moves (two poses, the touch and pictures, 30 a second), the 3D view, the gauges and the charts all stop after
    11 minutes. This takes the place of ctx.telemetry and spreads the run's items over its minutes:
    - the learning curve, the events and the scene always go (they are few), until the very end of the allowance;
    - the live items (the poses, the touch, the camera's pictures) share the rest, evenly over the run: whole
      episodes are bought before they are played (afford), so what is shown is shown smoothly, and the view waits
      between them rather than stuttering through all of them"""
    CAP = 20000
    LIVE = ("joints", "touch", "camera_picture")

    def __init__(self, send, seconds, cap=19400, reserve=3000, log=None):
        import threading
        self.send, self.cap, self.reserve, self.log = send, cap, reserve, log
        self.t0, self.seconds = time.time(), max(120.0, float(seconds))
        self.used, self.live_used, self.pool, self.dropped = 0, 0, 0, 0
        self.lock = threading.Lock()

    def _allowance(self):
        frac = min(1.0, (time.time() - self.t0) / self.seconds)
        return (self.cap - self.reserve) * frac + 400 - self.live_used        # (400: the first episode at once)

    def afford(self, n):
        """n live items for an episode about to be played: True and kept for it, or False (not yet: try later)"""
        with self.lock:
            if self._allowance() >= n and self.used + n <= self.cap - 300:
                self.pool += n
                self.live_used += n
                return True
            return False

    def settle(self):
        """the episode is over: what it did not use goes back"""
        with self.lock:
            self.live_used -= self.pool
            self.pool = 0

    def __call__(self, channel, data):
        with self.lock:
            if self.used >= self.cap:
                return
            if channel in self.LIVE:
                if self.pool > 0:
                    self.pool -= 1
                elif self._allowance() >= 1 and self.used < self.cap - 300:
                    self.live_used += 1
                else:
                    self.dropped += 1
                    return
            self.used += 1
        self.send(channel, data)


def budget_telemetry(ctx, seconds):
    """ctx.telemetry from now on within EsaSim's 20 000 items of a run (see TelemetryBudget)"""
    b = TelemetryBudget(ctx.telemetry, seconds, log=ctx.log)
    try:
        ctx.telemetry = b
    except Exception:      # noqa: BLE001
        pass
    return b


def afford(ctx, n):
    """(the live cell, the replays) may an episode of about n live items be shown now?"""
    f = getattr(ctx.telemetry, "afford", None)
    return True if f is None else f(n)


def settle(ctx):
    f = getattr(ctx.telemetry, "settle", None)
    if f is not None:
        f()


class Replayer:
    """plays the episodes of one cell in the 3D view in real time, on a thread of its own (learning goes on meanwhile)"""

    def __init__(self, ctx, pose, dt):
        import queue
        import threading
        self.ctx, self.pose, self.dt = ctx, pose, dt
        self.q = queue.Queue(maxsize=64)
        self.alive = True
        self.playing = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def idle(self):
        """nothing playing nor waiting: the next episode can be shown at once"""
        return not self.playing and self.q.empty()

    def waiting(self):
        """episodes in the queue, not yet shown"""
        return self.q.qsize()

    def play(self, frames, text=None, picture=None):
        try:
            self.q.put_nowait((frames, text, picture))
        except Exception:          # noqa: BLE001  (busy: this one is skipped)
            pass

    def _run(self):
        import time
        while self.alive:
            try:
                frames, text, picture = self.q.get(timeout=0.5)
            except Exception:      # noqa: BLE001
                continue
            if not afford(self.ctx, int(len(frames) * 2.6) + 6):
                continue            # (the run's live items would run out: this episode is not shown)
            self.playing = True
            try:
                if text:
                    self.ctx.telemetry("events", {"text": text})
                if picture:
                    self.ctx.telemetry("camera_picture", {"image": picture})
                self.pose.cut(self.ctx)
                t0 = time.time()
                for k, sn in enumerate(frames):
                    if not self.alive:
                        break
                    pic = None
                    if len(sn) == 2:            # (a frame with the camera's picture of that moment)
                        sn, pic = sn
                    self.pose.send(self.ctx, sn)
                    if pic is not None:
                        self.ctx.telemetry("camera_picture", {"image": rgb_png(pic)})
                    lag = t0 + (k + 1) * self.dt - time.time()
                    if lag > 0:
                        time.sleep(lag)
            except Exception as e:  # noqa: BLE001
                self.ctx.log("3D view: %s" % e)
            finally:
                settle(self.ctx)
                time.sleep(0.4)     # (a breath between episodes)
                self.playing = False

    def stop(self):
        self.alive = False


def rgb_png(pic):
    """a camera picture kept with a replay (rgb (h, w, 3) uint8, its background mask) as a PNG, twice its size"""
    rgb, bg = pic
    rgb = np.where(bg[..., None], np.array([28, 30, 34], np.uint8), rgb)
    rgb = np.repeat(np.repeat(rgb, 2, 0), 2, 1)
    return png_rgb(np.ascontiguousarray(rgb))


def png_rgb(rgb):
    """an (h, w, 3) uint8 picture as a PNG data URL"""
    import base64
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        return c + struct.pack(">I", zlib.crc32(t + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()
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
        self.lr_max = 1e-3            # (the adaptive step never beyond this: larger steps undid what had been learned)
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
            self.b_rew[t], self.b_done[t] = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0), done.float()

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
                            self.lr = min(getattr(self, "lr_max", 1e-2), self.lr * 1.5)
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
                bc = getattr(self, "bc", None)
                if bc is not None and not critic_only and getattr(self, "bc_coef", 0) > 0.01:
                    # (after learning from demonstrations: a term that keeps the policy near them, fading update by update)
                    jb = torch.randint(len(bc[0]), (4096,))
                    ob, ab = bc[0][jb].to(self.dev).float(), bc[1][jb].to(self.dev).float()
                    loss = loss + self.bc_coef * ((self.ac.actor(self.on(ob)) - ab) ** 2).mean()
                if not bool(torch.isfinite(loss)):
                    continue                                    # (a batch with a broken number in it: skipped, the networks stay sound)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
                stats["kl"], stats["vloss"], stats["ploss"] = float(kl), float(vl.detach()), float(pl.detach())
        if getattr(self, "bc", None) is not None and not critic_only:
            self.bc_coef = max(0.1, getattr(self, "bc_coef", 1.0) * 0.993)
            self.lr = min(self.lr, getattr(self, "lr_max", 1e-3))
            for g in self.opt.param_groups:
                g["lr"] = self.lr
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
        self.lr = min(float(st.get("lr") or self.lr), 3e-4)
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
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], want - w.shape[1], dtype=w.dtype, device=w.device)], 1)
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
from ppo_part import PPO, chunks, make_ac, Norm, policy_stamp, widen
import viz

FORMAT = "esasim.grasp-policy/1"
SEE_FORMAT = "esasim.grasp-see/2"          # (2: the network finds the object's pixels, then its box)
VIEW = "esasim.grasp/1"
SHOW_CELLS = 32                   # cells recorded for the 3D view's replays (each one is recorded at every step: more of them slow the learning)
REPLAY_QUEUE = 40                 # episodes waiting their turn in the 3D view (they come in bursts: the cells advance together)

STAGE_NAME = {"reach": "Reach", "grasp": "Grasp", "place": "Place"}


def object_item(env, i):
    o = env.info[int(env.obj[i])]
    parts = []
    for pt in o["parts"]:
        t, size, pos, eul = (pt["type"], pt["size"], pt["pos"], pt["euler"]) if isinstance(pt, dict) else pt
        parts.append({"t": t, "s": [round(float(x), 5) for x in size], "p": [round(float(x), 5) for x in pos],
                      "q": [round(float(x), 6) for x in S.part_quat(eul)]})
    return {"key": o["key"], "label": o["label"], "test": bool(o["test"]), "rgba": [round(float(x), 3) for x in o["rgba"]], "parts": parts}


PIC_EVERY = 12                    # steps between the camera pictures kept with a replay (0.5 s)


def camera_pics(env, n):
    """the camera's pictures of cells 0..n-1 now, as the network sees them (rgb uint8, background mask), for replays"""
    img = env.last_img[:n]
    rgb = (img[:, 3:6].clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    bg = (env.last_seg[:n] < 0)
    bg = (bg[..., 0] if bg.dim() == 4 else bg).cpu().numpy()
    return [(rgb[i], bg[i]) for i in range(n)]


def camera_png(env, i):
    """the camera's picture of cell i as the network sees it (its colours, shaded), as a PNG for the 3D view"""
    img = env.last_img[i]
    rgb = (img[3:6].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    seg = env.last_seg[i].cpu().numpy()
    rgb = np.where((seg < 0)[..., None], np.array([28, 30, 34], np.uint8), rgb)
    rgb = np.repeat(np.repeat(rgb, 2, 0), 2, 1)
    return viz.png_rgb(np.ascontiguousarray(rgb))


def episode_result(info, i, stage):
    return {"success": bool(info["success"][i]), "fail": bool(info["fail"][i])}


def results_text(stage, ok, fail):
    if ok:
        return {"reach": "reached it", "grasp": "grasped it and held it", "place": "put it in the bin"}[stage]
    return "dropped it or let it go by" if fail else "ran out of time"


# ── the live cell: the policy as it is now, playing in real time while it learns ──────────────────────────────
class LiveCell:
    """one cell of its own, on the CPU, where the policy being learned plays episode after episode in real time (its
    mean action, the hand starting at rest), for the 3D view, the camera's picture and the hand's sensors: the
    thousands of learning cells each advance far slower than real time, so they cannot be watched. After every
    update of the learning its newest weights come here, so what the view shows is the robot as it is now."""

    def __init__(self, ctx, stage, seed, dt):
        import threading
        import torch
        import mujoco
        import cell
        self.ctx, self.stage, self.dt, self.t = ctx, stage, dt, torch
        lib = cell.compile_library(mujoco)
        self.sim, _ = make_sim(lib[0], lib, 1, "CPU", lambda *a: None, cam=True)
        self.env = GraspEnv(torch, self.sim, lib, stage=stage, seed=seed + 7919)
        self.env.assist = 0.0
        see, _ = load_see(ctx, "cpu")
        if see is not None:
            self.env.see = see
        self.ac = make_ac(torch, OBS_DIM, PRIV_DIM, ACT_DIM)
        self.ac.eval()
        self.on = Norm(torch, OBS_DIM, "cpu")
        self.pose = viz.Pose3D(self.env)
        self.lock = threading.Lock()
        self.pending, self.version, self.ready = None, 0, False
        self.alive = True
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def update(self, algo, it):
        """the learning's newest weights (copied: the learning goes on changing its own)"""
        sd = {k: v.detach().to("cpu", copy=True) for k, v in algo.ac.state_dict().items()}
        st = {k: v.detach().to("cpu", copy=True) for k, v in algo.on.state().items()}
        with self.lock:
            self.pending = (sd, st, it)

    def stop(self):
        self.alive = False

    def _take(self):
        with self.lock:
            p, self.pending = self.pending, None
        if p is not None:
            self.ac.load_state_dict(p[0])
            self.on.load(p[1], "cpu")
            self.version, self.ready = p[2], True

    def _run(self):
        torch, env, sim = self.t, self.env, self.sim
        ep = 0
        while self.alive:
            try:
                self._take()
                if not self.ready:
                    time.sleep(0.2)
                    continue
                # (the episode's live items, at most: a pose every 3 steps (2 items, the touch every 3rd), the pictures)
                cost = int(env.max_steps / 3 * 2.4 + env.max_steps / PIC_EVERY) + 8
                if not viz.afford(self.ctx, cost):
                    self.waited = getattr(self, "waited", 0) + 1
                    time.sleep(1.0)     # (the run's live items are spread over its minutes: the next episode in a moment)
                    continue
                env.reset_all()
                sim.refresh()
                env.perceive()
                obj = object_item(env, 0)
                self.pose.cut(self.ctx)
                ep += 1
                t0, k = time.time(), 0
                cam_err, near, curl_max, lift_max = [], 9.0, 0.0, 0.0
                while self.alive:
                    obs, _ = env.obs()
                    with torch.no_grad():       # (as it acts while it learns: its moves with their spread; the fingers'
                        mu = self.ac.actor(self.on(obs))     # commands add up step by step, and that spread is what closes them)
                        a = mu + self.ac.log_std.exp() * torch.randn_like(mu)
                    _, done, info = env.step(a)
                    k += 1
                    curl_max = max(curl_max, float(env.curl[0].mean()))
                    lift_max = max(lift_max, float(info["lift"][0]))
                    c_true = env.truth()[2]
                    if float(env.est_age[0]) < 1e-6:          # (a camera frame that saw it: how far off its box was)
                        cam_err.append(float((env.est_c[0, :2] - c_true[0, :2]).norm()))
                    g, _ = env.hand_state()
                    near = min(near, float((g[0] - c_true[0]).norm()))
                    if k % 3 == 0:
                        self.pose.send(self.ctx, self.pose.snap(0))
                    if k % PIC_EVERY == 0:
                        self.ctx.telemetry("camera_picture", {"image": camera_png(env, 0)})
                    lag = t0 + k * CTRL_DT - time.time()
                    if lag > 0:
                        time.sleep(lag)
                    if bool(done[0]):
                        ok, fl = bool(info["success"][0]), bool(info["fail"][0])
                        txt = "Live, the policy after %d updates: %s, %s after %.1f s (camera %.1f cm off; hand within %.1f cm; fingers closed up to %.0f%%; lifted up to %.1f cm)" % (
                            self.version, obj["label"], results_text(self.stage, ok, fl), k * CTRL_DT,
                            100 * sum(cam_err) / max(1, len(cam_err)), 100 * near, 100 * curl_max, 100 * lift_max)
                        self.ctx.telemetry("events", {"text": txt})
                        self.ctx.log(txt)
                        self.pose.send(self.ctx, self.pose.snap(0))
                        time.sleep(0.8)          # (a moment to see how it ended)
                        viz.settle(self.ctx)
                        break
            except Exception as e:      # noqa: BLE001
                self.ctx.log("Live cell: %s" % e)
                time.sleep(2.0)


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
        nets = st.get("ac") or st.get("net") or {}
        if any(torch.is_tensor(v) and v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in nets.values()):
            ctx.log("Skipped %s: its network is broken (numbers that are not numbers)." % name)
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


def cell_signature():
    """a short fingerprint of the cell the policies learn in (its geometry, the arm, the hand, the belt, the bin): a
    policy learned in another version of the cell is not carried on from"""
    import hashlib
    keep = S.ARM_MESHES
    S.ARM_MESHES = None         # (whether the camera has the arm's meshes does not change the cell)
    try:
        import grasp_env
        return hashlib.md5((S.scene_xml() + "curl %g" % grasp_env.CURL_RATE).encode()).hexdigest()[:12]
    finally:
        S.ARM_MESHES = keep


def pick_policy(ctx, found, stage):
    """the policy to carry on from for a stage: the furthest stage there is up to this one, but never one older than a
    policy of an earlier stage (a place policy from before the grasp was learned again belongs to the old grasp),
    nor one learned in another version of the cell; among those the newest"""
    order = {s: k for k, s in enumerate(STAGES)}
    sig = cell_signature()
    ok = []
    for f in found:
        if order.get(f["stage"], 9) > order[stage]:
            continue
        cs = f["st"].get("cell")
        if cs != sig:           # (no fingerprint: from before it was recorded, so from an older cell as well)
            ctx.log("Not carrying on from %s: it learned in another version of the cell." % f["name"])
            continue
        ok.append(f)
    # the furthest stage first; within it the best success, then the newest (a newer but worse policy of an earlier
    # stage no longer throws away a good one of a later stage)
    cur = list(ok)
    cur.sort(key=lambda f: (order.get(f["stage"], 0), round(float(f["st"].get("score") or 0)), f["stamp"]), reverse=True)
    return cur


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
    ctx.telemetry("scene", viz.cell_objects(lib[0]))
    pose = viz.Pose3D(env)
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
    player = viz.Replayer(ctx, pose, 2 * CTRL_DT)     # (one cell's hand wandering over its object, replayed in the 3D view)
    rec, last_replay = [], -1e9

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
            if int(env.steps[0]) % 2 == 0:
                rec.append(pose.snap(0))
            dn = env.steps >= 120
            idx = dn.nonzero().flatten()
            if len(idx):
                if bool(dn[0]):
                    if len(rec) > 10 and time.time() - last_replay > 12:
                        last_replay = time.time()
                        player.play(rec, "Learning to see: the hand moves over a %s while the camera looks" % env.info[int(env.obj[0])]["label"].lower(), camera_png(env, 0))
                    rec = []
                env.reset(idx)
                sim.refresh()
        img, objpix = env.perceive()
        _, _, c, e, y = env.truth()
        s = objpix >= 6
        ok_ = torch.isfinite(img).flatten(1).all(1) & torch.isfinite(c).all(1) & torch.isfinite(e).all(1) & torch.isfinite(y).all(1)
        img, c, e, y, s = torch.nan_to_num(img), torch.nan_to_num(c), torch.nan_to_num(e), torch.nan_to_num(y), s & ok_
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
                ctx.telemetry("camera_picture", {"image": camera_png(env, 0)})
                ctx.telemetry("events", {"text": "Unseen object: %s, its centre found within %.1f cm" % (env.info[int(env.obj[0])]["label"], float((pred[0][0] - tc[0]).norm()) * 100)})
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
    player.stop()
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
    replay_every = float(p.get("replay_every") or 0)     # (at least this many seconds between replays; 0: one after the other)
    ch_bot, ch_train, ch_ctl = p.get("robot_channel") or "robot", p.get("channel") or "train", p.get("control_channel") or "control"
    torch.manual_seed(seed)
    lib, sim, label, env = build(ctx, n, stage, seed=seed)
    n, dev = sim.n, sim.tdev
    env.progress = ctx.progress           # (long steps before the first update, such as learning from demonstrations, report as they go)
    see, sp = load_see(ctx, dev)
    if see is not None:
        env.see = see
        ctx.log("It sees through the camera network (%s)." % sp["name"])
    else:
        ctx.log("No camera network yet (Learn to see): it learns with the object's true box, blurred as a camera would. Run Learn to see first for the real thing.")
    algo = PPO(torch, dev, n, OBS_DIM, PRIV_DIM, ACT_DIM, horizon=32, lr=3e-4, entropy=0.002, schedule="adaptive")
    steps_done, warm = 0, 0
    order = {s: k for k, s in enumerate(STAGES)}
    task = getattr(ctx, "task", None)
    n_tasks = int(getattr(task, "count", 1) or 1)
    t_idx = int(getattr(task, "index", 0) or 0)
    rt = int(float(p.get("round_task") or 0))         # (a round made of several nodes of this block, each its own worker)
    if n_tasks <= 1 and rt > 0:
        n_tasks, t_idx = max(2, rt), rt - 1
    lead = t_idx == 0                                  # (the first task drives the 3D view and the learning curve)
    ch_train = ch_train if lead else "%s_%d" % (ch_train, t_idx)
    found = [] if str(p.get("training") or "").lower().startswith("start") else find_files(ctx, "policy", "policy_checkpoint.pt", FORMAT)
    if n_tasks > 1:        # every task of a parallel round starts from the same policy: the newest given as input
        found = [f for f in found if f["where"] == "dataset"]
    found = pick_policy(ctx, found, stage)
    env.policy_stage = found[0]["stage"] if found else None
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
    ctx.log("Learning to %s on %s: %d cells at once, for up to %.0f minutes%s." % (stage, label, n, minutes,
            (" (task %d of this round: its policy is averaged with the others' when they are done)" % (t_idx + 1)) if n_tasks > 1 else ""))
    steps_start = steps_done
    pose = viz.Pose3D(env)
    live = None
    want_live = str(p.get("live_cell") or "No").lower().startswith("y")
    if lead:
        ctx.telemetry("scene", viz.cell_objects(lib[0]))
    if lead and want_live:
        try:
            live = LiveCell(ctx, stage, seed, CTRL_DT)
            live.update(algo, 0)
        except Exception as e:      # noqa: BLE001
            ctx.log("No live cell (%s): the 3D view replays episodes of the learning cells instead." % e)
            live = None
    if live is None and lead:
        player = viz.Replayer(ctx, pose, 2 * CTRL_DT)
    else:
        player = viz.Replayer.__new__(viz.Replayer)          # (a silent one: only the first task, without a live cell, shows replays)
        player.play = lambda *a, **k: None
        player.waiting = lambda: 10 ** 6
        player.stop = lambda: None
    env.reset_all()
    env.perceive()
    a_obs, a_priv = env.obs()
    curve = open(ctx.output("curve", "%s.csv" % stage), "w", newline="", encoding="utf-8")
    cw = csv.writer(curve)
    cw.writerow(["minutes", "iteration", "steps", "success_pct", "fail_pct", "episode_s", "reward", "steps_per_s"])
    hist = collections.deque(maxlen=max(300, min(3000, n)))
    per_obj = collections.defaultdict(lambda: collections.deque(maxlen=60))
    t0, last_tel, last_replay, it, best, stop = time.time(), 0.0, -1e9, 0, -1.0, "time"
    last_scene = 0.0
    out_path = os.path.join(os.getcwd(), "policy_checkpoint.pt")
    # the 3D view replays whole episodes: many cells are recorded at once (with thousands of cells each one advances
    # slowly: a few seconds of its time take a minute or so), and the episodes they finish wait in a queue to be
    # shown one after the other, so that the view keeps playing between the learning's updates
    show = list(range(min(SHOW_CELLS, n))) if live is None else []
    recs = {i: [] for i in show}
    rec_objs = {i: object_item(env, i) for i in show}
    rec_ep = 0
    dist_ = collections.deque(maxlen=3000)          # how far the grasp point ended from where it should be (cm)

    def score():
        return 100.0 * sum(h[0] for h in hist) / len(hist) if len(hist) >= 50 else 0.0

    def save(note):
        st = algo.state()
        st.update({"format": FORMAT, "stage": stage, "cell": cell_signature(), "obs_dim": OBS_DIM, "priv_dim": PRIV_DIM, "act_dim": ACT_DIM, "score": score(), "note": note,
                   "steps": steps_done, "round_steps": steps_done - steps_start, "device": label, "minutes": (time.time() - t0) / 60, "ctrl_dt": CTRL_DT,
                   "episodes": len(hist), "arm": "xarm6-in-physics", "round_task": (t_idx + 1) if n_tasks > 1 else 0,
                   "per_object": {k: round(100 * sum(v) / max(1, len(v))) for k, v in per_obj.items()}})
        torch.save(st, out_path)

    last_stats = {}

    def tel_row(it_, steps_):
        """the learning's numbers so far (the chart and gauge rows): sent after each update, and every few seconds
        while the cells gather the next batch, from the episodes that have ended (an update comes every minute or two)"""
        k_ = max(1, len(hist))
        row = {"iteration": round(it_, 2), "minutes": round((time.time() - t0) / 60, 2), "steps_m": round(steps_ / 1e6, 2), "success": round(score(), 1),
               "failed": round(100.0 * sum(h[1] for h in hist) / k_, 1), "episode_s": round(sum(h[2] for h in hist) / k_, 1),
               "reward": round(sum(h[3] for h in hist) / k_, 1), "end_dist_cm": round(sum(dist_) / max(1, len(dist_)), 1),
               "restarted": env.n_bad, "episodes": len(hist)}
        row.update(last_stats)
        # the success of the episodes that started with the hand at rest (it had to find the object and reach
        # it first) apart from those that started with the hand already over it (a share of the grasp and
        # place episodes, to learn the closing early): the first is what the robot can really do
        for key, flag in (("success_from_rest", False), ("success_hand_over", True)):
            hs = [h[0] for h in hist if h[4] == flag]
            if len(hs) >= 20:
                row[key] = round(100.0 * sum(hs) / len(hs), 1)
        if len(hist) < 50:          # (too few episodes ended yet for a rate: the short ones, the failures, end first)
            row.pop("success", None)
            row.pop("failed", None)
        return row

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
                if lead and show:
                    st_ = env.steps[:len(show)]
                    par = (st_ % 2 == 0).tolist()
                    sn = pose.snap_many(show)
                    pics = camera_pics(env, len(show)) if env.k % PIC_EVERY == 0 else None
                    for i in show:
                        if par[i] or pics is not None:
                            recs[i].append((sn[i], pics[i]) if pics is not None else sn[i])
                if bool(done.any()):
                    idx = done.nonzero(as_tuple=False).flatten()
                    sc_, fl_, ln_ = info["success"][idx].tolist(), info["fail"][idx].tolist(), env.steps[idx].tolist()
                    dist_.extend((info["d"][idx] * 100).tolist())
                    rt_, ob_, as_ = env.ep_ret[idx].tolist(), env.obj[idx].tolist(), env.assisted[idx].tolist()
                    for a_, b_, c_, d_, o_, h_ in zip(sc_, fl_, ln_, rt_, ob_, as_):
                        hist.append((1.0 if a_ else 0.0, 1.0 if b_ else 0.0, c_ * CTRL_DT, d_, bool(h_)))
                        per_obj[env.info[o_]["key"]].append(1.0 if a_ else 0.0)
                    ended = done[:len(show)].nonzero(as_tuple=False).flatten().tolist()
                    for i in ended:
                        if lead and len(recs[i]) > 10 and player.waiting() < REPLAY_QUEUE and time.time() - last_replay > replay_every:
                            last_replay = time.time()
                            rec_ep += 1
                            ok, fl = bool(info["success"][i]), bool(info["fail"][i])
                            lab = "Episode %d (%s, cell %d): %s, %s after %.1f s" % (rec_ep, STAGE_NAME[stage], i + 1, rec_objs[i]["label"], results_text(stage, ok, fl), float(env.steps[i]) * CTRL_DT)
                            player.play(recs[i], lab)        # (the camera's pictures go with the frames they were taken at)
                        recs[i] = []
                    env.reset(idx)
                    sim.refresh()
                    for i in ended:
                        rec_objs[i] = object_item(env, i)
                    o2, p2 = env.obs()
                a_obs, a_priv = o2, p2
                if time.time() - last_tel > 10.0 and hist:     # (between the updates too: the numbers move as episodes end)
                    last_tel = time.time()
                    ctx.telemetry(ch_train, tel_row(it + (t + 1) / algo.T, steps_done + (t + 1) * n))
            stats = algo.update(a_priv, critic_only=it < warm)
            it += 1
            if live is not None:
                live.update(algo, it)
            steps_done += algo.T * n
            sps = algo.T * n / max(1e-6, time.time() - tt)
            sc = score()
            if sc > best and len(hist) >= 50 and it > warm + 10:
                best = sc
                save("best so far")
            if True:                    # (after every update, with its own numbers: the spread of the actions, the step)
                last_tel = time.time()
                last_stats.update({"steps_per_s": round(sps), "std": round(stats["std"], 3), "kl": round(stats["kl"], 4)})
                row = tel_row(it, steps_done)
                ctx.telemetry(ch_train, row)
                if lead and time.time() - last_scene > 8:        # (the cell itself, for a view opened since: it shows at once)
                    last_scene = time.time()
                    ctx.telemetry("scene", pose.scene)
                cw.writerow([row["minutes"], it, steps_done, row.get("success", ""), row.get("failed", ""), row["episode_s"], row["reward"], row["steps_per_s"]])
                curve.flush()
                ctx.progress(min(99, 100 * (time.time() - t0) / (minutes * 60 + 1)), "%s: %.0f%% success%s · ends %.1f cm from the object · %.1f M steps" % (STAGE_NAME[stage], sc, (" (%.0f%% from rest)" % row["success_from_rest"]) if "success_from_rest" in row and "success_hand_over" in row else "", row["end_dist_cm"], steps_done / 1e6))
    finally:
        curve.close()
        player.stop()
        if live is not None:
            live.stop()
    if best < 0 or score() >= best or n_tasks > 1:
        save("final")
    final = torch.load(out_path, map_location="cpu", weights_only=False)
    name = "grasp_%s_%s_%02d.pt" % (stage, time.strftime("%Y%m%d_%H%M"), int(round(max(0.0, float(final.get("score") or 0)))))
    shutil.copyfile(out_path, ctx.output("policy", name))
    if sp:      # (and the camera network it saw with, so that the next stage has both wherever it runs)
        shutil.copyfile(sp["path"], ctx.output("policy", "grasp_see_%s.pt" % time.strftime("%Y%m%d_%H%M", time.localtime(sp["stamp"]))))
    po = final.get("per_object") or {}
    if po:
        ctx.log("Success per object: " + ", ".join("%s %d%%" % (S.TRAIN[k][0] if k in S.TRAIN else k, v) for k, v in sorted(po.items(), key=lambda x: -x[1])))
    if env.n_bad:
        ctx.log("%d times a cell's physics blew up (a hard contact among thousands): those episodes were dropped and the cells started again." % env.n_bad)
    ctx.log("Policy written: %s (%s success %.0f%%)." % (name, STAGE_NAME[stage], float(final.get("score") or 0)))
    ctx.log("Done (%s): %d iterations, %.0f million steps in %.1f minutes." % ("time is up" if stop == "time" else "finished on request", it, steps_done / 1e6, (time.time() - t0) / 60))
    return {"success_pct": round(float(final.get("score") or 0), 1), "stage": stage, "steps_m": round(steps_done / 1e6, 1)}


# ── a parallel round's combine task: the tasks' policies averaged ────────────────────────────────────────────────
def combine_run(ctx, stage):
    """the policies the round's tasks learned (each from the same start, on its own worker and hardware) averaged into
    one: each weighted by the experience it gathered in the round; the input scalings merged as the statistics of all
    their samples together. It is the start of the next round."""
    import torch
    pols, sees = [], []
    for port, files in ctx.inputs.items():
        for f in files:
            if str(f.name).endswith(".pt"):
                try:
                    st = torch.load(f.path, map_location="cpu", weights_only=False)
                except Exception as e:      # noqa: BLE001
                    ctx.log("Skipped %s: %s" % (f.name, e))
                    continue
                if st.get("format") == FORMAT and (stage is None or st.get("stage") == stage) and (getattr(f, "task", None) is not None or st.get("round_task")):
                    pols.append((f, st))
                elif st.get("format") == SEE_FORMAT:
                    sees.append(f)
    if not pols:
        raise RuntimeError("No policies from this round's tasks to combine (wire the Network outputs of the round's Learn nodes in).")
    if stage is None:              # (Combine a round: the stage of the policies wired in)
        order = {s_: k for k, s_ in enumerate(STAGES)}
        stage = max((st.get("stage") for f, st in pols), key=lambda s_: order.get(s_, -1))
        pols = [(f, st) for f, st in pols if st.get("stage") == stage]
    tag = lambda f, st: f.task if getattr(f, "task", None) is not None else st.get("round_task")    # noqa: E731
    w = np.array([max(1.0, float(st.get("round_steps") or 0)) for f, st in pols])
    w = w / w.sum()
    base = pols[int(np.argmax(w))][1]
    ac = {}
    for k, v in base["ac"].items():
        if torch.is_tensor(v) and v.is_floating_point():
            ac[k] = sum(float(wi) * st["ac"][k].float() for wi, (f, st) in zip(w, pols)).to(v.dtype)
        else:
            ac[k] = v

    def merge(key):
        ms = [st[key] for f, st in pols]
        cnt = [m["count"].double() if torch.is_tensor(m["count"]) else torch.tensor(float(m["count"])).double() for m in ms]
        tot = sum(cnt)
        mean = sum(c * m["mean"].double() for c, m in zip(cnt, ms)) / tot
        var = sum(c * (m["var"].double() + (m["mean"].double() - mean) ** 2) for c, m in zip(cnt, ms)) / tot
        return {"mean": mean.float(), "var": var.float(), "count": tot.float()}
    out = dict(base)
    out.pop("opt", None)          # (each task's optimizer moments belong to its own weights: the next round starts them afresh)
    out["ac"] = ac
    out["obs_norm"] = merge("obs_norm")
    out["priv_norm"] = merge("priv_norm")
    eps = [float(st.get("episodes") or 0) for f, st in pols]
    out["score"] = float(sum(e * float(st.get("score") or 0) for e, (f, st) in zip(eps, pols)) / max(1.0, sum(eps)))
    out["steps"] = int(max(float(st.get("steps") or 0) - float(st.get("round_steps") or 0) for f, st in pols) + sum(float(st.get("round_steps") or 0) for f, st in pols))
    out["round_steps"] = 0
    out["round_task"] = 0
    out["lr"] = float(np.median([float(st.get("lr") or 3e-4) for f, st in pols]))
    out["note"] = "averaged from %d workers" % len(pols)
    out["device"] = " + ".join(str(st.get("device")) for f, st in pols)
    name = "grasp_%s_%s_%02d.pt" % (stage, time.strftime("%Y%m%d_%H%M%S"), int(round(max(0.0, out["score"]))))
    torch.save(out, ctx.output("policy", name))
    for f in sees[:1]:
        shutil.copyfile(f.path, ctx.output("policy", f.name))
    for (f, st), wi in zip(pols, w):
        ctx.log("Task %s on %s: %.0f%% success, %.1f M steps this round, weight %.2f." % (tag(f, st), st.get("device"), float(st.get("score") or 0), float(st.get("round_steps") or 0) / 1e6, wi))
    ctx.log("Combined: %s (%.0f%% success across the workers, %.1f M steps in all)." % (name, out["score"], out["steps"] / 1e6))
    row = {"success": round(out["score"], 1), "steps_m": round(out["steps"] / 1e6, 2), "workers": len(pols)}
    for (f, st) in pols:
        row["task%s_steps_m" % tag(f, st)] = round(float(st.get("round_steps") or 0) / 1e6, 2)
        row["task%s_success" % tag(f, st)] = round(float(st.get("score") or 0), 1)
    ctx.telemetry("rounds", row)
    return {"success_pct": round(out["score"], 1), "stage": stage, "workers": len(pols), "steps_m": round(out["steps"] / 1e6, 1)}


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
    drop = float(p.get("drop_cm") if p.get("drop_cm") not in (None, "") else 25) / 100.0      # from how high it falls onto the belt
    lib, sim, label, env = build(ctx, 1, "place", seed=seed, test_objects=True)
    dev = sim.tdev
    env.assist = 0.0
    see, sp = load_see(ctx, dev, need=True)
    env.see = see
    pols = pick_policy(ctx, find_files(ctx, "policy", "policy_checkpoint.pt", FORMAT), "place")
    if not pols:
        raise RuntimeError("No policy yet: run Learn to reach, Learn to grasp and Learn to place first.")
    pol = pols[0]
    st = pol["st"]
    env.stage = pol["stage"]
    env.max_steps = int(60.0 / CTRL_DT)       # (the object travels the whole belt: the episode ends when it is picked or falls off)
    ac = make_ac(torch, OBS_DIM, PRIV_DIM, st["act_dim"]).to(dev)
    ac.load_state_dict(widen(torch, st["ac"], OBS_DIM, PRIV_DIM))     # (a policy from before the phalanges' sensors: they start unweighted)
    ac.eval()
    on = Norm(torch, OBS_DIM, dev)
    on.load(st["obs_norm"], dev)
    ctx.log("The test: %s policy (%s, %.0f%% in training), camera network %s, on %s." % (STAGE_NAME[pol["stage"]], pol["name"], float(st.get("score") or 0), sp["name"], label))
    objs = [o for o in env.info if o["test"]]
    if str(p.get("objects") or "").lower().startswith("all"):
        objs = env.info
    ctx.telemetry("scene", viz.cell_objects(lib[0]))
    pose = viz.Pose3D(env)
    names = ", ".join(o["label"] for o in env.info if o["test"])
    ctx.telemetry(ch_ev, {"text": "Ready. New to it: %s. Choose an object, how it lies, how it is turned and the belt's speed, "
                                  "then Put it on the belt (or let it pick)." % names})
    queue = collections.deque()
    rows = []
    out_csv = open(ctx.output("curve", "test.csv"), "w", newline="", encoding="utf-8")
    cw = csv.writer(out_csv)
    cw.writerow(["trial", "object", "seen_in_training", "pose", "yaw_deg", "belt_speed", "grasped", "placed", "result", "seconds", "max_force_N"])
    t0, trial, cyc = time.time(), 0, 0
    idle_since = time.time()
    keys = {o["key"]: k for k, o in enumerate(env.info)}
    labels = {o["label"].lower(): k for k, o in enumerate(env.info)}
    pick = {"object": None, "pose": 0, "yaw": 30.0, "speed": 0.10}

    def obj_index(v):
        v = str(v or "").strip()
        k = keys.get(v)
        if k is None:
            k = labels.get(v.lower().replace(" (new)", "").replace(" (seen)", ""))
        return k

    def read_controls():
        for m in ctx.messages(ch_ctl):
            d = m.get("data") if isinstance(m, dict) else None
            if not isinstance(d, dict):
                continue
            c = str(d.get("control", "")).strip().lower()
            v = d.get("value")
            if c in ("finish now", "finish"):
                return "finish"
            if c == "object":
                k = obj_index(v)
                if k is None:
                    ctx.telemetry(ch_ev, {"text": "Unknown object: %s" % v})
                else:
                    pick["object"] = k
            elif c == "pose":
                pick["pose"] = POSES.get(str(v or "any").lower(), 2)
            elif c in ("turned", "yaw"):
                pick["yaw"] = float(v)
            elif c in ("belt speed", "speed"):
                pick["speed"] = float(v) / 100.0 if float(v) > 1 else float(v)
            elif c in ("put it on the belt", "drop", "pick", "test"):
                w = v if isinstance(v, dict) else {}
                k = obj_index(w.get("object")) if w.get("object") else pick["object"]
                if k is None:
                    k = keys[objs[cyc % len(objs)]["key"]]
                queue.append({"obj": k, "pose": POSES.get(str(w.get("pose")).lower(), pick["pose"]) if w.get("pose") else pick["pose"],
                              "yaw": float(w["yaw"]) if w.get("yaw") not in (None, "") else pick["yaw"],
                              "speed": float(w["speed"]) if w.get("speed") not in (None, "") else pick["speed"]})
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
        env.reset(idx, obj=torch.tensor([req["obj"]], device=dev), mode=req["pose"], yaw=yaw, speed=req["speed"], drop=drop, x0=S.TEST_SPAWN_X)
        sim.refresh()
        o = env.info[req["obj"]]
        pose_txt = ["upright", "on its side", "tumbled"][req["pose"]]
        # (the trial's live items, bought before it is played: a pose every 3 steps, every 6 when the run's items run
        # short, none (the trial still runs, the view waits) when they have run out)
        every = 0
        for e in (3, 6):
            if viz.afford(ctx, int((env.max_steps + 105) / e * 2.4 + env.max_steps / 12) + 8):
                every = e
                break
        if every:
            pose.cut(ctx)       # (the new object appears where it is dropped: it does not glide there in the view)
        else:
            ctx.telemetry(ch_ev, {"text": "(The 3D view sits this trial out: the run's live items are spread over its minutes.)"})
        ctx.telemetry(ch_ev, {"text": "Trial %d: %s%s, dropped %s onto the belt at %.0f cm/s" % (trial, o["label"], " (new to it)" if o["test"] else " (seen in training)", pose_txt, float(env.belt_v[0]) * 100)})
        # it falls, bounces and settles on the moving belt (the hand waits); then the robot's turn begins
        wall0 = time.time()
        if drop:
            still = 0
            for k in range(60):
                env.step(torch.zeros(1, ACT_DIM, device=dev))
                if every and k % every == 0:
                    pose.send(ctx, pose.snap(0))
                ov = sim.qvel[0, env.do_:env.do_ + 6].float()
                rel = float((ov[0] + env.belt_v[0]).abs() + ov[1:3].abs().sum() + 0.05 * ov[3:].abs().sum())
                still = still + 1 if rel < 0.03 else 0
                if pace > 0:
                    lag = wall0 + (k + 1) * CTRL_DT / pace - time.time()
                    if lag > 0:
                        time.sleep(lag)
                if still >= 5 and k >= 10:
                    break
            env.restart_clock(idx)
            env.z_rest[0] = sim.xpos[0, env.ob, 2].float()
            ctx.telemetry(ch_ev, {"text": "Landed at %.0f cm along the belt, %.0f cm from its middle, after %.1f s" % (float(sim.xpos[0, env.ob, 0]) * 100, (float(sim.xpos[0, env.ob, 1]) - S.BELT_CY) * 100, (k + 1) * CTRL_DT)})
        # the belt brings it along while the hand waits at rest: the robot's turn begins where the objects it learned
        # on were put on the belt (the policy is never asked to act on what it never saw in training)
        wall0, kw = time.time(), 0
        while float(sim.xpos[0, env.ob, 0]) > S.SPAWN_X[1] - 0.02 and float(sim.xpos[0, env.ob, 2]) > -0.05 and kw < 1500:
            env.step(torch.zeros(1, ACT_DIM, device=dev))
            kw += 1
            if every and kw % every == 0:
                pose.send(ctx, pose.snap(0))
            if pace > 0:
                lag = wall0 + kw * CTRL_DT / pace - time.time()
                if lag > 0:
                    time.sleep(lag)
        env.restart_clock(idx)
        env.z_rest[0] = sim.xpos[0, env.ob, 2].float()
        env.perceive()
        wall0 = time.time()
        grasped, maxf, result = False, 0.0, "ran out of time"
        while True:
            obs, _ = env.obs()
            with torch.no_grad():       # (as it acted while it learned: its moves with their spread, which is what closes its fingers)
                mu = ac.actor(on(obs))
                a = mu + ac.log_std.exp() * torch.randn_like(mu)
            r, done, info = env.step(a)
            grasped = grasped or bool(env.lifted_ever[0]) or bool(env.hold_cnt[0] >= 25) or float(info["lift"][0]) > 0.05
            maxf = max(maxf, float(env.touches()[0].max()))
            ts = float(env.steps[0]) * CTRL_DT
            if every and int(env.steps[0]) % every == 0:
                pose.send(ctx, pose.snap(0))
            if every and int(env.steps[0]) % 12 == 0:
                ctx.telemetry("camera_picture", {"image": camera_png(env, 0)})
            if pace > 0:
                lag = wall0 + ts / pace - time.time()
                if lag > 0:
                    time.sleep(lag)
            if bool(done[0]):
                if bool(info["success"][0]):
                    result = "in the bin" if env.stage == "place" else "grasped and held"
                elif bool(info["fail"][0]):
                    result = "dropped" if grasped else "missed it: it went off the end of the belt"
                    for k2 in range(45):       # (it falls to the floor, in the view)
                        env.step(torch.zeros(1, ACT_DIM, device=dev))
                        if every and k2 % every == 0:
                            pose.send(ctx, pose.snap(0))
                        if pace > 0:
                            time.sleep(CTRL_DT / pace)
                break
            if read_controls() == "finish":
                stop = "asked"
                break
        placed = result == "in the bin"
        rows.append((o["key"], grasped, placed))
        viz.settle(ctx)
        ctx.telemetry(ch_ev, {"text": "Trial %d, %s: %s%s (%.1f s, grip up to %.1f N)" % (trial, o["label"], result, "" if placed or not grasped else ", it had grasped it", ts, maxf)})
        ctx.telemetry("test", {"trials": len(rows), "grasped": sum(r[1] for r in rows), "placed": sum(r[2] for r in rows),
                               "placed_pct": round(100.0 * sum(r[2] for r in rows) / len(rows), 1)})
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


def models_run(ctx):
    """the 3D view's two robots (the xArm with the hand, the objects), for the project's assets"""
    import mujoco
    import cell
    lib = cell.compile_library(mujoco)
    glb = [f.path for f in (ctx.inputs.get("conveyor") or []) + (ctx.inputs.get("policy") or []) if str(f.name).lower().endswith(".glb")]
    if not glb:
        ctx.log("No conveyor 3D file wired in (Conveyor model input): the belt is drawn with boxes.")
    viz.write_models(ctx, "models", lib[0], lib[6], mesh_cache(), glb[0] if glb else None)
    return {"models": [viz.ARM, viz.OBJS], "conveyor_from_file": bool(glb)}


def mesh_cache():
    """the folder where the xArm's meshes are kept on this machine, from one job to the next"""
    here = os.path.abspath(os.getcwd())
    jobs = os.path.dirname(os.path.dirname(here))
    base = os.path.dirname(jobs) if os.path.basename(jobs).lower() == "jobs" else os.path.dirname(here)
    return os.path.join(base, "cache", "xarm6")


def run(ctx):
    what = str(ctx.params.get("mode") or "Learn to reach").lower()
    viz.budget_telemetry(ctx, 60.0 * float(ctx.params.get("minutes") or 60))      # (EsaSim keeps 20 000 live items a run)
    S.ARM_MESHES = viz.camera_arm_meshes(mesh_cache(), ctx.log)     # (the camera sees the xArm as it is)
    bd = str(ctx.params.get("belt_drawing") or "flat").lower()
    viz.CONVEYOR["mode"] = "boxes" if bd.startswith("box") else "textured" if ("material" in bd or "textur" in bd) else "flat"
    glb = [f for f in (ctx.inputs.get("conveyor") or []) if str(f.name).lower().endswith(".glb")]
    if viz.CONVEYOR["mode"] == "textured" and glb and "model" not in what:
        try:              # (where the 3D view is to draw the conveyor's own file: fitted to the belt as the models are)
            viz.CONVEYOR["asset"] = str(getattr(glb[0], "rel", None) or glb[0].name).replace("\\", "/")
            viz.conveyor_mesh(glb[0].path, lambda *a: None)
        except Exception as e:      # noqa: BLE001
            ctx.log("The conveyor's 3D file could not be read (%s): the belt is drawn with boxes." % e)
    viz.CONVEYOR_DRAWN["value"] = viz.CONVEYOR["mode"] == "flat" or bool(viz.CONVEYOR["fit"])
    if "model" in what:
        return models_run(ctx)
    if "combine" in what:
        return combine_run(ctx, None)
    if "see" in what:
        return see_run(ctx)
    if "test" in what:
        return test_run(ctx)
    stage = next((s for s in STAGES if s in what), "reach")
    if getattr(getattr(ctx, "task", None), "kind", "run") == "combine":
        return combine_run(ctx, stage)
    return learn_run(ctx, stage)
'''


def _load():
    for name in ['scene', 'cell', 'sims', 'grasp_env', 'viz', 'ppo_part', 'train']:
        if name in _sys.modules and getattr(_sys.modules[name], "_grasp_block", False):
            continue
        mod = _types.ModuleType(name)
        mod.__file__ = "<grasp:%s>" % name
        mod._grasp_block = True
        _sys.modules[name] = mod
        exec(compile(_SOURCES[name], mod.__file__, "exec"), mod.__dict__)



def _overrides(ctx):
    """a newer version of one part of this block, given as grasp_override_<part>.py among the Policies input (in the
    grasp-networks dataset), replaces that part for the run: a change tried out without saving the block"""
    import os
    for f in (ctx.inputs.get("policy") or []):
        nm = os.path.basename(str(f.name))
        if nm.startswith("grasp_override_") and nm.endswith(".py") and nm[15:-3] in _SOURCES:
            _SOURCES[nm[15:-3]] = open(f.path, encoding="utf-8").read()
            ctx.log("Using %s in place of the block's own %s." % (nm, nm[15:-3]))


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
    _overrides(ctx)
    _load()
    return _sys.modules["train"].run(ctx)

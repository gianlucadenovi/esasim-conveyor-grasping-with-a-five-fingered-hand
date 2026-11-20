<!-- esasim: cover -->
![Conveyor grasping with a five-fingered hand](assets/cover.svg)
<!-- /esasim: cover -->

<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures (the cell from the Whole cell view works well), then: ![Conveyor grasping](pictures/banner.png) -->

# Conveyor grasping with a five-fingered hand

**An xArm with a five-fingered hand learns to see the objects coming on a conveyor belt, reach them, grasp them with simulated contact forces and put them in a bin. Then you test it on objects it has never seen, in the pose you choose.**

The hand senses through a camera beside the robot and nine touch sensors (the five fingertips and four zones of the palm). Every force between the fingers, the palm and the object is simulated by MuJoCo. The arm positions the hand kinematically, since the objects weigh next to nothing compared with the arm, but the grasp itself is physical.

![type](https://img.shields.io/badge/type-Robotics-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![physics](https://img.shields.io/badge/physics-MuJoCo%20%7C%20MuJoCo%20Warp-orange) ![learning](https://img.shields.io/badge/learning-PPO%20%2B%20CNN-178c8c) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

Objects of different shapes travel along a belt, upright, on their side or tumbled, at any orientation, at 3 to 8 cm/s. Next to the belt are an xArm 6 carrying a five-fingered hand, a camera on a pole and a bin.

Learning happens in stages, and each stage carries on from the one before:

1. **See**: a small convolutional network learns, from the camera's depth and colour, which pixels belong to the object and from them its box: where it is, how big it is and which way its long side lies.
2. **Reach**: the hand learns to meet the object on the moving belt, palm down, turned across its long side, keeping pace with the belt.
3. **Grasp**: it learns to close the fingers and the thumb around the object, lift it 8 cm and hold it for a second while the object is pushed about.
4. **Place**: it learns to carry the object to the bin and let it go there.

Thousands of cells learn at once on one GPU (MuJoCo Warp, with its batch renderer for the cameras). The objects come from the 18 training shapes. The 12 test shapes (a banana, a stepped cone, a drill, a pear, a duck and others) are never used in training.

All models are built by the project itself from simple shapes (boxes, cylinders, spheres, capsules, ellipsoids): the cell, the hand and the objects. Better-looking models can replace them later without changing the learning.

## Flows

| Flow | What it does |
|---|---|
| [**1 · Learn to see**](https://esacode.com/app/projects/25/flows/84) | Trains the camera network (about 25 minutes on an RTX 3070). |
| [**2 · Learn to reach**](https://esacode.com/app/projects/25/flows/85) | PPO on 2048 cells, 60 minutes. |
| [**3 · Learn to grasp**](https://esacode.com/app/projects/25/flows/87) | Carries on from reach, 90 minutes. |
| [**4 · Learn to place**](https://esacode.com/app/projects/25/flows/88) | Carries on from grasp, 90 minutes. |
| [**5 · Test with new objects**](https://esacode.com/app/projects/25/flows/89) | One cell, in real time, driven from the View. |

Every flow has the **Conveyor grasping (xArm + 5-finger hand)** block in the matching mode, reading and writing its networks in the **grasp-networks** dataset. Each flow also has the **xArm model** block (UFACTORY's own meshes, BSD-3), the **Conveyor grasping view** and charts of the learning curve.

## The View

- **The cell** in 3D: the belt (its surface runs), the bin, the camera on its pole, the object, the hand with every finger joint, and the xArm. The arm is posed by inverse kinematics on the hand's pose at every frame.
- The **camera box**, in yellow: where the camera network thinks the object is.
- **Touch**: the force on each fingertip and palm zone, in newtons.
- **What the camera sees**: the picture the network gets, with its box (seeing and testing).
- During training, one episode is replayed every 20 seconds, with its result.
- **Test with a new object**: choose the object (new, or one it trained on), how it lies (upright, on its side, tumbled), how it is turned and the belt speed, then press **Put it on the belt**. The trial plays live, and the count of trials, grasps and objects in the bin goes up.
- **Camera view**, **Close up**, **Whole cell**, and buttons that show or hide the camera box and the xArm.

## What is computed

| Part | How |
|---|---|
| **Physics** | MuJoCo 3 (CPU) or MuJoCo Warp (GPU): 2 ms steps, 40 ms control steps; elliptic friction cones; finger joints driven by position servos with limited force |
| **Hand** | 5 fingers, 15 joints. The policy sets each finger's curl (shared over its three joints) and the thumb's swing under the palm. The hand's pose (x, y, z, yaw, pitch, roll) comes from six stiff servos: kinematic positioning |
| **Touch** | 9 MuJoCo touch sensors: thumb, index, middle, ring and little fingertips, and four palm zones |
| **Camera** | 96×72 depth, colour and segmentation, 6 frames a second, turned into a point map |
| **Camera network** | encoder-decoder CNN: an object mask, then mask-weighted point statistics give the box (centre, half sizes, yaw) |
| **Policy** | PPO with an asymmetric critic. The policy sees only what the robot could sense: its hand, fingers, touch, belt speed and the camera's box. The critic also sees the true object state. Network 512-256-128 |
| **Rewards** | reach: distance to the grasp point and progress towards it, palm level, turned across the object. Grasp: pinch and wrap contacts, lift, held while pushed. Place: carried to the bin, released, resting in it. Knocking the object over or crushing it costs |

## Where it runs

- **NVIDIA GPU** (recommended): MuJoCo Warp, 1024 to 4096 cells at once, 10 to 30 thousand steps a second on an RTX 3070.
- **CPU**: MuJoCo, one simulation per thread and at most 256 cells. It works but takes far longer.
- A worker whose PyTorch is built for the CPU only gets a CUDA build of it once, in its data folder, for this block only.

## What you need

- A worker with an NVIDIA GPU and its driver. The packages `torch`, `mujoco`, `mujoco-warp`, `warp-lang` and `numpy` are installed by the worker.
- No account or key.

## How to run it

1. Run **1 · Learn to see**, then **2 · Learn to reach**, **3 · Learn to grasp** and **4 · Learn to place**, in order. Each stage finds the networks of the one before, in the **grasp-networks** dataset or left on the machine.
2. Run **5 · Test with new objects**, switch to **View**, choose an object and its pose, and press **Put it on the belt**. Left alone, it picks the new objects itself, one after the other.
3. To learn longer, run a stage again: **Training: Continue** carries on from the newest policy. Use **Start over** to begin again.

## Settings worth knowing

On the **Conveyor grasping** block: **What to do** (the mode), **Run on** (auto, NVIDIA GPU or CPU), **Cells learning at once**, **Minutes**, **Training** (Continue or Start over), **Replay every**, and for the test: **pace** (1 = real time), **pick objects when idle**, **Test objects** (new only or all).

## What you get

- `grasp_see_<date>.pt`: the camera network.
- `grasp_<stage>_<date>_<success>.pt`: the policies, each saved together with the camera network it used.
- `see.csv`, `reach.csv`, `grasp.csv`, `place.csv`: the learning curves. `test.csv`: one row per test trial (object, pose, yaw, belt speed, grasped, in the bin, maximum grip force).

## Data and licences

The xArm's kinematics and meshes come from [xarm_ros](https://github.com/xArm-Developer/xarm_ros) by UFACTORY (BSD 3-Clause). The cell, the hand and the objects are made by the project. MuJoCo and MuJoCo Warp are Apache 2.0; PyTorch is BSD.

## Notes

- The camera network places objects it never saw within about 0.6 cm and their long axis within about 4°.
- In the 3D view the xArm is drawn from its meshes when the xArm model's file reaches the browser. Otherwise it is drawn from its kinematics with simple shapes, and a note says so.
- In training, reach success is counted with the policy's exploration noise. Keeping pace with the belt within 8 cm/s is hard under that noise, so the reach success figure understates what the policy does in the test, which runs without noise. The distance at which the hand ends is the better guide.
- A cell whose physics blows up (rare, among thousands) is restarted, and the log counts how often.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to see](blueprints/1-learn-to-see.json) · 9 blocks · version 8
- [2 · Learn to reach](blueprints/2-learn-to-reach.json) · 11 blocks · version 5
- [3 · Learn to grasp](blueprints/3-learn-to-grasp.json) · 11 blocks · version 5
- [4 · Learn to place](blueprints/4-learn-to-place.json) · 11 blocks · version 4
- [5 · Test with new objects](blueprints/5-test-with-new-objects.json) · 10 blocks · version 7
- [Maintenance · Rebuild the 3D models (after the hand, objects or conveyor change)](blueprints/maintenance-rebuild-the-3d-models-after-the-hand-objects-or-.json) · 3 blocks · version 13

### Workspace blocks it uses

- [Conveyor grasping (xArm + 5-finger hand)](blocks/ws-conveyor-grasping-xarm-5-finger-hand) · version 30
- [Hand pressure map](blocks/ws-hand-pressure-map) · version 1

### Assets

The files the project's views draw with (3D models, textures, sounds), in [assets/](assets/):

- [models/belt_objects.robot.json.gz](assets/models/belt_objects.robot.json.gz) · 98 KB
- [models/simple_rubber_conveyor.glb](assets/models/simple_rubber_conveyor.glb) · 2.9 MB
- [models/xarm_hand.robot.json.gz](assets/models/xarm_hand.robot.json.gz) · 238 KB
- [textures/esacode-logo.png](assets/textures/esacode-logo.png) · 25 KB

The data the blueprints read (grasp-networks) stays in EsaSim.

Keys pasted in blocks (secret settings) are not saved here, nor are the team and the results of runs.

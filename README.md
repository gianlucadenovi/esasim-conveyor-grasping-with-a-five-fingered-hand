<!-- esasim: cover -->
![Conveyor grasping with a five-fingered hand](assets/cover.jpg)
<!-- /esasim: cover -->

<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures (the cell from the 3D view works well), then: ![Conveyor grasping](pictures/banner.png) -->

# Conveyor grasping with a five-fingered hand

**An xArm with a five-fingered hand learns to see the objects coming on a conveyor belt, pick them up with simulated contact forces and put them in a bin. Then you test it on objects it has never seen, in the pose you choose.**

The robot senses through a camera on a pole beside the belt and 19 touch sensors (each digit's tip, middle and first phalanx, and four zones of the palm). The xArm 6 and its left hand are simulated bodies in MuJoCo: the arm's six joints are servo-driven through inverse kinematics, and every force between the fingers, the palm, the object, the belt and the bin is computed.

![type](https://img.shields.io/badge/type-Robotics-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-NVIDIA%20%7C%20CPU-6f42c1) ![physics](https://img.shields.io/badge/physics-MuJoCo%20%7C%20MuJoCo%20Warp-orange) ![learning](https://img.shields.io/badge/learning-Imitation%20%2B%20PPO%20%2B%20CNN-178c8c) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

</div>

## Overview

Objects of 18 training shapes are dropped onto a belt running at 5 to 10 cm/s: upright, on their side or tumbled, at any orientation. Next to the belt are the xArm 6 with the hand, a security camera on an aluminium pole and a plastic bin.

1. **See**: a small convolutional network learns, from the camera's depth and colour, which pixels belong to the object and from them its box: where it is, how big, which way its long side lies.
2. **Place** (the main stage): the policy first learns from a scripted pick-and-place controller that uses only what the robot senses (its hand, its touch sensors and the camera's box): it imitates it, then drives itself for four rounds while the controller corrects it (DAgger). Then PPO fine-tunes it on the real reward (object in the bin), in small steps that keep it close to what it imitated.
3. **Test**: one cell in real time, on 12 shapes it never trained on (a banana, a stepped cone, a drill, a pear, a duck and others), driven by the learned policy or by the scripted controller.

**Reach** and **Grasp** are also available as stages of their own; they are not needed before Place.

The simulation's bodies (the hand's collision shapes, the objects, the belt, the bin) are built by the project from simple shapes. What you see in the 3D view is drawn from detailed 3D files: your rigged xArm with the left hand, the conveyor and the security camera (see the credits below).

## Flows

| Flow | What it does |
|---|---|
| [**1 · Learn to see**](https://esacode.com/app/projects/25/flows/84) | Trains the camera network (about 25 minutes on an RTX 3070). |
| [**4 · Learn to place**](https://esacode.com/app/projects/25/flows/88) | Learns from demonstrations, then PPO: the stage that does the whole task. 1024 cells, about 90 minutes. |
| [**5 · Test with new objects**](https://esacode.com/app/projects/25/flows/89) | One cell, in real time, driven from the View. |
| [2 · Learn to reach](https://esacode.com/app/projects/25/flows/85) · [3 · Learn to grasp](https://esacode.com/app/projects/25/flows/87) | Optional stages on their own. |
| [Maintenance · Rebuild the 3D models](https://esacode.com/app/projects/25/flows/86) | Makes the 3D view's models, textures, sounds and scene from the 3D files in the assets. Run it after changing the robot, the conveyor or the camera file, then copy its output into the assets' `models` folder. |

Every flow has the **Conveyor grasping (xArm + 5-finger hand)** block in the matching mode, reading and writing its networks in the **grasp-networks** dataset.

## The View

- **The cell** in 3D (the 3D scene view, from `models/cell.scene.json`):
  - the xArm with its left hand, drawn from your rigged model and shaded smooth, every finger joint moving;
  - the conveyor in glossy blue paint, worn at its edges and corners, with grime in its inner corners and shade in its channels; metal corner plates and feet, a black motor; the rubber belt's texture, gloss and relief scroll as the belt runs;
  - the security camera on a 40 × 40 mm aluminium extrusion with a steel base, its head turned to look exactly where the simulated camera looks;
  - a dark polished concrete floor under a black sky, the sun the only light, with soft bloom on the highlights.
- **Sounds** (🔊 / 🔇 at the top right of the view; browsers play sound only after a click): the belt's motor hum, the arm's actuators as it moves, the fingertips brushing the moving belt, and a thud when an object lands on the belt or in the bin.
- **What the camera sees**: the camera's picture, drawn three times finer than the network's own 96×72. Over it, in **yellow**, the object's box as the camera network sees it, in **orange** its long axis, and in **cyan** the hand's grasping axis (from the thumb's tip to between the index and middle fingertips). In a good grasp the cyan line crosses the orange one.
- **Touch**: the force on each digit and palm zone, in newtons, and the hand's pressure map.
- During training, episodes of the learning cells are replayed with their result.
- **Test with a new object**: choose the object, how it lies, how it is turned and the belt speed (5 to 10 cm/s, the range it trained on), then press **Put it on the belt**. The object is dropped 25 cm in from the start of the belt; the hand waits at rest until the object reaches the stretch where training objects start, then picks it.

## The grasp

The hand grasps across the object, not along it: the line along which the thumb closes against the fingers lies across the object's long axis, and the palm stays level.

- The hand is turned 40° beyond square to the object's long axis (the thumb-to-fingers line is not the hand's own x axis), and the grasp point sits towards the thumb's side, further for wide objects.
- The grasp (the hand's turn, height and offset) is chosen from the camera's clear view, before the hand comes down. As the fingers come round the object they hide part of it, so while they are down the cell keeps the object's height, size and axis as last seen clearly and only goes on following where it is.
- The hand follows the object where the camera sees it. It closes even when the belt or the object stops it short of the grasp height, and gives up a descent the arm cannot complete and plans it again from above.
- In the first moments of the lift the hand keeps moving with the belt, so that the belt does not drag the object's underside out of the fingers.
- It feels the object: if fewer than two of its contacts (digits or palm) press for a third of a second while it lifts or carries, it opens and tries again, a little deeper. (Thin objects, a rod or a disc, are often held between the fingers alone.)

## How the policy learns

- **It knows where it is in the task.** Besides what it senses, the policy gets the task's phase (over the object, going down, closing, lifting, carrying, letting go), followed from its sensors as the scripted controller follows it, the time spent in it and the grasp chosen from above. Without it, the policy could not tell, for instance, whether it was going down to grasp or had just let go.
- **Imitation**: the scripted controller drives 1024 cells while its actions are recorded; the policy is fitted to them; then for four rounds the policy drives (with a little noise) and the controller says what it would have done.
- **PPO** then fine-tunes it in small, capped steps, with the demonstrations still pulling it towards the controller and the input scaling kept as it was during imitation (letting PPO's own episodes rescale the inputs undid what the policy had imitated within a few updates).
- **The policy kept** is the best one checked on every cell (as it acts, with its small exploration), right after imitation and every 30 updates; a run never ends with a worse one than it started with.
- The log says how each batch of demonstrations ended: in the bin, dropped, or never lifted (and why: stuck going down, slipped while lifting, let go of an object it still held), and the controller's success per object.

## What is computed

| Part | How |
|---|---|
| **Physics** | MuJoCo 3 (CPU) or MuJoCo Warp (GPU): 2 ms steps, 40 ms control steps; elliptic friction cones |
| **Arm** | xArm 6 kinematics as your rigged model has them; position servos with the motors' torques, gravity compensated; the policy says where the hand should be (x, y, z, yaw, pitch, roll) and inverse kinematics sets the joints |
| **Hand** | a left hand: 5 fingers, 15 joints, position servos with limited torque. The policy sets each finger's curl and the thumb's swing under the palm |
| **Touch** | 19 MuJoCo touch sensors, all of them given to the policy |
| **Camera** | 96×72 depth, colour and segmentation, 6 frames a second; the arm in its pictures is your rigged model |
| **Camera network** | encoder-decoder CNN on each pixel's 3D point (from the depth) and its shaded colour: an object mask, then the box (centre, half sizes, yaw). The policy gets the box, not the picture |
| **Scripted controller** | over the object, down, close the hand, lift slowly at first, carry round the arm's base, open over the bin; the grasp as above |
| **Policy** | behaviour cloning + DAgger from the controller, then PPO with an asymmetric critic (the policy sees only what the robot could sense, and the task's phase; the critic also sees the true object state). Network 512-256-128 |
| **Rewards** | place: distance gained towards the bin with the object in hand, released over the bin, resting in it; dropping it or knocking it off costs |

## Where it runs

- **NVIDIA GPU** (needed for learning in reasonable time): MuJoCo Warp. On an 8 GB card (RTX 3070) **1024 cells learn faster than 2048**.
- **CPU**: MuJoCo, one simulation per thread. Fine for the test and for checks; slow for learning.
- **Run on: NVIDIA GPU** sends the block only to workers with an NVIDIA GPU.
- A worker whose PyTorch is built for the CPU only gets a CUDA build of it once, in its data folder, for this block only.
- Rebuilding the 3D models runs on the worker that keeps the project's datasets; a worker kept in a OneDrive folder cannot hand its result files back.

## What you need

- A worker with an NVIDIA GPU and its driver. The packages `torch`, `mujoco`, `mujoco-warp`, `warp-lang` and `numpy` are installed by the worker.
- In the assets: `models/xarm6_left_hand_rigged.glb` (the robot), `models/simple_rubber_conveyor.glb` (the conveyor) and `models/game-ready_security_camera.glb` (the camera; any 3D file with "camera" in its name).
- No account or key.

## How to run it

1. Run **1 · Learn to see** (once; again only if the cell changes).
2. Run **4 · Learn to place**. With **Training: Start over** it learns from demonstrations first; with **Continue** it carries on from the newest place policy of this cell: if that one puts things in the bin at least half the time, PPO goes on from it in small steps; otherwise it learns from demonstrations again first.
3. Run **5 · Test with new objects**, switch to **View**, choose an object and its pose, and press **Put it on the belt**. Left alone, it picks the new objects itself, one after the other.

## Settings worth knowing

On the **Conveyor grasping** block: **What to do** (the mode), **Run on**, **Cells learning at once** (1024 on an 8 GB GPU), **Minutes**, **Training** (Continue or Start over), **Live cell while learning** (No by default: a real-time cell in the same process slows learning several times; the view then replays episodes of the learning cells), and for the test: **Test: who drives** (the learned policy, or the scripted controller it learns from), **pace**, **pick objects when idle**, **Test objects** (new only or all).

## Which policy is used

Policies are carried on from (and tested) only if they were learned in the same version of the cell (each policy records a fingerprint of the cell). Among those, the furthest stage is used, then the newest. Each run keeps the best snapshot it reached, so the newest policy of a stage is that run's best.

## What you get

- `grasp_see_<date>.pt`: the camera network.
- `grasp_<stage>_<date>_<success>.pt`: the policies, each saved together with the camera network it used.
- `see.csv`, `reach.csv`, `grasp.csv`, `place.csv`: the learning curves. `test.csv`: one row per test trial (object, pose, yaw, belt speed, grasped, in the bin, largest touch force).
- From **Rebuild the 3D models**: the robot and objects models (`*.robot.json.gz`), the conveyor's belt and frame, the floor, the camera's head, mount, pole and base (`*.glb`), the sounds (`*.wav`) and the scene (`cell.scene.json`).

## How well it does now

- **The scripted controller** puts 86–88% of training objects in the bin (about 950 episodes a run). The weakest are thin and flat objects: rod 64–68%, disc 58–70%, flat box 71–81%; most others 85–100%. Its remaining misses are mostly objects that slip out of the hand as it lifts them (8–9% of episodes).
- **The learned policy**, driving with its small exploration, puts 85% in the bin during training (79% checked without it).
- **On the 12 new objects** (29 trials): 25 grasped, **21 in the bin (72%)**. Its misses: the plank and the flat egg passing by ungrasped, the U and Z blocks, a banana and a pill bottle dropped on the way.
- The goal is 90%. The next step is the grasp of thin, flat and angular objects (how low the hand reaches for flat ones, where the thumb presses on angular ones), which the policy will then imitate.

## Data and licences

The xArm's kinematics and meshes come from [xarm_ros](https://github.com/xArm-Developer/xarm_ros) by UFACTORY (BSD 3-Clause). The hand and the objects are made by the project. MuJoCo and MuJoCo Warp are Apache 2.0; PyTorch is BSD.

### 3D model credits

- Game-Ready Security Camera by [Maxitaxx](https://sketchfab.com/maxitaxx) (https://sketchfab.com/maxitaxx)
- Simple rubber conveyor by [scailman](https://sketchfab.com/scailman) (https://sketchfab.com/scailman)

## Notes

- Success during learning is counted over the episodes that have ended; early in a run the short episodes end first, so the first figures run low or high before they settle.
- EsaSim keeps 20 000 live items per run, so the block spreads the 3D view's episodes over the run's minutes.
- A cell whose physics blows up (rare, among thousands) is restarted, and the log counts how often.

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to see](blueprints/1-learn-to-see.json) · 10 blocks · version 10
- [2 · Learn to reach](blueprints/2-learn-to-reach.json) · 12 blocks · version 7
- [3 · Learn to grasp](blueprints/3-learn-to-grasp.json) · 12 blocks · version 7
- [4 · Learn to place](blueprints/4-learn-to-place.json) · 12 blocks · version 16
- [5 · Test with new objects](blueprints/5-test-with-new-objects.json) · 11 blocks · version 19
- [Learn on all my machines (a round)](blueprints/learn-on-all-my-machines-a-round.json) · 14 blocks
- [Maintenance · Rebuild the 3D models (after the hand, objects or conveyor change)](blueprints/maintenance-rebuild-the-3d-models-after-the-hand-objects-or-.json) · 5 blocks · version 34
- [tmp copy models](blueprints/tmp-copy-models.json) · 3 blocks · version 2

### Workspace blocks it uses

- [Conveyor grasping (xArm + 5-finger hand)](blocks/ws-conveyor-grasping-xarm-5-finger-hand) · version 142
- [Camera picture](blocks/ws-camera-picture) · version 1
- [Hand pressure map](blocks/ws-hand-pressure-map) · version 2
- [Lab tube handling (xArm + 5-finger hand)](blocks/ws-lab-tube-handling-xarm-5-finger-hand) · version 12

### Assets

The files the project's views draw with (3D models, textures, sounds), in [assets/](assets/):

- [covers/37d82793-5abc-4476-91d0-99524aba329e.png](assets/covers/37d82793-5abc-4476-91d0-99524aba329e.png) · 921 KB
- [covers/a70b3fa3-7da0-4067-ae25-e0bcbb39fb19.png](assets/covers/a70b3fa3-7da0-4067-ae25-e0bcbb39fb19.png) · 1.2 MB
- [grasp-policy/grasp_grasp_20261119_0649_86.pt](assets/grasp-policy/grasp_grasp_20261119_0649_86.pt) · 4.9 MB
- [grasp-policy/grasp_grasp_20261120_0000_87.pt](assets/grasp-policy/grasp_grasp_20261120_0000_87.pt) · 5.1 MB
- [grasp-policy/grasp_place_20261119_0849_67.pt](assets/grasp-policy/grasp_place_20261119_0849_67.pt) · 4.9 MB
- [grasp-policy/grasp_reach_20261119_0516_17.pt](assets/grasp-policy/grasp_reach_20261119_0516_17.pt) · 4.9 MB
- [grasp-policy/grasp_reach_20261119_1633_90.pt](assets/grasp-policy/grasp_reach_20261119_1633_90.pt) · 4.9 MB
- [grasp-policy/grasp_reach_20261119_2227_00.pt](assets/grasp-policy/grasp_reach_20261119_2227_00.pt) · 5.1 MB
- [grasp-policy/grasp_see_20261119_0340.pt](assets/grasp-policy/grasp_see_20261119_0340.pt) · 1.4 MB
- [grasp-policy/grasp_see_20261119_0352.pt](assets/grasp-policy/grasp_see_20261119_0352.pt) · 1.4 MB
- [grasp-policy/grasp_see_20261119_1531.pt](assets/grasp-policy/grasp_see_20261119_1531.pt) · 1.4 MB
- [grasp-policy/grasp_see_20261119_2103.pt](assets/grasp-policy/grasp_see_20261119_2103.pt) · 1.4 MB
- [grasp-policy/grasp_see_20261119_2126.pt](assets/grasp-policy/grasp_see_20261119_2126.pt) · 1.4 MB
- [grasp-policy/grasp_see_20261121_1813.pt](assets/grasp-policy/grasp_see_20261121_1813.pt) · 1.4 MB
- [models/belt_hum.wav](assets/models/belt_hum.wav) · 250 KB
- [models/belt_objects.robot.json.gz](assets/models/belt_objects.robot.json.gz) · 75 KB
- [models/camera_base.glb](assets/models/camera_base.glb) · 2.9 KB
- [models/camera_head.glb](assets/models/camera_head.glb) · 5.5 MB
- [models/camera_mount.glb](assets/models/camera_mount.glb) · 5.5 MB
- [models/camera_pole.glb](assets/models/camera_pole.glb) · 8.7 KB
- [models/cell.scene.json](assets/models/cell.scene.json) · 4.3 KB
- [models/conveyor_belt.glb](assets/models/conveyor_belt.glb) · 1.0 MB
- [models/conveyor_frame.glb](assets/models/conveyor_frame.glb) · 8.3 MB
- [models/drop_belt.wav](assets/models/drop_belt.wav) · 9.4 KB
- [models/drop_bin.wav](assets/models/drop_bin.wav) · 38 KB
- [models/floor.glb](assets/models/floor.glb) · 3.0 MB
- [models/game-ready_security_camera.glb](assets/models/game-ready_security_camera.glb) · 5.5 MB
- [models/rub.wav](assets/models/rub.wav) · 125 KB
- [models/servo.wav](assets/models/servo.wav) · 125 KB
- [models/simple_rubber_conveyor.glb](assets/models/simple_rubber_conveyor.glb) · 2.9 MB
- [models/xarm6_left_hand_rigged.glb](assets/models/xarm6_left_hand_rigged.glb) · 16 MB
- [models/xarm_hand.robot.json.gz](assets/models/xarm_hand.robot.json.gz) · 441 KB
- [textures/esacode-logo.png](assets/textures/esacode-logo.png) · 25 KB
- [textures/preview (56).webp](assets/textures/preview%20(56).webp) · 46 KB

The data the blueprints read (grasp-networks) stays in EsaSim.

Keys pasted in blocks (secret settings) are not saved here, nor are the team and the results of runs.

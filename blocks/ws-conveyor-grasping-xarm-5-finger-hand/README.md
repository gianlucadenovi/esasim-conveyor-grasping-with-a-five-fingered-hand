# Conveyor grasping (xArm + 5-finger hand)

A five-fingered hand on an xArm (in the physics, with collisions) learns to see objects dropped on a moving conveyor, reach, grasp with real contact forces and place them in a bin; training can be spr

Modes. Learn to see: the camera beside the robot (96x72, depth and colour) feeds a small convolutional network that learns which pixels are the object, where it is, how big and which way it lies (18 training objects, checked on 12 kept for the test). Learn to reach / grasp / place: PPO over thousands of cells (MuJoCo Warp on an NVIDIA GPU, or MuJoCo on the CPU). The xArm 6 is simulated: six position servos with torque limits follow the hand target through inverse kinematics, and the arm, the hand and the fingers collide with the pedestal, the conveyor (frame and side rails), the bin and its stand. Objects are dropped onto the moving belt so where they land varies. Each stage carries on from the one before. Multi-worker rounds: put several Learn nodes in a flow, each with its own Task in a round (1, 2, 3...) and its own Run on (GPU or CPU), all fed the same start policy; wire their Network outputs into a node in Combine a round mode, which averages their policies (weighted by the steps each gathered) into the round's policy; chain rounds for distributed training. Test with new objects: the policy runs live; pick the object, its pose, its turn and the belt speed in the 3D view; the object falls from Drop height onto the moving belt. Make the 3D models: writes the xArm-with-hand and objects robot files for the project's assets; a conveyor .glb wired to Conveyor model is fitted to the belt and baked in. Telemetry channels: joints, scene, events, camera_picture, train (train_1.. for the other tasks), rounds, test.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-conveyor-grasping-xarm-5-finger-hand` · version 13 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any worker, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt, py) | Camera networks (grasp_see_*.pt) and policies (grasp_reach/grasp/place_*.pt) to carry on from, test or combine; those left on the machine by earlier runs are found by themselves (not in a round, which Several connections. Optional. |
| `conveyor` Conveyor model | Files (glb) | Make the 3D models: a conveyor 3D file (.glb), fitted to the simulated belt and drawn in place of the boxes Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Network | Files (pt) | grasp_see_<date>.pt (Learn to see) or grasp_<stage>_<date>_<success>.pt with the camera network it used; Combine a round: the combined policy |
| `curve` Learning curve | Table (csv) | see.csv, reach.csv, grasp.csv, place.csv, or test.csv (one row per trial) |
| `models` 3D models | Files (gz) | Make the 3D models: xarm_hand.robot.json.gz and belt_objects.robot.json.gz (with the conveyor's 3D file when one is wired in), for the project's assets |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to reach"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU |
| `round_task` Task in a round | number | `0` | 0: learns alone. 1, 2, 3...: one of the Learn nodes of a multi-worker round (all start from the policy wired in; task 1 drives the 3D view); a Combine a round node averages them |
| `worlds` Cells learning at once | number | `2048` | Each cell is a belt, an object, the arm with the hand and the camera (CPU: at most 256) |
| `minutes` Minutes | number | `60` |  |
| `training` Training | select | `"Continue"` | Continue carries on from the newest network or policy (of this stage, or of the stage before) |
| `replay_every` Replay every (s) | number | `20` |  |
| `pace` Test pace (1 = real time) | number | `1` | 0: as fast as it can |
| `drop_cm` Test: drop height (cm) | number | `25` | How high above the belt the chosen object is let go |
| `autoplay` Test: pick objects when idle | select | `"Yes"` |  |
| `idle` Test: idle before it picks (s) | number | `8` |  |
| `objects` Test objects | select | `"New objects only"` |  |
| `belt_drawing` Belt drawn | select | `"Flat colours (3D file)"` | Its own materials: the 3D view draws the conveyor's file itself, textures and all (wire it to Conveyor model; needs a 3D view with scene models). Flat colours: baked into the objects model by Make the 3D models. |
| `seed` Seed | number | `1` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

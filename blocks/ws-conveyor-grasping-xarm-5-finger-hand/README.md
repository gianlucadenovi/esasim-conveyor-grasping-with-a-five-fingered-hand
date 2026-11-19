# Conveyor grasping (xArm + 5-finger hand)

A five-fingered hand on an xArm learns to see objects on a conveyor, reach them, grasp them with real contact forces and put them in a bin (MuJoCo, GPU or CPU); then a test with new objects.

Modes. Learn to see: the camera next to the robot (96x72, depth and colour, rendered in every cell) is turned into a point map; a small convolutional network learns where the object is, how big it is and which way it lies, on the 18 training objects, and is checked on the 12 objects kept for the test. Learn to reach / grasp / place: PPO, thousands of cells at once on the GPU (MuJoCo Warp); the policy senses the hand, its fingers, the 9 touch sensors (fingertips and palm), the belt speed and the camera network's box of the object; it moves the hand (kinematic: 6 stiff servos) and the fingers (real joints, real contacts). Each stage carries on from the one before. Grasp: lift 8 cm and hold for a second while the object is pushed about. Place: release it into the bin. Test with new objects: the policy runs live; pick the object and its pose in the view.

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-conveyor-grasping-xarm-5-finger-hand` · version 3 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any worker, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Policies | Files (pt) | Camera networks (grasp_see_*.pt) and policies (grasp_reach/grasp/place_*.pt) to carry on from or to test; those left on the machine by earlier runs are found by themselves Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Network | Files (pt) | grasp_see_<date>.pt (Learn to see) or grasp_<stage>_<date>_<success>.pt (Learn to reach, grasp, place) |
| `curve` Learning curve | Table (csv) | see.csv, reach.csv, grasp.csv, place.csv, or test.csv (one row per trial) |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to reach"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU. A worker whose PyTorch is built for the CPU only gets a CUDA build of it once, for this block only |
| `worlds` Cells learning at once | number | `2048` | Each cell is a belt, an object, the hand and the camera (CPU: at most 256) |
| `minutes` Minutes | number | `60` |  |
| `training` Training | select | `"Continue"` | Continue carries on from the newest network or policy (of this stage, or of the stage before) |
| `replay_every` Replay every (s) | number | `20` |  |
| `pace` Test pace (1 = real time) | number | `1` | 0: as fast as it can |
| `autoplay` Test: pick objects when idle | select | `"Yes"` |  |
| `idle` Test: idle before it picks (s) | number | `8` |  |
| `objects` Test objects | select | `"New objects only"` |  |
| `seed` Seed | number | `1` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

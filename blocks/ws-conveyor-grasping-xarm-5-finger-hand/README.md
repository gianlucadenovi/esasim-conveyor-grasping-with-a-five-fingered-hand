# Conveyor grasping (xArm + 5-finger hand)

A five-fingered hand on an xArm (in the physics, with collisions) learns to see objects dropped on a moving conveyor, reach, grasp with real contact forces and place them in a bin

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-conveyor-grasping-xarm-5-finger-hand` · version 118 |
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
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU. NVIDIA GPU: only workers that have one |
| `round_task` Task in a round | number | `0` | 0: learns alone. 1, 2, 3...: one of the Learn nodes of a multi-worker round (all start from the policy wired in; task 1 drives the 3D view); a Combine a round node averages them |
| `worlds` Cells learning at once | number | `1024` | Each cell is a belt, an object, the arm with the hand and the camera. On an 8 GB GPU 1024 learn faster than 2048 (CPU: at most 256) |
| `minutes` Minutes | number | `60` |  |
| `training` Training | select | `"Continue"` | Continue carries on from the best policy of this stage (or the stage before) |
| `live_cell` Live cell while learning | select | `"No"` | Yes: one extra cell plays the policy in real time for the 3D view. It shares the process with the learning and slows it down several times; No: the 3D view replays episodes of the learning cells instead |
| `replay_every` Replay every (s) | number | `20` |  |
| `driver` Test: who drives | select | `"The learned policy"` | The scripted controller: the teacher the policy learns from (the same camera network and touch sensors, its steps written by hand) |
| `pace` Test pace (1 = real time) | number | `1` | 0: as fast as it can |
| `drop_cm` Test: drop height (cm) | number | `25` | How high above the belt the chosen object is let go |
| `autoplay` Test: pick objects when idle | select | `"Yes"` |  |
| `idle` Test: idle before it picks (s) | number | `8` |  |
| `objects` Test objects | select | `"New objects only"` |  |
| `belt_drawing` Belt drawn | select | `"Flat colours (3D file)"` | Its own materials: the 3D view draws the conveyor's file itself, textures and all (wire it to Conveyor model; needs a 3D view with scene models). Flat colours: baked into the objects model by Make the 3D models. |
| `belt_texture` Belt texture | text | `""` | A picture of the project's assets laid on the belt and scrolled with it. Used by Make the 3D models (Flat colours). |
| `seed` Seed | number | `1` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

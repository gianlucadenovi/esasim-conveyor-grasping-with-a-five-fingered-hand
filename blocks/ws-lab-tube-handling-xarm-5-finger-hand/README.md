# Lab tube handling (xArm + 5-finger hand)

A five-fingered hand on an xArm (in the physics, with touch sensors) learns to see lab tubes and vials standing in a rack, lift each one out of its slot and set it upright in its slot of a second rack

Learn to see: the camera beside the robot learns to find the tube (where it is, how big). Learn to place: the policy first imitates the scripted controller (over the tube, down, close, lift it clear of the fences, carry it over its destination slot, lower it in, let go, rise), then PPO improves on it; success is the tube standing upright in its slot, let go and still. Test with new tubes: tubes and vials it never learned on, one at a time, live. Make the 3D models: the arm, the tubes and the cell for the project's assets (wire the rigged robot model and the camera model to Policies).

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-lab-tube-handling-xarm-5-finger-hand` · version 13 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | Any worker, NVIDIA GPU |

## Inputs

| Port | Type | Description |
|---|---|---|
| `policy` Networks and models | Files (pt, py, glb) | Camera networks (grasp_see_*.pt) and policies (grasp_place_*.pt) to carry on from or test (those left on this machine by earlier runs are found by themselves); for Make the 3D models, the rigged robot Several connections. Optional. |

## Outputs

| Port | Type | Description |
|---|---|---|
| `policy` Network | Files (pt) | grasp_see_<date>.pt (Learn to see) or grasp_place_<date>_<success>.pt with the camera network it used |
| `curve` Learning curve | Table (csv) | see.csv, place.csv or test.csv (one row per trial) |
| `models` 3D models | Files (gz, glb, json, wav) | Make the 3D models: the arm with the hand, the tubes, the cell's scene and its sounds, for the project's assets |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` What to do | select | `"Learn to place"` |  |
| `accelerator` Run on | select | `"auto"` | auto: the NVIDIA GPU when there is one (MuJoCo Warp), else the CPU. NVIDIA GPU: only workers that have one |
| `worlds` Cells learning at once | number | `1024` | Each cell is the bench with its two racks, a tube, the arm with the hand and the camera. On an 8 GB GPU 1024 learn faster than 2048 (CPU: at most 256) |
| `minutes` Minutes | number | `60` |  |
| `training` Training | select | `"Continue"` | Continue carries on from the best policy learned in this same cell; Start over learns from the scripted controller again |
| `live_cell` Live cell while learning | select | `"No"` | Yes: one extra cell plays the policy in real time for the 3D view (slows the learning down several times); No: the 3D view replays episodes of the learning cells |
| `replay_every` Replay every (s) | number | `20` |  |
| `driver` Test: who drives | select | `"The learned policy"` | The scripted controller: the teacher the policy learns from (the same camera network and touch sensors, its steps written by hand) |
| `pace` Test pace (1 = real time) | number | `1` | 0: as fast as it can |
| `autoplay` Test: pick tubes when idle | select | `"Yes"` |  |
| `idle` Test: idle before it picks (s) | number | `3` |  |
| `objects` Test tubes | select | `"New objects only"` |  |
| `seed` Seed | number | `1` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

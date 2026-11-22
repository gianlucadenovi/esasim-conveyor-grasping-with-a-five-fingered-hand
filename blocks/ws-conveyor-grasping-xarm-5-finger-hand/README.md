# Conveyor grasping (xArm + 5-finger hand)

A five-fingered hand on an xArm (in the physics, with collisions) learns to see objects dropped on a moving conveyor, reach, grasp with real contact forces and place them in a bin

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-conveyor-grasping-xarm-5-finger-hand` · version 47 |
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
| `curve` Learning curve | Table (csv) | see.csv, reach.csv, grasp.csv, place.csv, diagnosis.csv or test.csv (one row per trial) |
| `models` 3D models | Files (gz) | Make the 3D models: xarm_hand.robot.json.gz and belt_objects.robot.json.gz (with the conveyor's 3D file when one is wired in), for the project's assets |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

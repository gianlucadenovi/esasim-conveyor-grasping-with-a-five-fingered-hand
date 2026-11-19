# GPU PyTorch check (grasping)

Checks which CUDA build of PyTorch runs on this worker's NVIDIA GPU (for the conveyor grasping block).

An [EsaSim](https://esacode.com/) blueprint block: it runs on a worker, in Python (`backend.py`, `def run(ctx)`).

| | |
|---|---|
| Block | `ws-gpu-pytorch-check-grasping` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |
| Runs on | NVIDIA GPU |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `spec` Torch spec | text | `"torch==2.14.1"` |  |
| `cu` CUDA index | text | `"cu126"` |  |
| `install` Install | select | `"No"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

<!-- esasim: cover -->
![Conveyor grasping with a five-fingered hand](assets/cover.svg)
<!-- /esasim: cover -->

<div align="center">

<!-- Banner: upload a picture named banner.png under Data › Readme pictures; it shows here. -->

# Conveyor grasping with a five-fingered hand

**An xArm with a five-fingered hand learns to see objects coming on a conveyor belt, reach them, grasp them with real contact forces (touch sensors in the fingertips and palm) and put them in a bin; then it is tested on objects it never saw, chosen by you.**

![type](https://img.shields.io/badge/type-Robotics%20%26%20manipulation-blue) ![language](https://img.shields.io/badge/language-Python-3776AB) ![runs on](https://img.shields.io/badge/runs%20on-CPU%20%7C%20NVIDIA-6f42c1) [![made with](https://img.shields.io/badge/made%20with-EsaSim-0b7285)](https://esacode.com/)

<!-- More badges: ![licence](https://img.shields.io/badge/licence-MIT-green) — label, value and colour, written in the address. -->

</div>

## Flows

- **0 · xArm model to EsaSim storage (one-off)**: An empty grid: drag blocks from the menu.
- **1 · Learn to see**: The camera next to the robot learns to find the object on the belt: where it is, how big, which way it lies; checked on objects it never saw.
- **2 · Learn to reach**: The hand learns to meet the object on the moving belt: grasp point on it, palm down, turned across it, moving with the belt.
- **3 · Learn to grasp**: From the reach it learned: close the fingers and the thumb on the object, lift it 8 cm and hold it for a second while it is pushed about.
- **4 · Learn to place**: From the grasp it learned: carry the object to the bin and let it go there.
- **5 · Test with new objects**: You choose an object it never learned on and how it lies on the belt; the robot sees it, grasps it and puts it in the bin, live.

## What you need

- A computer with the EsaSim worker (Project > Compute); it runs on any worker, faster with: NVIDIA GPU.
- Data it reads: the dataset **grasp-networks** of the workspace, read by *Networks so far*; the dataset **grasp-networks** of the workspace, read by *Networks learned*.
- Python packages, installed by the worker the first time: `numpy`, `torch`, `mujoco>=3.3`, `mujoco-warp`, `warp-lang`.

## How to run it

1. Under **Compute**, check that a worker is available for this project.
2. Press **Run**, choose the worker, and follow the run on the flow's **View** page.
3. The files it made are under **Results**; they stay on the worker until you save or download them.

## What you get

- On the **View** page: *Networks so far* (2 · Learn to reach), *Cell and camera* (1 · Learn to see), *Learning curve* (2 · Learn to reach), *What the camera sees* (1 · Learn to see), *How far off* (1 · Learn to see), *Which way it lies* (1 · Learn to see), *Learning* (1 · Learn to see), *Episodes for the view* (2 · Learn to reach), *The cell* (2 · Learn to reach), *Success* (2 · Learn to reach), *How close* (2 · Learn to reach), *Reward* (2 · Learn to reach).

## Data and licences

- the dataset **grasp-networks** of the workspace, read by *Networks so far*
- the dataset **grasp-networks** of the workspace, read by *Networks learned*

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [0 · xArm model to EsaSim storage (one-off)](blueprints/0-xarm-model-to-esasim-storage-one-off.json) · 1 block · version 4
- [1 · Learn to see](blueprints/1-learn-to-see.json) · 8 blocks · version 5
- [2 · Learn to reach](blueprints/2-learn-to-reach.json) · 10 blocks · version 2
- [3 · Learn to grasp](blueprints/3-learn-to-grasp.json) · 9 blocks
- [4 · Learn to place](blueprints/4-learn-to-place.json) · 9 blocks
- [5 · Test with new objects](blueprints/5-test-with-new-objects.json) · 6 blocks

### Workspace blocks it uses

- [xArm model](blocks/ws-xarm-6-model) · version 2
- [Conveyor grasping (xArm + 5-finger hand)](blocks/ws-conveyor-grasping-xarm-5-finger-hand) · version 5
- [Conveyor grasping view](blocks/ws-conveyor-grasping-view) · version 2

The data the blueprints read (grasp-networks) stays in EsaSim.

Keys pasted in blocks (secret settings) are not saved here, nor are the team and the results of runs.

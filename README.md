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

- **1 · Learn to see**: The camera next to the robot learns to find the object on the belt: where it is, how big, which way it lies; checked on objects it never saw.
- **2 · Learn to reach**: The hand learns to meet the object on the moving belt: grasp point on it, palm down, turned across it, moving with the belt.

## What you need

- A computer with the EsaSim worker (Project > Compute); it runs on any worker, faster with: NVIDIA GPU.
- Python packages, installed by the worker the first time: `torch`, `mujoco>=3.3`, `mujoco-warp`, `warp-lang`, `numpy`.

## How to run it

1. Under **Compute**, check that a worker is available for this project.
2. Press **Run**, choose the worker, and follow the run on the flow's **View** page.
3. The files it made are under **Results**; they stay on the worker until you save or download them.

## What you get

- On the **View** page: *Cell and camera* (1 · Learn to see), *Learning curve* (1 · Learn to see), *How far off* (1 · Learn to see), *Which way it lies* (1 · Learn to see), *Learning* (1 · Learn to see).

<!-- esasim: files of the project -->

---

An [EsaSim](https://esacode.com/) project of the workspace Gianluca De Novi's workspace (Robotics & manipulation).

### Blueprints in this repository

- [1 · Learn to see](blueprints/1-learn-to-see.json) · 6 blocks · version 3
- [2 · Learn to reach](blueprints/2-learn-to-reach.json) · 0 blocks

### Workspace blocks it uses

- [Conveyor grasping (xArm + 5-finger hand)](blocks/ws-conveyor-grasping-xarm-5-finger-hand) · version 3

Keys pasted in blocks (secret settings) are not saved here. Datasets, results and the team stay in EsaSim.

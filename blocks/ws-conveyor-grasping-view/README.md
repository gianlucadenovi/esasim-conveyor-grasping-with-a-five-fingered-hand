# Conveyor grasping view

The grasping cell in 3D (belt, bin, camera, object, five-fingered hand and the xArm posed by inverse kinematics), the camera's picture with the network's box, the touch sensors, and the test's object

Wire the Conveyor grasping block's telemetry into Live and the xArm model's output into Arm model. Training runs: an episode is replayed every few seconds. Test: choose an object (new to the robot, or seen in training), how it lies, how it is turned and the belt's speed, then Put it on the belt; the trial plays live with what the camera sees and what the fingertips and the palm feel.

An [EsaSim](https://esacode.com/) blueprint block: a view that draws in the browser, in JavaScript (`frontend.js`, `function render(ctx)`).

| | |
|---|---|
| Block | `ws-conveyor-grasping-view` · version 2 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |

## Inputs

| Port | Type | Description |
|---|---|---|
| `live` Live | Stream | The Conveyor grasping block's telemetry (robot and events channels), straight or through a Telemetry block |
| `arm` Arm model | Files (gz) | xarm6.robot.json.gz from the xArm model block (or the same file in the project's assets) Optional. |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `robot_channel` Robot channel | text | `"robot"` |  |
| `events_channel` Events channel | text | `"events"` |  |
| `control_channel` Controls channel | text | `"control"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

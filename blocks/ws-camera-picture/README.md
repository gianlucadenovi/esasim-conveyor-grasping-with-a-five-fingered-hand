# Camera picture

The newest picture of a live camera, filling its panel (no margins, no grid), with how long ago it came.

Wire a live stream (a Telemetry block, or a block's telemetry) whose items, on the channel named below, carry {"image": "data:image/png;base64,..."}. The newest picture fills the whole panel, its proportions kept (resize the panel to change its size); a badge says Live and how long ago it came, grey when no picture has come for a while. Pixels: smooth, or sharp to see the camera's own pixels.

An [EsaSim](https://esacode.com/) blueprint block: a view that draws in the browser, in JavaScript (`frontend.js`, `function render(ctx)`).

| | |
|---|---|
| Block | `ws-camera-picture` · version 1 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |

## Inputs

| Port | Type | Description |
|---|---|---|
| `images` Pictures | Stream | Live items with an image (a data URL) Several connections. |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `channel` Channel | text | `"camera_picture"` | The stream channel with the pictures (empty: any item with an image) |
| `pixels` Pixels | select | `"smooth"` | sharp: the camera's own pixels, as squares |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

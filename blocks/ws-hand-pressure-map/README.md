# Hand pressure map

A five-fingered hand seen from the palm: each touch sensor where it is on the hand, coloured from black to white by its pressure, with its reading.

Wire a live stream (a Telemetry block, or a block's telemetry) whose items, on the channel named below, carry one number per sensor in newtons: thumb_tip, thumb_mid, thumb_prox (the thumb's metacarpal), index_tip, index_mid, index_prox ... little_prox, and palm_1 to palm_4 (palm_1 the fingers' side by the index, palm_2 the fingers' side by the little finger, palm_3 and palm_4 the wrist side). Optional: grip_n (the grip in all) and holding (1 when holding). Colours go black, blue, cyan, green, yellow, red, white as the pressure grows to the highest value; hover a sensor for its name.

An [EsaSim](https://esacode.com/) blueprint block: a view that draws in the browser, in JavaScript (`frontend.js`, `function render(ctx)`).

| | |
|---|---|
| Block | `ws-hand-pressure-map` · version 1 |
| Workspace | Gianluca De Novi's workspace |
| Author | Gianluca De Novi |

## Inputs

| Port | Type | Description |
|---|---|---|
| `data` Sensors | Stream | Items with one field per sensor (N) Several connections. |

## Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `channel` Channel | text | `"touch"` | The stream channel with the sensors' readings |
| `max` Highest pressure (N) | number | `30` | White from here up |
| `scale` Colour scale | select | `"log"` | log: light touches still show |
| `hand` Hand | select | `"right"` |  |

## Use it in EsaSim

In a workspace, open **Projects > Blocks > New block**, choose this kind of block, copy the ports and settings above and paste the code; or ask an AI assistant connected to EsaSim (MCP) to create it from this folder.

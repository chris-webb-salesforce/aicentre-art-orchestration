# Feature: Operator Config Panel

## Overview

Add a web-based operator panel to the capture app that allows live editing of per-arm configuration (z heights, feedrates, drawing bounds, etc.) without SSH or file editing. Changes apply immediately and persist across restarts.

## Motivation

Each arm station needs individual tuning — especially `z_up` and `z_down` which vary based on pen length, paper thickness, and surface height. Currently this requires editing `config/settings.yaml` and restarting the arm process. An operator panel allows real-time tweaking from any device on the network.

## Architecture

### New route: `/operator`

A password-protected page on the existing capture app (`capture_app.py`). Separate from the user-facing `/capture` page.

- Shows all connected arms with their current config values
- Editable form fields for per-arm settings
- "Apply" sends changes to the arm immediately (no restart)
- "Save" persists changes to `settings.yaml` on the Mac Mini

### New WebSocket events

| Event | Direction | Namespace | Payload | Description |
|-------|-----------|-----------|---------|-------------|
| `get_config` | server → arm | `/robot` | `{}` | Request current config from arm |
| `arm_config` | arm → server | `/robot` | `{arm_id, config: {...}}` | Arm reports its current config |
| `config_update` | server → arm | `/robot` | `{config: {...}}` | Apply new config values at runtime |
| `config_saved` | arm → server | `/robot` | `{arm_id, success}` | Confirm config was written to YAML |

### Editable fields (per arm)

**Drawing settings:**
- `z_up` — pen lift height (mm)
- `z_down` — pen contact height (mm)
- `x_min`, `x_max`, `y_min`, `y_max` — drawing bounds (mm)
- `flip_x`, `flip_y` — axis mirroring

**DexArm settings:**
- `feedrate` — drawing speed (mm/min)
- `travel_feedrate` — pen-up travel speed (mm/min)
- `acceleration` — drawing acceleration (mm/s^2)
- `travel_acceleration` — travel acceleration (mm/s^2)
- `jerk` — jerk limit (mm/s)

### Runtime hot-swap in `remote_client.py`

When a `config_update` event is received, the arm process updates:

1. `DexArmController` — `z_up`, `z_down`, `feedrate`, `travel_feedrate`, and sends new `M204`/`M205` GCode commands to the arm firmware for acceleration/jerk changes
2. `GCodeGenerator` — `DrawingBounds` with new x/y bounds, z heights, feedrates, flip settings

No restart required. The next drawing job uses the updated values.

### Persistence via `config_save`

When the operator clicks "Save", the relay server sends a `config_save` event to the arm. The arm process:

1. Reads the current `settings.yaml`
2. Updates the matching `arms[].drawing` and `arms[].dexarm` sections
3. Writes the file back
4. Emits `config_saved` confirmation

This ensures the tuned values survive process restarts.

## Files to create/modify

| File | Action | Description |
|------|--------|-------------|
| `src/web/templates/operator.html` | Create | Operator panel UI |
| `src/web/capture_app.py` | Modify | Add `/operator` route, `config_update`/`config_save` relay events |
| `src/web/remote_client.py` | Modify | Add `config_update` and `config_save` handlers, hot-swap logic |
| `src/hardware/dexarm_controller.py` | Modify | Add method to update acceleration/jerk at runtime via GCode |

## UI sketch

```
+--------------------------------------------------+
|  Operator Panel                          [Logout] |
+--------------------------------------------------+
|                                                    |
|  Station 1 (arm-1)              [Ready] [Drawing] |
|  ┌──────────────────────────────────────────────┐ |
|  │  z_up:  [-65.0]    z_down: [-70.0]           │ |
|  │  feedrate: [6000]  travel: [4000]             │ |
|  │  accel: [200]      jerk: [7.0]               │ |
|  │  bounds: x[-53, 54]  y[232, 337]             │ |
|  │                        [Apply]  [Save]        │ |
|  └──────────────────────────────────────────────┘ |
|                                                    |
|  Station 2 (arm-2)                        [Idle]  |
|  ┌──────────────────────────────────────────────┐ |
|  │  z_up:  [-65.0]    z_down: [-70.0]           │ |
|  │  ...                                          │ |
|  └──────────────────────────────────────────────┘ |
|                                                    |
+--------------------------------------------------+
```

## Feature: Pen Height Test Mode

### Motivation

Getting `z_up` and `z_down` right is critical — too high and the pen doesn't touch paper, too low and it digs in or stalls the arm. Currently the only way to test is to submit a full drawing job. The operator needs a way to test pen heights interactively from the panel.

### UX flow

1. Operator selects an arm on the operator panel
2. Clicks "Test Pen Height"
3. The arm moves to center of its drawing bounds at `z_up` height
4. Panel shows two buttons: **"Pen Down"** and **"Pen Up"** with the current z values displayed
5. Operator adjusts `z_down` / `z_up` sliders and taps the buttons to see the result in real time
6. Once satisfied, clicks **"Apply"** to keep the values or **"Cancel"** to revert
7. Arm returns to safe position

### New WebSocket events

| Event | Direction | Namespace | Payload | Description |
|-------|-----------|-----------|---------|-------------|
| `test_pen_start` | server → arm | `/robot` | `{}` | Move arm to center and enter test mode |
| `test_pen_move` | server → arm | `/robot` | `{z: float}` | Move pen to specific Z height |
| `test_pen_stop` | server → arm | `/robot` | `{}` | Exit test mode, return to safe position |
| `test_pen_status` | arm → server | `/robot` | `{arm_id, z, mode: "up"\|"down"}` | Confirm current pen position |

### Implementation notes

- Test mode should block the arm from accepting drawing jobs (arm status becomes `testing`)
- The `DexArmController` already has `pen_up()`, `pen_down()`, and `move_to()` — the test handler just calls these with operator-supplied Z values
- The operator panel should show the live Z position reported back from the arm
- Slider range should be constrained to safe values (e.g. -50 to -75 for typical DexArm setups)

### UI addition to operator panel

```
+--------------------------------------------------+
|  Station 1 — Test Pen Height                      |
|  ┌──────────────────────────────────────────────┐ |
|  │                                               │ |
|  │  z_up:   ━━━━━━━━●━━━━━━  [-65.0]           │ |
|  │          [Move Up]                            │ |
|  │                                               │ |
|  │  z_down: ━━━━━━━━━━●━━━━  [-70.0]           │ |
|  │          [Move Down]                          │ |
|  │                                               │ |
|  │  Current Z: -65.0                             │ |
|  │                                               │ |
|  │              [Apply & Exit]  [Cancel]          │ |
|  └──────────────────────────────────────────────┘ |
+--------------------------------------------------+
```

## Open questions

- Should there be a "copy config from arm X to arm Y" shortcut?
- Should config changes be logged/audited?

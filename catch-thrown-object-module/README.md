# catch-thrown-object

Viam generic service that attempts a **soft underhand intercept** of a colored object inside a tight catch volume on armfarm05.

This is **not** a high-speed sports catcher. It refuses fast throws, stays inside a small AABB above the table, keeps workcell obstacles active, and defaults to preview-only.

## DoCommand

| Action | Moves? | Requirements |
|--------|--------|----------------|
| `preview` | No | — |
| `arm_ready` | Yes (slow) | `enable_catch: true` |
| `catch` | Yes | `enable_catch: true` + `safety_ack: "I_UNDERSTAND_CATCH_RISKS"` |
| `abort` | Stop | — |

## Deploy

Preferred for the hackathon: [INLINE_SETUP.md](./INLINE_SETUP.md) (Viam Control Code).

## Hard caps (config cannot raise)

- Arm ≤ 15 deg/s  
- Object ≤ 900 mm/s  
- Session ≤ 8 s  
- TCP ≥ table top + 70 mm  
- Catch volume diagonal ≤ 900 mm  

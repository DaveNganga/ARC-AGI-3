# Preliminary experiment outputs

These files are engineering smoke tests from the controlled 64-hypothesis environment.

| File | Conditions | Worlds | Result |
|---|---|---:|---|
| `factorized_benchmark_64_worlds.json` | Policy proxy and exact classical QAHS | 64 | Both solved 64/64; 7 actions on average |
| `qaoa_smoke_8_worlds.json` | QAOA, post-development smoke configuration | 8 | Solved 8/8; 6.875 actions on average |
| `qaoa_smoke_worlds_0_3.json` | QAOA development worlds | 4 | Solved 4/4 |
| `qaoa_heldout_worlds_4_7.json` | Separate QAOA smoke worlds | 4 | Solved 4/4 |

The QAOA configuration and consistency penalties were developed using very small samples. These results are not held-out evidence and must not be described as a quantum advantage. The 6.875-action average includes a correct early guess on one world.


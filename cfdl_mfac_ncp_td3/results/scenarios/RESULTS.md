# Robustness Scenarios (6-DOF hybrid AUV controller)

Three robustness scenarios adapted to the REMUS 6-DOF plant and this study's controllers (Hybrid vs MPC, PID, Fuzzy-PID). Planar references (x, y, psi) are embedded with zero depth/roll/pitch. Disturbance magnitudes are scaled to the same fraction of actuator authority as the source scenario (F_max = 2000 N).

## Scenario 1 - square trajectory + measurement noise

Position RMSE [m] per velocity-measurement SNR (position SNR = 80 dB):

| Controller | Test 0 (none) | 60 dB | 40 dB | 30 dB | 25 dB | 20 dB |
|---|---|---|---|---|---|---|
| Hybrid | 0.1110 | 0.1046 | 0.1023 | 0.1059 | 0.1053 | 0.1055 |
| MPC | 0.1356 | 0.1356 | 0.1356 | 0.1356 | 0.1356 | 0.1357 |
| PID | 14.1113 | 14.0164 | 13.7302 | 13.9760 | 14.6395 | 16.0065 |
| Fuzzy | 6.9325 | 6.8838 | 6.7804 | 6.7396 | 6.4059 | 6.7064 |

## Scenario 2 - lemniscate + external disturbances (10-40 s)

| Controller | pos RMSE | yaw RMSE | pos RMSE (dist.) | yaw RMSE (dist.) | energy |
|---|---|---|---|---|---|
| Hybrid | 0.1262 | 0.0106 | 0.1254 | 0.0134 | 22491.9 |
| MPC | 0.1361 | 0.0193 | 0.1424 | 0.0220 | 21521.0 |
| PID | 9.5224 | 1.5576 | 7.6150 | 1.6360 | 225498.8 |
| Fuzzy | 5.2108 | 1.2190 | 5.6999 | 1.1933 | 154716.3 |

## Scenario 3 - circle + +/-10% time-varying parametric uncertainty

| Controller | pos RMSE | yaw RMSE | energy |
|---|---|---|---|
| Hybrid | 0.1158 | 0.0096 | 84872.6 |
| MPC | 0.1862 | 0.0275 | 86522.6 |
| PID | 3.7576 | 1.0216 | 159930.7 |
| Fuzzy | 2.7395 | 0.6064 | 122501.4 |

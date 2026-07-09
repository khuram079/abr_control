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
| Hybrid | 0.2948 | 0.0109 | 0.3178 | 0.0135 | 36871.8 |
| MPC | 0.3690 | 0.0265 | 0.4135 | 0.0267 | 37428.3 |
| PID | 8.9296 | 1.5068 | 7.8467 | 1.6430 | 229144.0 |
| Fuzzy | 4.8473 | 1.2782 | 5.7177 | 1.2080 | 157573.9 |

## Scenario 3 - circle + +/-15% time-varying parametric uncertainty

| Controller | pos RMSE | yaw RMSE | energy |
|---|---|---|---|
| Hybrid | 0.2869 | 0.0091 | 87003.8 |
| MPC | 0.3278 | 0.0339 | 87759.0 |
| PID | 8.9292 | 1.4049 | 245810.9 |
| Fuzzy | 10.2280 | 0.8342 | 213069.2 |

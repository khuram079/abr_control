# Robustness Scenarios (6-DOF hybrid AUV controller)

Three robustness scenarios adapted to the REMUS 6-DOF plant and this study's controllers (Hybrid vs MPC, PID, Fuzzy-PID). Planar references (x, y, psi) are embedded with zero depth/roll/pitch. Disturbance magnitudes are scaled to the same fraction of actuator authority as the source scenario (F_max = 2000 N).

## Scenario 1 - square trajectory + measurement noise

Position RMSE [m] per velocity-measurement SNR (position SNR = 80 dB):

| Controller | Test 0 (none) | 60 dB | 40 dB | 30 dB | 25 dB | 20 dB |
|---|---|---|---|---|---|---|
| Hybrid | 0.4181 | 0.4216 | 0.4205 | 0.4208 | 0.4210 | 0.4226 |
| MPC | 0.3803 | 0.3802 | 0.3802 | 0.3803 | 0.3803 | 0.3803 |
| PID | 17.1685 | 16.2898 | 16.5668 | 15.7531 | 16.4069 | 16.8241 |
| Fuzzy | 6.4023 | 6.3571 | 6.5305 | 6.4408 | 6.5417 | 6.4275 |

## Scenario 2 - lemniscate + external disturbances (10-40 s)

| Controller | pos RMSE | yaw RMSE | pos RMSE (dist.) | yaw RMSE (dist.) | energy |
|---|---|---|---|---|---|
| Hybrid | 0.3311 | 0.0112 | 0.3971 | 0.0136 | 41717.4 |
| MPC | 0.4255 | 0.0291 | 0.5018 | 0.0286 | 41581.5 |
| PID | 5.5444 | 1.4242 | 7.3762 | 1.7481 | 174188.2 |
| Fuzzy | 4.9751 | 1.1937 | 6.7171 | 1.2238 | 136133.2 |

## Scenario 3 - circle + 100% time-varying parametric uncertainty

| Controller | pos RMSE | yaw RMSE | energy |
|---|---|---|---|
| Hybrid | 2.8898 | 0.0649 | 251685.2 |
| MPC | 3.4377 | 0.0513 | 269173.9 |
| PID | 5.6755 | 0.2785 | 231992.0 |
| Fuzzy | 4.7455 | 0.2607 | 228013.8 |

.\.venv\Scripts\activate
Here is a quick comparison of all three:

BC (Fastest): All three angular velocities (wx, wy, and wz) settle at 0 rad/s by approximately Timestep 40.

PID (Second Fastest): This controller is also quite fast, but it takes slightly longer for all lines to stabilize at 0, converging around Timestep 50.

RL (Slowest): This controller is noticeably slower than the other two. The lines don't fully settle at 0 until around Timestep 120.

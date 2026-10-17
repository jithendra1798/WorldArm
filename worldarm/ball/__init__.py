"""Chapter 2: a ball rolled down the table, tracked by the overhead camera, its path predicted by a rolling
model learned from our own rolls, and intercepted by the arm before it arrives.

  table.py      overhead-camera pixels <-> table-plane positions in an arm's world frame (homography)
  detect.py     the black ball in an overhead frame
  model.py      the world model of motion: a rolling ball slows at a learned rate (vs. constant velocity)
  track.py      fit that model to the latest sightings: where the ball is, how fast, where it will cross
  intercept.py  where to put the paddle and whether it can get there in time (vs. chasing)
  sim.py        rolls, camera and arm simulated, to measure intercept rates without the robot
  live.py       the real loop on the robot
"""

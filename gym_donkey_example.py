import os
import gym
import gym_donkeycar
import numpy as np

# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object
exe_path = f"/content/AV-Driving/ donkey-car-sim-usi/donkey-sim-linux.x86_64"
port = 9091

env_id = "donkey-warehouse-v0"

conf = { "exe_path" : exe_path, "port" : port }

env = gym.make(env_id, conf=conf)

# PLAY
for _ in range(3):
    obs = env.reset()
    for t in range(100):
      action = np.array([0.0, 0.5]) # drive straight with small speed
      # execute the action
      obs, reward, done, info = env.step(action)
      print(reward)

# Exit the scene
env.close()

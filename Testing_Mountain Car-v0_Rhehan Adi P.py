import numpy as np

# Patch kompatibilitas untuk numpy >= 1.24
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import numpy as np
import gym
from tensorflow.keras.models import load_model
import time

# Load model hasil training
model = load_model("dqn_mountaincar.keras")

# Buat environment
env = gym.make("MountainCar-v0", render_mode="human")  # render_mode="human" untuk lihat visual
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

episodes = 10  # jumlah episode untuk testing

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time_step in range(200):  # max step di MountainCar = 200
        # Model pilih aksi terbaik (tanpa epsilon exploration)
        q_values = model.predict(state, verbose=0)
        action = np.argmax(q_values[0])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.2f}")
            break

        # Biar jalannya tidak terlalu cepat
        time.sleep(0.02)

env.close()

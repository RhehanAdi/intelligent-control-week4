import gym
from dqn_agent import DQNAgent
from tensorflow.keras.models import load_model

# Buat environment untuk mendapatkan ukuran state dan action
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen dengan arsitektur yang sama
agent = DQNAgent(state_size, action_size)

# Load full model hasil training lama (.h5)
print("🔄 Loading full model dari dqn_cartpole.h5 ...")
agent.model = load_model("dqn_cartpole.h5", compile=False)

# ✅ Simpan ulang bobot dengan nama sesuai aturan Keras 3
agent.model.save_weights("dqn_cartpole.weights.h5")
print("✅ Konversi berhasil! Bobot disimpan sebagai dqn_cartpole.weights.h5")

env.close()

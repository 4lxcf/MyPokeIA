from stable_baselines3 import PPO
import os
from PokemonRedEnv import PokemonRedEnv

# Paths para salvar
models_dir = f"models/"
logdir = f"logs/"

# Criação das pastas, caso ainda não existam
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Criando o ambiente (environment)
env = PokemonRedEnv("D:\Dev\MyPokeIA\PokemonRed.gb")
env.reset()

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, learning_rate=0.00003)

# episodes = 5
TIMESTEPS = 2048*490
# for episode in range(episodes):
    # model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=f"PPO-ep{episode}")
    # model.save(f"{models_dir}/model_TIMESTEPS{TIMESTEPS}_LR00003_EP{episode}")
model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=f"PPO")
model.save(f"{models_dir}/model_TIMESTEPS{TIMESTEPS}_LR00003")

env.close()

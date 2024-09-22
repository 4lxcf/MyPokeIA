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

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

episodes = 50
TIMESTEPS = 2048*50
for episode in episodes:
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, tb_log_name=f"PPO-ep{episode}")
    model.save(f"{models_dir}/model_TIMESTEPS{TIMESTEPS}_LR{model.learning_rate}_EP{episode}")

env.close()

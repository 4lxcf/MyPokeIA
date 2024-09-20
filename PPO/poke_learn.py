from stable_baselines3 import PPO
import os
import time
from PokemonRedEnv import PokemonRedEnv

# Paths para salvar
models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"

# Criação das pastas, caso ainda não existam
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Criando o ambiente (environment)
env = PokemonRedEnv("D:\Dev\MyPokeIA\PokemonRed.gb")
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(100):
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
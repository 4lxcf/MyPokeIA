from stable_baselines3 import PPO
from PokemonRedEnv import PokemonRedEnv

env = PokemonRedEnv("D:\Dev\MyPokeIA\PokemonRed.gb")
env.reset()

model_path = 'models/model_TIMESTEPS1003520_LR00003.zip'

model = PPO.load(model_path, env=env)

episodes = 100

for ep in range(episodes):
    observation = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        # env.render()

env.close()

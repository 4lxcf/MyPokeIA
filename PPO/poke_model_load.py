from stable_baselines3 import PPO
from PokemonRedEnv import PokemonRedEnv

env = PokemonRedEnv("D:\Dev\MyPokeIA\PokemonRed.gb")
env.reset()

model_path = 'models/1726796713/990000.zip'

model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    observation = env.reset()
    terminated = False
    while not terminated:
        # env.render()
        action, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)

env.close()

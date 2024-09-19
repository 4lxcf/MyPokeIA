from PokemonRedEnv import PokemonRedEnv 

if __name__ == "__main__":
    # Inicialize o ambiente
    env = PokemonRedEnv('D:\Dev\MyPokeIA\PokemonRed.gb')

    # Resetar o ambiente
    observation = env.reset()
    print(f"Observação Inicial: {observation}")

    for _ in range(500):
        random_action = env.action_space.sample()
        observation, reward, terminated, info = env.step(random_action)
        print(f"Observação: {observation}, Recompensa: {reward}, Terminado: {terminated}, Info: {info}")
        
    print("O episódio terminou.")
    env.close()
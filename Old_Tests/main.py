from stable_baselines3 import DQN
import gymnasium as gym  # Alterado para gymnasium
from gymnasium import spaces
from pyboy import PyBoy
import numpy as np
from PIL import Image

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.pyboy = PyBoy('D:\Dev\MyPokeIA\PokemonRed.gb')
        self.render_mode = 'human'
        self.action_space = spaces.Discrete(5) # Cima, baixo, esquerda, direita, a, b
        self.resized_width = 80
        self.resized_height = 72 
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8)  # Reduzido para metade da resolução original
        self.step_count = 0
        self.max_steps = 16384
        self.visited_positions = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Adicione suporte ao argumento seed (se necessário)
        self._initialize_game() # Preparar o emulador carregando o estado do jogo
        self.step_count = 0
        self.max_steps = 16384
        self.visited_positions.clear()
        
        return self._get_state(), {}

    def step(self, action):
        # Executa a ação usando a função button()
        if action == 0:
                self.pyboy.button("up")
        elif action == 1:
                self.pyboy.button("down")
        elif action == 2:
                self.pyboy.button("right")
        elif action == 3:
                self.pyboy.button("left")
        elif action == 4:
                self.pyboy.button("a")
        self.pyboy.tick(False) # Tick do emulador
        current_position = self._get_position() # Obter posição e atualizar posições visitadas
        reward = self._calculate_reward(current_position) # Calcula a recompensa 
        
        if current_position in self.visited_positions:
            self.visited_positions[current_position] += 1
        else:
            self.visited_positions[current_position] = 1
        self.step_count += 1 # Incrementa o contador de passos

        done = self.step_count >= self.max_steps # Verifica se o jogo terminou
        state = self._get_state()
        truncated = done # Adiciona um valor extra para o truncated
        info = {"TimeLimit.truncated": truncated} # Adiciona um dicionário info com informações adicionais
        
        if done:
            self.reset_emulator()  # Reinicializa o emulador se o jogo estiver concluído
        
        return state, reward, done, truncated, info

    def render(self, mode='human'):
        pass
        
    def _get_state(self):
        image = self.pyboy.screen.image  # Captura a tela do emulador como uma imagem PIL
        image = image.convert('RGB')  # Converte a imagem para o modo RGB
        image = image.resize((self.resized_width, self.resized_height))  # Redimensiona a imagem
        screen_array = np.array(image)  # Converte a imagem PIL para um array numpy
        return screen_array

    def _get_position(self):
        # Implementar a lógica para obter a posição do personagem
        map_id = self.pyboy.memory[0XD35E]
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        return (map_id, x, y)

    def _initialize_game(self):
        # Carrega o estado inicial do jogo pós puzzles iniciais
        with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.set_emulation_speed(9)

    def _calculate_reward(self, position):
        if self.pyboy.memory[0xD057] == 0: # Check se está em batalha
            if self.visited_positions.get(position, 0) == 0:
                reward = 10.0  # Recompensa por explorar um local novo
            else:
                reward = -0.2 # Penalidade por revisitar locais conhecidos
        else:
            return 0
        return reward
        
    def reset_emulator(self):
        self.pyboy.stop()
        # Recria o emulador
        self.pyboy = PyBoy('D:\Dev\MyPokeIA\PokemonRed.gb')  

# Envolver o ambiente com Monitor e DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = Monitor(PokemonEnv())
env = DummyVecEnv([lambda: env])

# Treinar o modelo DQN
model = DQN('CnnPolicy', env, verbose=1, buffer_size=500000, batch_size=512, exploration_fraction=0.6, exploration_final_eps=0.2, learning_rate=0.0005)
model.learn(total_timesteps=16384000)

# Usar o modelo treinado para jogar
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

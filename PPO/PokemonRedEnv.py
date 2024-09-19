import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

CHAR_LEN_VISITED_POSITIONS_GOAL = 900*57
N_DISCRETE_ACTIONS = 5 # Cima, baixo, direita, esquerda, a
N_CHANNELS = 3

class PokemonRedEnv(gym.Env):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, rom_path):
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, 255, 255]), dtype=np.uint8)

        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(3)

    def step(self, action):
        # Executa a ação usando a função button()
        if action == 0:
                self.pyboy.button_press("up")
        elif action == 1:
                self.pyboy.button_press("down")
        elif action == 2:
                self.pyboy.button_press("right")
        elif action == 3:
                self.pyboy.button_press("left")
        elif action == 4:
                self.pyboy.button_press("a")

        self.pyboy.tick() # Tick do emulador
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("right")
        self.pyboy.button_release("left")
        self.pyboy.button_release("a")

        # Obter nova observação
        map_id = self.pyboy.memory[0xD35E]
        pos_x = self.pyboy.memory[0xD362]
        pos_y = self.pyboy.memory[0xD361]
        observation = np.array([map_id, pos_x, pos_y], dtype=np.uint8)

        # 4. Calcular a recompensa
        reward = self.calculate_reward((map_id, pos_x, pos_y))  # Defina a lógica de recompensa conforme necessário

        # Verificar se o episódio terminou
        terminated = {}

        # Atualizar a lista de posições visitadas
        self.visited_positions.add((map_id, pos_x, pos_y))

        # 6. Informações adicionais
        info = {}  # Pode ser um dicionário com informações adicionais, se necessário

        return observation, reward, terminated, info

    def reset(self):
        # 1. Carregar o estado salvo no início do jogo
        with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)

        # 2. Ler os valores dos três endereços de memória
        map_id = self.pyboy.memory[0xD35E]  # ID do mapa
        pos_x = self.pyboy.memory[0xD362]   # Posição X do personagem
        pos_y = self.pyboy.memory[0xD361]   # Posição Y do personagem
        observation = np.array([map_id, pos_x, pos_y], dtype=np.uint8)

        # 3. Inicializar o conjunto de posições visitadas
        self.visited_positions = set()
        self.visited_positions.add((map_id, pos_x, pos_y))

        # Inicializar terminated como False
        terminated = {}

        return observation, terminated
    
    def calculate_reward(self, observation):
        # Recompensa por visitar novas posições
        if observation not in self.visited_positions:
            reward = 1  # Recompensa positiva por descobrir uma nova posição
        else:
            reward = 0  # Nenhuma recompensa por visitar uma posição já conhecida

        return reward
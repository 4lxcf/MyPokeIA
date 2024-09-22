import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

N_DISCRETE_ACTIONS = 5 # Cima, baixo, direita, esquerda, a
N_CHANNELS = 3

class PokemonRedEnv(gym.Env):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, rom_path):
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, 255, 255]), dtype=np.uint8)

        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(0)

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

        # Verificar se o personagem está em batalha
        self.in_battle = self.pyboy.memory[0xD057] == 1

        # Verificar se o episódio terminou
        terminated = False
        truncated = False

        actual_position = (map_id, pos_x, pos_y)

        # Calcular a recompensa de posição
        reward = self.calculate_reward((map_id, pos_x, pos_y))

        if not self.in_battle:              
            # Atualizar a lista de posições visitadas
            if actual_position in self.visited_positions:
                self.visited_positions[actual_position] += 1
            else:
                self.visited_positions[actual_position] = 1

        # Verificar se alguma posição foi visitada e resetar o ambiente caso tenha sido muitas vezes
        if self.visited_positions[actual_position] >= 150:
            print(f"Resetting environment. Position {actual_position} visited {self.visited_positions[actual_position]} times.")
            reward = -10
            observation, terminated = self.reset()
            return observation, reward, True, False, {}  # Retornar com `terminated = True`

        # Verifica se os níveis dos Pokémon aumentaram e adiciona à recompensa
        current_levels_sum = self.get_levels_sum()
        if current_levels_sum > self.last_levels_sum:
            reward += (current_levels_sum - self.last_levels_sum)  # Recompensa adicional por aumento de nível
            self.last_levels_sum = current_levels_sum  # Atualiza o nível total da party

        # Informações adicionais
        info = {"visited_positions_size": len(self.visited_positions)}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options={}):
        # Se o seed for fornecido, defina a semente do ambiente
        if seed is not None:
            np.random.seed(seed)

        # Carregar o estado salvo no início do jogo
        with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)

        # Ler os valores dos três endereços de memória
        map_id = self.pyboy.memory[0xD35E]  # ID do mapa
        pos_x = self.pyboy.memory[0xD362]   # Posição X do personagem
        pos_y = self.pyboy.memory[0xD361]   # Posição Y do personagem
        observation = np.array([map_id, pos_x, pos_y], dtype=np.uint8)

        # Inicializar o conjunto de posições visitadas
        self.visited_positions = {}
        actual_position = (map_id, pos_x, pos_y)
        self.visited_positions[actual_position] = 1

        # Inicializar áreas e posições exploradas
        self.explored_areas = set()
        self.explored_positions = set()
        
        # Armazena o nível total inicial da party
        self.last_levels_sum = self.get_levels_sum()

        # Inicializar terminated como False
        terminated = False

        return observation, terminated
    
    def calculate_reward(self, observation):
    # Penalize positions revisited multiple times
        # if not self.in_battle:
        #     if observation in self.visited_positions:
        #         if self.visited_positions[observation] > 50:  # Definir um limite de revisitações aceitáveis
        #             reward = -0.5  # Penalização maior para revisitações excessivas
        #         else:
        #             reward = -0.1  # Penalização leve para revisitações
        #     else:
        #         reward = 1  # Recompensa por descobrir uma nova posição
        # else:
        #      reward = 0
        # return reward

        map_id, pos_x, pos_y = observation
        reward = 0

        if not self.in_battle:
        # Verifica se o agente está em uma nova área
            if map_id not in self.explored_areas:
                reward += 10  # Recompensa por descobrir uma nova área
                self.explored_areas.add(map_id)

        # Verifica se o agente está em uma nova posição dentro da área
            position_key = (map_id, pos_x, pos_y)
            if position_key not in self.explored_positions:
                reward += 2  # Recompensa por explorar uma nova posição dentro da área
                self.explored_positions.add(position_key)
            else:
                reward -= 0.1  # Penalização leve por revisitar uma posição

        # Penaliza revisitações excessivas (mais de 50 vezes na mesma posição)
            if self.visited_positions.get(position_key, 0) > 50:
                reward -= 1  # Penalização maior para revisitações excessivas
        # print(f"visited_positions:{self.visited_positions} | explored_areas:{self.explored_areas}")

        return reward
    
    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.pyboy.memory[addr] - min_poke_level, 0)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]  # Endereços de memória dos Pokémon
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

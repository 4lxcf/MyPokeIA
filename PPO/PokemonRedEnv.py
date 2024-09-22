import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

N_DISCRETE_ACTIONS = 5 # Cima, baixo, direita, esquerda, a
N_CHANNELS = 3
MAX_STEPS = 2048

class PokemonRedEnv(gym.Env):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(6)

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, 255, 255]), dtype=np.uint8)

    def reset(self, seed=None, options={}):
        # Se o seed for fornecido, defina a semente do ambiente
        if seed is not None:
            np.random.seed(seed)

        # Carregar o estado salvo no início do jogo
        with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)

        self.agent_stats = []
        self.visited_positions = {}
        self.step_count = 0
        self.max_steps = MAX_STEPS
        self.max_levels = 0
        self.max_visited_positions = 0

        # Ler os valores dos três endereços de memória
        map_id = self.pyboy.memory[0xD35E]  # ID do mapa
        pos_x = self.pyboy.memory[0xD362]   # Posição X do personagem
        pos_y = self.pyboy.memory[0xD361]   # Posição Y do personagem
        observation = np.array([map_id, pos_x, pos_y], dtype=np.uint8)

        return observation, {}

    def step(self, action):
        self.make_action(action)
        self.update_agent_stats()

        map_id, pos_x, pos_y = (self.pyboy.memory[0xD35E], self.pyboy.memory[0xD362], self.pyboy.memory[0xD361])
        self.step_count += 1

        actual_position = (map_id, pos_x, pos_y)
        if actual_position not in self.visited_positions:
            self.visited_positions[actual_position] = self.step_count

        observation = np.array([map_id, pos_x, pos_y], dtype=np.uint8) 
        reward = self.calculate_reward() 
        truncated = self.step_limit_count()
        info = {f"Agent_Stats:": self.agent_stats} # Infos adicionais
        return observation, reward, False, truncated, info

    def make_action(self, action):
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

        self.pyboy.tick(8) # Tick do emulador
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("right")
        self.pyboy.button_release("left")
        self.pyboy.button_release("a")
    
    def update_agent_stats(self):
         map_id, pos_x, pos_y = (self.pyboy.memory[0xD35E], self.pyboy.memory[0xD362], self.pyboy.memory[0xD361])
         levels = [self.pyboy.memory[e] for e in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
         self.agent_stats.append(
              {
                   "step": self.step_count,
                   "pos_x": pos_x,
                   "pos_y": pos_y,
                   "map": map_id,
                   "visited_positions_count": len(self.visited_positions),
                   "levels": levels,
                   "levels_sum": sum(levels)
              }
         )

    def step_limit_count(self):
        step_limit_flag = self.step_count >= self.max_steps
        return step_limit_flag

    def calculate_reward(self):
        reward = 0
        level_sum = self.get_levels_sum()
        visited_positions_count = len(self.visited_positions)

        if level_sum > self.max_levels:
             reward += level_sum - self.max_levels
             self.max_levels = level_sum

        if visited_positions_count > self.max_visited_positions:
             reward += 0.04 * (visited_positions_count - self.max_visited_positions)
             self.max_visited_positions = visited_positions_count
        
        return reward
    
    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.pyboy.memory[addr] - min_poke_level, 0)
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]  # Endereços de memória dos Pokémon
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

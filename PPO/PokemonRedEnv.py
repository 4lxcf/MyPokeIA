import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy
from skimage.transform import downscale_local_mean, resize

N_DISCRETE_ACTIONS = 6 # Cima, baixo, direita, esquerda, a, b
N_CHANNELS = 3
MAX_STEPS = 2048*3

class PokemonRedEnv(gym.Env):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path)
        self.pyboy.set_emulation_speed(15)
        self.new_width = 80
        self.new_height = 72

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        #->OLD: self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([255, 255, 255]), dtype=np.uint8)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8),
            "max_visited_positions": spaces.Box(low=0, high=MAX_STEPS, shape=(1,), dtype=np.float32),
            "levels_sum": spaces.Box(low=0, high=600, shape=(1,), dtype=np.float32),
        })

    def reset(self, seed=None, options={}):
        # Se o seed for fornecido, defina a semente do ambiente
        if seed is not None:
            np.random.seed(seed)

        # Carregar o estado salvo no início do jogo
        with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)

        self.agent_stats = []
        self.visited_positions = []
        self.visited_areas = set()  # Conjunto para áreas já visitadas (map_ids)
        self.step_count = 0
        self.max_steps = MAX_STEPS
        self.max_levels = 0
        self.max_visited_positions = 0

        # Ler os valores dos três endereços de memória
        map_id = self.pyboy.memory[0xD35E]  # ID do mapa

        # Verifica se o agente já passou pela área (map_id)
        if map_id not in self.visited_areas:
            self.visited_areas.add(map_id)  # Marca a área como visitada

        observation = self.get_observation() # Atualiza a observation

        return observation, {}

    def step(self, action):
        self.make_action(action)
        self.update_agent_stats()

        self.step_count += 1

        observation = self.get_observation()
        reward = self.calculate_reward() 
        terminated = self.check_if_terminated()
        info = {f"Agent_Stats:": self.agent_stats} # Infos adicionais

        return observation, reward, terminated, False, info

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
        elif action == 5:
                self.pyboy.button_press("b")

        self.pyboy.tick(8) # Tick do emulador
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("right")
        self.pyboy.button_release("left")
        self.pyboy.button_release("a")
        self.pyboy.button_release("b")
    
    def update_agent_stats(self):
        map_id, pos_x, pos_y = (self.pyboy.memory[0xD35E], self.pyboy.memory[0xD362], self.pyboy.memory[0xD361])
        actual_position = (map_id, pos_x, pos_y)
        levels = [self.pyboy.memory[e] for e in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        if actual_position not in self.visited_positions:
            self.visited_positions.append(actual_position)
             
        self.agent_stats.append(
              {
                   "step": self.step_count,
                   "pos_x": pos_x,
                   "pos_y": pos_y,
                   "map": map_id,
                #    "visited_positions": self.visited_positions,
                   "max_visited_positions": self.max_visited_positions,
                   "visited_areas": self.visited_areas,
                   "levels": levels,
                   "levels_sum": self.get_levels_sum(),
              }
        )

    def check_if_terminated(self):
        is_Terminated = self.step_count >= self.max_steps
        return is_Terminated

    def calculate_reward(self):
        reward = 0
        level_sum = self.get_levels_sum()
        visited_positions_count = len(self.visited_positions)
        map_id = self.pyboy.memory[0xD35E]
        
        if level_sum > self.max_levels:
            reward += level_sum - self.max_levels
            self.max_levels = level_sum

        if visited_positions_count > self.max_visited_positions:
            reward += 0.04 * (visited_positions_count - self.max_visited_positions)
            self.max_visited_positions = visited_positions_count

        # Recompensa adicional por entrar em uma nova área (map_id)
        if map_id not in self.visited_areas:  # Verifica se o ID da área já foi visitado
            if map_id in locations_rewards:  # Verifica se o map_id existe em locations_rewards
                reward += locations_rewards[map_id]["reward"]  # Adiciona a recompensa da área
                self.visited_areas.add(map_id)  # Adiciona o ID da área à lista de áreas visitadas
                print(self.agent_stats)

        return reward
    
    def get_levels_sum(self):
        poke_levels = [
            self.pyboy.memory[addr]
            for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]  # Endereços de memória dos Pokémon
        ]
        return sum(poke_levels)
    
    def get_observation(self):
        screen_pixel = self.pyboy.screen.ndarray[:,:,0:1]

        resized_screen = downscale_local_mean(screen_pixel, (2, 2, 1)).astype(np.uint8)

        resized_screen = resize(resized_screen, (self.new_height, self.new_width), anti_aliasing=True)

        # Ajustar a tela para ter 3 canais, replicando o canal único
        # --> Só é NECESSÁRIO no LOAD do agente <--
        # resized_screen = np.repeat(resized_screen, 3, axis=-1)

        observation = {
            "screen": resized_screen,
            "max_visited_positions": np.array([self.max_visited_positions], dtype=np.float32),
            "levels_sum": np.array([self.get_levels_sum()], dtype=np.float32)
        }
        
        return observation
    
locations_rewards = {
    0: {"name": "Pallet Town", "reward": 0},
    12: {"name": "Route 1", "reward": 10},
    1: {"name": "Viridian City", "reward": 20},
    13: {"name": "Route 2", "reward": 30},
    51: {"name": "Viridian Forest", "reward": 40},
    2: {"name": "Pewter City", "reward": 50},
    14: {"name": "Route 3", "reward": 60},
    # 0x3C: {"name": "Mt. Moon", "reward": 70},
    15: {"name": "Route 4", "reward": 80},
    3: {"name": "Cerulean City", "reward": 90},
    35: {"name": "Route 24", "reward": 100},
    36: {"name": "Route 25", "reward": 110},
    # 0x10: {"name": "Route 5", "reward": 120},
    # 0x11: {"name": "Route 6", "reward": 130},
    # 0x04: {"name": "Vermilion City", "reward": 140},
    # 0x16: {"name": "Route 11", "reward": 150},
    # 0x14: {"name": "Route 9", "reward": 160},
    # 0x56: {"name": "Rock Tunnel", "reward": 170},
    # 0x15: {"name": "Route 10", "reward": 180},
    # 0x05: {"name": "Lavender Town", "reward": 190},
    # 0x13: {"name": "Route 8", "reward": 200},
    # 0x12: {"name": "Route 7", "reward": 210},
    # 0x06: {"name": "Celadon City", "reward": 220},
    # 0x0A: {"name": "Saffron City", "reward": 230},
    # 0x1B: {"name": "Route 16", "reward": 240},
    # 0x1C: {"name": "Route 17", "reward": 250},
    # 0x1D: {"name": "Route 18", "reward": 260},
    # 0x07: {"name": "Fuchsia City", "reward": 270},
    # 0x1A: {"name": "Route 15", "reward": 280},
    # 0x19: {"name": "Route 14", "reward": 290},
    # 0x18: {"name": "Route 13", "reward": 300},
    # 0x17: {"name": "Route 12", "reward": 310},
    # 0x1E: {"name": "Route 19", "reward": 320},
    # 0x1F: {"name": "Route 20", "reward": 330},
    # 0x5A: {"name": "Seafoam Islands", "reward": 340},
    # 0x08: {"name": "Cinnabar Island", "reward": 350},
    # 0x70: {"name": "Pokémon Mansion", "reward": 360},
    # 0x20: {"name": "Route 21", "reward": 370},
    33: {"name": "Route 22", "reward": 380},
    # 0x22: {"name": "Route 23", "reward": 390},
    # 0x61: {"name": "Victory Road", "reward": 400},
    # 0x09: {"name": "Indigo Plateau (Pokémon League)", "reward": 410}
}
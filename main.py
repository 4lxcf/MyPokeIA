from stable_baselines3 import DQN
import gymnasium as gym  # Alterado para gymnasium
from gymnasium import spaces
from pyboy import PyBoy
import numpy as np
from PIL import Image

class PokemonEnv(gym.Env):
    def __init__(self):
        super(PokemonEnv, self).__init__()
        self.pyboy = PyBoy('PokemonRed.gb')
        self.pyboy.set_emulation_speed(6)
        self.render_mode = 'human'
        
        self.action_space = spaces.Discrete(6) # Cima, baixo, esquerda, direita, a, b

        self.resized_width = 80
        self.resized_height = 72
        
        # Reduzir a resolução para economizar memória
        self.observation_space = spaces.Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8)  # Reduzido para metade da resolução original

        self.step_count = 0
        self.max_steps = 1000
        self.visited_positions = set()
        self.in_combat = False  # Flag para identificar se estamos em combate

    def reset(self, seed=None, options=None):
        # Adicione suporte ao argumento seed (se necessário)
        super().reset(seed=seed)
        self.render_mode = 'human'
        
        # Preparar o emulador carregando o estado do jogo pressionando a tecla Z
        self._initialize_game()

        self.step_count = 0
        self.max_steps = 1000
        self.visited_positions.clear()
        self.in_combat = False
        
        return self._get_state(), {}

    def step(self, action):
        if self._is_in_combat():
            self.in_combat = True
            reward = 0
        else:
            self.in_combat = False
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
            elif action == 5:
                self.pyboy.button("b")
            
            # Tick do emulador
            self.pyboy.tick()

            # Obter posição e atualizar posições visitadas
            current_position = self._get_position()
            # Calcula a recompensa
            reward = self._calculate_reward(current_position)
            self.visited_positions.add(current_position)

            # Incrementa o contador de passos se não estiver em combate
            self.step_count += 1 if not self.in_combat else 0


        # Verifica se o jogo terminou
        done = self.step_count >= self.max_steps
        state = self._get_state()

        # Adiciona um valor extra para o truncated
        truncated = done

        # Adiciona um dicionário info com informações adicionais
        info = {"TimeLimit.truncated": truncated}

        return state, reward, done, truncated, info

    def render(self, mode='human'):
        pass
        # image = self.pyboy.screen.image  # Captura a tela do emulador como uma imagem PIL
        # image = image.resize((self.resized_width, self.resized_height))  # Redimensiona a imagem
        # image.show()  # Mostra a imagem para visualização, pode ser substituído conforme necessário
        
    def _get_state(self):
        image = self.pyboy.screen.image  # Captura a tela do emulador como uma imagem PIL
        image = image.convert('RGB')  # Converte a imagem para o modo RGB
        image = image.resize((self.resized_width, self.resized_height))  # Redimensiona a imagem
        screen_array = np.array(image)  # Converte a imagem PIL para um array numpy
        return screen_array

    def _get_position(self):
        # Implementar a lógica para obter a posição do personagem
        x = self.pyboy.memory[0xD362]
        y = self.pyboy.memory[0xD361]
        return (x, y)

    def _is_in_combat(self):
        # Implementar a lógica para verificar se o personagem está em combate
        # Aqui é um exemplo de como você pode verificar se a tela mudou para o modo de combate
        image = self.pyboy.screen.image
        try:
            screen_array = np.array(image.convert('L'))  # Converte a imagem para escala de cinza
        except Exception as e:
            print(f"Erro ao processar a imagem: {e}")
            
        return np.mean(screen_array) < 50  # Exemplo simples; substitua pela lógica correta

    def _initialize_game(self):
        # Pressiona a tecla Z para carregar o estado do jogo
        with open('PokemonRed.gb.state', "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick()  # Avança o emulador para aplicar o input

    def _calculate_reward(self, position):
        # Se a posição é nova, recompensa positiva
        if position not in self.visited_positions:
            return 1.0  # Recompensa por visitar uma nova posição
        else:
            return -0.1  # Penalidade leve por revisitar uma posição conhecida


# Envolver o ambiente com Monitor e DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

env = Monitor(PokemonEnv())
env = DummyVecEnv([lambda: env])

# Treinar o modelo DQN
model = DQN('CnnPolicy', env, verbose=1, buffer_size=50000, batch_size=32)
model.learn(total_timesteps=5000)

# Usar o modelo treinado para jogar
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

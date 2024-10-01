from pyboy import PyBoy

pyboy = PyBoy('..\PokemonRed.gb')
pyboy.set_emulation_speed(9)
with open('D:\Dev\MyPokeIA\PokemonRed.gb.state', "rb") as f:
    pyboy.load_state(f)

while True:
    print(pyboy.memory[0xD35E])
        
    pyboy.tick()
    

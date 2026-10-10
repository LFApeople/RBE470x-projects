# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game
from monsters.stupid_monster import StupidMonster
from monsters.selfpreserving_monster import SelfPreservingMonster

sys.path.insert(1, '../teamNN')
from neuralnet import Agent

map_file = 'map.txt'
variant = int(sys.argv[1])
drill = sys.argv[2]
input_mode = int(sys.argv[3])
if drill:
    map_file = drill

# Keep one agent across episodes so replay memory and optimizer state survive.
agent = Agent("me", "C", 0, 0, mode=input_mode)
for epoch in range(1000):
    g = Game.fromfile(map_file)
    agent.training = input_mode
    agent.guided_turn = 0
    agent.x = 0
    agent.y = 0
    agent.dx = 0
    agent.dy = 0
    agent.maybe_place_bomb = False
    g.add_character(agent)

    if variant == 2 or variant == 5:
        g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 9      # position
        ))
    if variant == 3:
        g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
        ))
    if variant == 4 or variant == 5:
        g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
        ))

    # Run!
    g.go(1)

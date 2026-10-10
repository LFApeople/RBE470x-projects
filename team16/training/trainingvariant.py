# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

sys.path.insert(1, '../teamNN')
from neuralnet import Agent

map_file = 'map.txt'
DRILL = sys.argv[1]
input_mode = int(sys.argv[2])
# Drill selection
match DRILL:
    case "fence": # Short wall to bomb across
        map_file = 'fence_map.txt'

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

    # Run!
    g.go(1)

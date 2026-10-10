# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from neuralnet import Agent

DRILL = "fence"

# Drill selection
match DRILL:
    case "fence": # Short wall to bomb across
        map_file = 'map.txt'

epoch = 0

# Create the game
while epoch < 1000:
    g = Game.fromfile(map_file)
    g = Game.fromfile('map.txt')

    # TODO Add your character
    """ g.add_character(TestCharacter("me", # name
                                  "C",  # avatar
                                  0, 0  # position
    )) """

    g.add_character(Agent("me", # name
                                  "C",  # avatar
                                  0, 0,  # position
                                  2
    ))

    # Run!
    g.go(1)
    g = None

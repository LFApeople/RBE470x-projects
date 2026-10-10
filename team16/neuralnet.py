import os
from abc import ABC, abstractmethod
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from testcharacter import TestCharacter
import heapq
import sys

import collections
import math
import json
import os
import random
from collections import deque
from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back

import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

class Plot(ABC):
    def __init__(self, title='Training...'):
        self.title = title

    @abstractmethod
    def render(self, scores, mean_scores):
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        raise NotImplementedError


class TrainingPlot(Plot):
    def render(self, scores, mean_scores):
        scores = list(scores or [])
        mean_scores = list(mean_scores or [])

        if not scores and not mean_scores:
            return

        if len(mean_scores) < len(scores):
            last_mean = mean_scores[-1] if mean_scores else 0
            mean_scores.extend([last_mean] * (len(scores) - len(mean_scores)))

        # Ensure the chart always has a valid x-axis and produces a connected line.
        if not plt.get_fignums():
            plt.figure()
        plt.gcf().clear()

        x_scores = list(range(len(scores)))
        x_mean = list(range(len(mean_scores)))

        plt.title(self.title)
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(x_scores, scores, label='Score', alpha=0.9, linewidth=2)
        plt.plot(x_mean, mean_scores, label='Mean Score', alpha=0.9, linewidth=2)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.4)

        if scores:
            plt.text(len(scores) - 1, scores[-1], str(scores[-1]), va='bottom', fontsize=8)
        if mean_scores:
            plt.text(len(mean_scores) - 1, mean_scores[-1], str(round(mean_scores[-1], 2)), va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches='tight')


def plot(scores, mean_scores, title='Training...', save_path=None):
    plotter = TrainingPlot(title)
    plotter.render(scores, mean_scores)
    if save_path:
        plotter.save(save_path)


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent(CharacterEntity):
    
    def __init__(self, name, avatar, x, y):
        super().__init__(name, avatar, x, y)
        self.Test = TestCharacter(name, avatar, x, y)
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(43, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.training = True
        self._plot_scores = []
        self._plot_mean_scores = []
        self._total_score = 0
        self._record = -5000
        self._training_state_path = os.path.join(self.model.model_folder_path, 'training_state.json')
        self._load_training_state()

    def _normalize_state_value(self, value):
        if value is None:
            return 0
        if isinstance(value, (list, tuple, set, dict)):
            return 1 if len(value) > 0 else 0
        if hasattr(value, '__iter__') and not isinstance(value, (str, bytes, bytearray)):
            return 1
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    def get_state(self, wrld):

        
        # (-1, -1)
        valid__1__1 = self.Test.valid_spot(wrld, self.me.x - 1, self.me.y - 1)
        empty__1__1 = valid__1__1 and wrld.empty_at(self.me.x - 1, self.me.y - 1)
        wall__1__1 = valid__1__1 and wrld.wall_at(self.me.x - 1, self.me.y - 1)
        bomb__1__1 = valid__1__1 and wrld.bomb_at(self.me.x - 1, self.me.y - 1)
        explosion__1__1 = valid__1__1 and wrld.explosion_at(self.me.x - 1, self.me.y - 1)
        monster__1__1 = valid__1__1 and wrld.monsters_at(self.me.x - 1, self.me.y - 1)

        # (-1, 0)
        valid__1__0 = self.Test.valid_spot(wrld, self.me.x - 1, self.me.y)
        empty__1__0 = valid__1__0 and wrld.empty_at(self.me.x - 1, self.me.y)
        wall__1__0 = valid__1__0 and wrld.wall_at(self.me.x - 1, self.me.y)
        explosion__1__0 = valid__1__0 and wrld.explosion_at(self.me.x - 1, self.me.y)
        monster__1__0 = valid__1__0 and wrld.monsters_at(self.me.x - 1, self.me.y)

        # (-1, 1)
        valid__1_1 = self.Test.valid_spot(wrld, self.me.x - 1, self.me.y + 1)
        empty__1_1 = valid__1_1 and wrld.empty_at(self.me.x - 1, self.me.y + 1)
        wall__1_1 = valid__1_1 and wrld.wall_at(self.me.x - 1, self.me.y + 1)
        bomb__1_1 = valid__1_1 and wrld.bomb_at(self.me.x - 1, self.me.y + 1)
        explosion__1_1 = valid__1_1 and wrld.explosion_at(self.me.x - 1, self.me.y + 1)
        monster__1_1 = valid__1_1 and wrld.monsters_at(self.me.x - 1, self.me.y + 1)

        # (0, -1)
        valid__0__1 = self.Test.valid_spot(wrld, self.me.x, self.me.y - 1)
        empty__0__1 = valid__0__1 and wrld.empty_at(self.me.x, self.me.y - 1)
        wall__0__1 = valid__0__1 and wrld.wall_at(self.me.x, self.me.y - 1)
        explosion__0__1 = valid__0__1 and wrld.explosion_at(self.me.x, self.me.y - 1)
        monster__0__1 = valid__0__1 and wrld.monsters_at(self.me.x, self.me.y - 1)

        # (0, 1)
        valid__0_1 = self.Test.valid_spot(wrld, self.me.x, self.me.y + 1)
        empty__0_1 = valid__0_1 and wrld.empty_at(self.me.x, self.me.y + 1)
        wall__0_1 = valid__0_1 and wrld.wall_at(self.me.x, self.me.y + 1)
        explosion__0_1 = valid__0_1 and wrld.explosion_at(self.me.x, self.me.y + 1)
        monster__0_1 = valid__0_1 and wrld.monsters_at(self.me.x, self.me.y + 1)

        # (1, -1)
        valid_1__1 = self.Test.valid_spot(wrld, self.me.x + 1, self.me.y - 1)
        empty_1__1 = valid_1__1 and wrld.empty_at(self.me.x + 1, self.me.y - 1)
        wall_1__1 = valid_1__1 and wrld.wall_at(self.me.x + 1, self.me.y - 1)
        bomb_1__1 = valid_1__1 and wrld.bomb_at(self.me.x + 1, self.me.y - 1)
        explosion_1__1 = valid_1__1 and wrld.explosion_at(self.me.x + 1, self.me.y - 1)
        monster_1__1 = valid_1__1 and wrld.monsters_at(self.me.x + 1, self.me.y - 1)

        # (1, 0)
        valid_1__0 = self.Test.valid_spot(wrld, self.me.x + 1, self.me.y)
        empty_1__0 = valid_1__0 and wrld.empty_at(self.me.x + 1, self.me.y)
        wall_1__0 = valid_1__0 and wrld.wall_at(self.me.x + 1, self.me.y)
        explosion_1__0 = valid_1__0 and wrld.explosion_at(self.me.x + 1, self.me.y)
        monster_1__0 = valid_1__0 and wrld.monsters_at(self.me.x + 1, self.me.y)

        # (1, 1)
        valid_1_1 = self.Test.valid_spot(wrld, self.me.x + 1, self.me.y + 1)
        empty_1_1 = valid_1_1 and wrld.empty_at(self.me.x + 1, self.me.y + 1)
        wall_1_1 = valid_1_1 and wrld.wall_at(self.me.x + 1, self.me.y + 1)
        bomb_1_1 = valid_1_1 and wrld.bomb_at(self.me.x + 1, self.me.y + 1)
        explosion_1_1 = valid_1_1 and wrld.explosion_at(self.me.x + 1, self.me.y + 1)
        monster_1_1 = valid_1_1 and wrld.monsters_at(self.me.x + 1, self.me.y + 1)

        exit = TestCharacter.get_exit(self.Test, wrld)
        exitdx = exit[0] - self.me.x
        exitdy = exit[1] - self.me.y

        path = TestCharacter.astar_path_to_exit(self.Test, wrld, self.start, self.goal)
        if path and len(path) > 1:
            astarx, astary = path[1]
            pathToExit = len(path) - 1
        else:
            astarx, astary = self.me.x, self.me.y
            pathToExit = 0

        bomb_list = list(wrld.bombs.values())
        if bomb_list:
            bombx, bomby = self.me.x == bomb_list[0].x, self.me.y == bomb_list[0].y
        else:
            bombx, bomby = 0, 0

        state = [
            # (-1, -1)
            empty__1__1,
            wall__1__1,
            bomb__1__1,
            explosion__1__1,
            monster__1__1,

            # (-1, 0)
            empty__1__0,
            wall__1__0,
            explosion__1__0,
            monster__1__0,

            # (-1, 1)
            empty__1_1,
            wall__1_1,
            bomb__1_1,
            explosion__1_1,
            monster__1_1,

            # (0, -1)
            empty__0__1,
            wall__0__1,
            explosion__0__1,
            monster__0__1,

            # (0, 1)
            empty__0_1,
            wall__0_1,
            explosion__0_1,
            monster__0_1,

            # (1, -1)
            empty_1__1,
            wall_1__1,
            bomb_1__1,
            explosion_1__1,
            monster_1__1,

            # (1, 0)
            empty_1__0,
            wall_1__0,
            explosion_1__0,
            monster_1__0,

            # (1, 1)
            empty_1_1,
            wall_1_1,
            bomb_1_1,
            explosion_1_1,
            monster_1_1,

            exitdx,
            exitdy,
            astarx,
            astary,
            pathToExit,

            bombx,
            bomby,
        ]

        state = [self._normalize_state_value(v) for v in state]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def _load_training_state(self):
        if not os.path.isfile(self._training_state_path):
            return

        try:
            with open(self._training_state_path, 'r') as f:
                state = json.load(f)

            self.n_games = int(state.get('n_games', self.n_games))
            self._record = float(state.get('_record', self._record))
            self._plot_scores = state.get('_plot_scores', self._plot_scores)
            self._plot_mean_scores = state.get('_plot_mean_scores', self._plot_mean_scores)
            self._total_score = float(state.get('_total_score', self._total_score))
        except (TypeError, ValueError, OSError):
            self.n_games = 0
            self._record = 0
            self._plot_scores = []
            self._plot_mean_scores = []
            self._total_score = 0

    def _save_training_state(self):
        os.makedirs(os.path.dirname(self._training_state_path), exist_ok=True)
        state = {
            'n_games': self.n_games,
            '_record': self._record,
            '_plot_scores': self._plot_scores,
            '_plot_mean_scores': self._plot_mean_scores,
            '_total_score': self._total_score,
        }
        with open(self._training_state_path, 'w') as f:
            json.dump(state, f)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[0] = random.randint(-1,1)
            final_move[1] = random.randint(-1,1)
            if random.randint(0,4000) < self.epsilon:
                final_move[2] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        bomb_list = list(self.world.bombs.values())
        monster_list = list(self.world.monsters.values())
        path = TestCharacter.astar_path_to_exit(self.Test, self.world, self.start, self.goal)
        if bomb_list or:
            final_move[2] = 0
        elif not monster_list or path:
            final_move[2] = 0

        return final_move

    def do_action(self, action):
        if action[2]:
            self.place_bomb()

        self.move(action[0], action[1])
    
    def train(self, wrld):
        # do() calls this once per world step, so keep the game and statistics
        # on this agent rather than starting an unbounded loop on each call.
        state_old = self.get_state(wrld)
        final_move = self.get_action(state_old)
        self.do_action(final_move)
        score = wrld.scores["me"]
        
        # Use the sensed world to evaluate the immediate outcome of this move.
        _, events = SensedWorld.from_world(wrld).next()
        reward = -1
        game_over = False

        for event in events:
            if event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 100
                score += 2 * wrld.time
                game_over = True
            elif event.tpe in (Event.CHARACTER_KILLED_BY_MONSTER, Event.BOMB_HIT_CHARACTER):
                reward -= 100
                game_over = True
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 25
            elif event.tpe == Event.BOMB_HIT_WALL:
                reward += 5

        
        state_new = self.get_state(wrld)

        self.train_short_memory(state_old, final_move, reward, state_new, game_over)
        self.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            self.n_games += 1
            self.train_long_memory()

            if score > self._record:
                self._record = score
                self.model.save()
                
            print('Game', self.n_games, 'Score', score, 'Record', self._record)
            self._plot_scores.append(score)
            self._total_score += score
            self._plot_mean_scores.append(self._total_score / self.n_games)
            self._save_training_state()
            plot(self._plot_scores, self._plot_mean_scores)

    def do(self, wrld):
        self.world = wrld
        self.me = wrld.me(self)
        self.start = (self.me.x, self.me.y)
        self.goal = TestCharacter.get_exit(self.Test, wrld)
        model_file = os.path.join(self.model.model_folder_path, 'model.pth')
        if os.path.isfile(model_file):
            self.model.load()

        if self.training:
            self.train(wrld)
        else:
            state_old = self.get_state(wrld)
            self.action(state_old)

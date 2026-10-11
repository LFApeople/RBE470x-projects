import os
import tempfile
import uuid
from abc import ABC, abstractmethod
import torch
from testcharacter import TestCharacter
import json
import os
import random
from collections import deque
from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
import numpy as np
from model import DEVICE, Linear_QNet, QTrainer
import matplotlib.pyplot as plt

# This plot is of vital importance as it was the only thing I bothered to set-up that gave me information for how the model was doing during training. It will show each game's score and total running mean. I wanted more information to see inside of the model and tell how the network is doing: what weight's are important, proper sizing, are there useless features? But settled for this as the midpoint of effort while I was already extremely tired working on it.
plt.ion()

class TrainingPlot(ABC):
    def __init__(self, title='Training...'):
            self.title = title

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


# Some values to configure the memory and learning of the network. I had no way to really tell if they were good, so I looked at some other networks and took some approximations.
MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.001
PROGRESS_REWARD = 1.0
ACTION_SPACE = (
    (-1, -1, 0), (0, -1, 0), (1, -1, 0),
    (-1, 0, 0), (0, 0, 0), (1, 0, 0),
    (-1, 1, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1),
)

# Where all the magic happens
class Agent(CharacterEntity):

    # More variables in the configuration that could be changed.
    def __init__(self, name, avatar, x, y, mode=0):
        super().__init__(name, avatar, x, y)
        self.Test = TestCharacter(name, avatar, x, y)
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(34, 256, len(ACTION_SPACE)).to(DEVICE)
        model_file = os.path.join(self.model.model_folder_path, 'model.pth')
        if os.path.isfile(model_file):
            self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.training = mode # 0 = evaluation, 1 = mixed training, 2 = guided, 3 = teaching, 4 = self-training
        self._plot_scores = []
        self._plot_mean_scores = []
        # Training modes for guided and teaching were used in some drills, but got dropped by the end.
        self._total_score = 0
        self.bankedGames = [
            "bdsccccwddds",
            "bdscccbwddsd",
            "dbsaccccdwddds",
            "bsdcccbwddds",
            "bdsccccwddswds",
            "sdbawcccbdsddd",
            "bscdccbcwddds",
        ]
        self.guide_game = random.randint(0, len(self.bankedGames) - 1)
        self.guided_turn = 0
        self.guide_string = ""
        self._record = -5000
        self._training_state_path = os.path.abspath(
            os.path.join(self.model.model_folder_path, 'training_state.json')
        )
        self._load_training_state()

    # Keeping the values that go into state to be consistent even with trickery that is done to get them.
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

    # Using some functions from the Q approximate in Testcharacter to get information on the world
    def danger(self, wrld, x, y):
        if not self.Test.valid_spot(wrld, x, y):
            return 1
        return int(self.Test.is_threatened(wrld, (x, y)))

    def _bomb_threatens(self, wrld, bomb, x, y):
        return (x, y) in self.Test.blast_cells(wrld, bomb.x, bomb.y)

    def _can_escape_bomb(self, wrld):
        me = wrld.me(self)
        if me is None or any(bomb.owner == me for bomb in wrld.bombs.values()):
            return False
        return self.Test.bomb_escape_move(wrld, (me.x, me.y)) is not None

    # Some fuckery for maneul correction to take over and help when the Neural Network is dumb and tries to make really bad decisions.
    def _available_action_indices(self):
        available_actions = list(range(len(ACTION_SPACE) - 1))
        wrld = getattr(self, 'world', None)
        if wrld is None:
            return available_actions

        me = wrld.me(self)
        if me is None:
            return available_actions

        start = (me.x, me.y)
        safe_actions = [
            action_idx
            for action_idx in available_actions
            if self.Test.is_safe_step(
                wrld,
                start,
                (start[0] + ACTION_SPACE[action_idx][0],
                 start[1] + ACTION_SPACE[action_idx][1]),
            )
        ]
        if safe_actions:
            moving_actions = [
                action_idx
                for action_idx in safe_actions
                if ACTION_SPACE[action_idx][:2] != (0, 0)
            ]
            if moving_actions:
                safe_actions = moving_actions
            available_actions = safe_actions

        goal = self.Test.get_exit(wrld)
        if self.Test.plan_bomb(wrld, start, goal) is not None:
            available_actions.append(len(ACTION_SPACE) - 1)

        imminent_bombs = [bomb for bomb in wrld.bombs.values() if bomb.timer <= 1]
        if imminent_bombs:
            safe_actions = []
            for action_idx in available_actions:
                dx, dy, place_bomb = ACTION_SPACE[action_idx]
                if place_bomb:
                    continue
                x = max(0, min(wrld.width() - 1, me.x + dx))
                y = max(0, min(wrld.height() - 1, me.y + dy))
                if wrld.wall_at(x, y):
                    x, y = me.x, me.y
                if not any(
                    self._bomb_threatens(wrld, bomb, x, y)
                    for bomb in imminent_bombs
                ):
                    safe_actions.append(action_idx)
            if safe_actions:
                return safe_actions

        return available_actions

    def _action_for_index(self, action_idx, wrld=None):
        if action_idx != len(ACTION_SPACE) - 1:
            return ACTION_SPACE[action_idx]

        wrld = wrld or getattr(self, 'world', None)
        me = wrld.me(self) if wrld is not None else None
        if me is None:
            raise ValueError("A world and agent position are required for bomb actions")
        escape = self.Test.plan_bomb(
            wrld, (me.x, me.y), self.Test.get_exit(wrld)
        )
        if escape is None:
            raise ValueError("Bomb action selected without a safe, useful escape plan")
        return (escape[0], escape[1], 1)

    # All the features and information about the environment that is fed into the network to try and make it smart. Originall had more features but they were culled, still unsure about how many of these approach 0 effect after L1 Regularization; if I could see in the weighting then I probably would have removed more of these.
    def get_state(self, wrld):

        me = wrld.me(self) or self.me

        # (-1, -1)
        valid__1__1 = self.Test.valid_spot(wrld, me.x - 1, me.y - 1)
        empty__1__1 = valid__1__1 and wrld.empty_at(me.x - 1, me.y - 1)
        wall__1__1 = valid__1__1 and wrld.wall_at(me.x - 1, me.y - 1)
        danger__1__1 = self.danger(wrld, me.x - 1, me.y - 1)

        # (-1, 0)
        valid__1__0 = self.Test.valid_spot(wrld, me.x - 1, me.y)
        empty__1__0 = valid__1__0 and wrld.empty_at(me.x - 1, me.y)
        wall__1__0 = valid__1__0 and wrld.wall_at(me.x - 1, me.y)
        danger__1__0 = self.danger(wrld, me.x - 1, me.y)

        # (-1, 1)
        valid__1_1 = self.Test.valid_spot(wrld, me.x - 1, me.y + 1)
        empty__1_1 = valid__1_1 and wrld.empty_at(me.x - 1, me.y + 1)
        wall__1_1 = valid__1_1 and wrld.wall_at(me.x - 1, me.y + 1)
        danger__1_1 = self.danger(wrld, me.x - 1, me.y + 1)

        # (0, -1)
        valid__0__1 = self.Test.valid_spot(wrld, me.x, me.y - 1)
        empty__0__1 = valid__0__1 and wrld.empty_at(me.x, me.y - 1)
        wall__0__1 = valid__0__1 and wrld.wall_at(me.x, me.y - 1)
        danger__0__1 = self.danger(wrld, me.x, me.y - 1)

        # (0, 0)
        danger__0__0 = self.danger(wrld, me.x, me.y)

        # (0, 1)
        valid__0_1 = self.Test.valid_spot(wrld, me.x, me.y + 1)
        empty__0_1 = valid__0_1 and wrld.empty_at(me.x, me.y + 1)
        wall__0_1 = valid__0_1 and wrld.wall_at(me.x, me.y + 1)
        danger__0_1 = self.danger(wrld, me.x, me.y + 1)

        # (1, -1)
        valid_1__1 = self.Test.valid_spot(wrld, me.x + 1, me.y - 1)
        empty_1__1 = valid_1__1 and wrld.empty_at(me.x + 1, me.y - 1)
        wall_1__1 = valid_1__1 and wrld.wall_at(me.x + 1, me.y - 1)
        danger_1__1 = self.danger(wrld, me.x + 1, me.y - 1)

        # (1, 0)
        valid_1__0 = self.Test.valid_spot(wrld, me.x + 1, me.y)
        empty_1__0 = valid_1__0 and wrld.empty_at(me.x + 1, me.y)
        wall_1__0 = valid_1__0 and wrld.wall_at(me.x + 1, me.y)
        danger_1__0 = self.danger(wrld, me.x + 1, me.y)

        # (1, 1)
        valid_1_1 = self.Test.valid_spot(wrld, me.x + 1, me.y + 1)
        empty_1_1 = valid_1_1 and wrld.empty_at(me.x + 1, me.y + 1)
        wall_1_1 = valid_1_1 and wrld.wall_at(me.x + 1, me.y + 1)
        danger_1_1 = self.danger(wrld, me.x + 1, me.y + 1)

        exit = self.Test.get_exit(wrld)
        if exit is None:
            exitdx = exitdy = distance_to_exit = 0
            path = []
        else:
            exitdx = exit[0] - me.x
            exitdy = exit[1] - me.y
            distance_to_exit = self.Test.exit_distance((me.x, me.y), exit)
            path = self.Test.astar(wrld, (me.x, me.y), exit)
        if path and len(path) > 1:
            astarx, astary = path[1]
            pathToExit = len(path) - 1
        else:
            astarx, astary = me.x, me.y
            pathToExit = 0

        bomb_list = list(wrld.bombs.values())
        if bomb_list:
            bombx, bomby = me.x == bomb_list[0].x, me.y == bomb_list[0].y
            bomb = 1
        else:
            bombx, bomby, bomb = 0, 0, 0

        # Combine all the information gathered into one really long list
        state = [
            exitdx,
            exitdy,
            distance_to_exit,
            astarx,
            astary,
            pathToExit,

            bombx,
            bomby,
            bomb,

            # (-1, -1)
            empty__1__1,
            wall__1__1,
            danger__1__1,

            # (-1, 0)
            empty__1__0,
            wall__1__0,
            danger__1__0,

            # (-1, 1)
            empty__1_1,
            wall__1_1,
            danger__1_1,

            # (0, 0)
            danger__0__0,

            # (0, -1)
            empty__0__1,
            wall__0__1,
            danger__0__1,

            # (0, 1)
            empty__0_1,
            wall__0_1,
            danger__0_1,

            # (1, -1)
            empty_1__1,
            wall_1__1,
            danger_1__1,

            # (1, 0)
            empty_1__0,
            wall_1__0,
            danger_1__0,

            # (1, 1)
            empty_1_1,
            wall_1_1,
            danger_1_1,
        ]

        state = [self._normalize_state_value(v) for v in state]
        return np.array(state, dtype=int)

    # Add to replay buffer
    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    # Get training data that is used for the graphs and plotting. My only real insight into how the network is doing.
    def _load_training_state(self):
        state_dir = os.path.dirname(self._training_state_path)
        checkpoint_prefix = f"{os.path.basename(self._training_state_path)}.checkpoint-"
        checkpoint_paths = [
            os.path.join(state_dir, candidate)
            for candidate in os.listdir(state_dir)
            if candidate.startswith(checkpoint_prefix) and candidate.endswith('.json')
        ] if os.path.isdir(state_dir) else []
        available_paths = [
            path
            for path in [self._training_state_path, *checkpoint_paths]
            if os.path.isfile(path)
        ]
        if not available_paths:
            return

        try:
            state_path = max(available_paths, key=os.path.getmtime)
            with open(state_path, 'r', encoding='utf-8') as f:
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

    # Saves training data. I orinally set this up because the agent was being re-initialized every game, but then figured out how to configure training to be smarted. Kept this sso that I could have one plott across multiple tests as the model learned.
    def _save_training_state(self):
        state_dir = os.path.dirname(self._training_state_path)
        os.makedirs(state_dir, exist_ok=True)
        state = {
            'n_games': self.n_games,
            '_record': self._record,
            '_plot_scores': self._plot_scores,
            '_plot_mean_scores': self._plot_mean_scores,
            '_total_score': self._total_score,
        }
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(self._training_state_path)}.",
            suffix=".tmp",
            dir=state_dir,
        )

        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as f:
                json.dump(state, f)
            try:
                os.replace(temporary_path, self._training_state_path)
            except OSError as error:
                if getattr(error, 'winerror', None) not in (5, 1224):
                    raise
                checkpoint_path = os.path.join(
                    state_dir,
                    f"{os.path.basename(self._training_state_path)}.checkpoint-{uuid.uuid4().hex}.json",
                )
                os.replace(temporary_path, checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def train_long_memory(self):
        if not self.memory:
            return
        mini_sample = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)


    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    # Use the network to convert state into q value for action
    def get_action(self, state):
        action_idx = self._get_action_index(state)
        return self._action_for_index(action_idx)

    # Used to encourage the bot to be go for the exit when there is no monsters.
    def _no_monster_action_index(self, wrld, available_actions):
        if self.Test.monster_list(wrld):
            return None

        me = wrld.me(self)
        goal = self.Test.get_exit(wrld)
        if me is None or goal is None:
            return None

        start = (me.x, me.y)
        bomb_action = len(ACTION_SPACE) - 1
        if bomb_action in available_actions:
            return bomb_action

        target = None
        if not wrld.bombs:
            target = self.Test.breach_move(wrld, start, goal)
        if target is None:
            path = self.Test.astar(wrld, start, goal)
            if path and len(path) > 1:
                target = path[1]
            else:
                target = goal
                if wrld.bombs and not self.Test.is_threatened(wrld, start):
                    return ACTION_SPACE.index((0, 0, 0))

        safe_actions = [
            action_idx for action_idx in available_actions
            if action_idx != bomb_action
            and self.Test.is_safe_step(
                wrld,
                start,
                (start[0] + ACTION_SPACE[action_idx][0],
                 start[1] + ACTION_SPACE[action_idx][1]),
            )
        ]
        candidates = safe_actions or [
            action_idx for action_idx in available_actions
            if action_idx != bomb_action
        ]
        if not candidates:
            return None

        unthreatened = []
        for action_idx in candidates:
            dx, dy, _ = ACTION_SPACE[action_idx]
            destination = (start[0] + dx, start[1] + dy)
            if not self.Test.is_threatened(wrld, destination):
                unthreatened.append(action_idx)
        if unthreatened:
            candidates = unthreatened

        def route_cost(action_idx):
            dx, dy, _ = ACTION_SPACE[action_idx]
            destination = (
                max(0, min(wrld.width() - 1, start[0] + dx)),
                max(0, min(wrld.height() - 1, start[1] + dy)),
            )
            if wrld.wall_at(*destination):
                destination = start
            path = self.Test.astar(wrld, destination, target)
            if path:
                return len(path)
            return abs(destination[0] - target[0]) + abs(destination[1] - target[1])

        return min(candidates, key=route_cost)

    # Even with this the network is very scared of monsters that can't touch it. Tried to encourage aggressive behavior
    def _trapped_monster_action_index(self, wrld, available_actions):
        trapped = self.Test.trapped_monsters(wrld)
        me = wrld.me(self)
        goal = self.Test.get_exit(wrld)
        if not trapped or me is None or goal is None or wrld.bombs:
            return None

        start = (me.x, me.y)
        targets = {(monster.x, monster.y) for monster in trapped}
        blast_action = len(ACTION_SPACE) - 1
        bomb_sites = []
        candidate_sites = set()
        for target_x, target_y in targets:
            candidate_sites.add((target_x, target_y))
            for dx, dy in self.Test.CARDINALS:
                for distance in range(1, getattr(wrld, "expl_range", 4) + 1):
                    site = (target_x + dx * distance, target_y + dy * distance)
                    if (not self.Test.in_bounds(wrld, *site)
                            or wrld.wall_at(*site)):
                        break
                    candidate_sites.add(site)

        for site in candidate_sites:
            if site != start and not self.Test.valid_spot(wrld, *site):
                continue
            hit_targets = targets & self.Test.blast_cells(wrld, *site)
            if not hit_targets or self.Test.plan_bomb(wrld, site, goal) is None:
                continue
            path = self.Test.astar(wrld, start, site)
            if path:
                bomb_sites.append((len(path), -len(hit_targets), site))

        if not bomb_sites:
            return None

        _, _, site = min(bomb_sites)
        if site == start:
            return blast_action if blast_action in available_actions else None

        safe_actions = [
            action_idx for action_idx in available_actions
            if action_idx != blast_action
            and self.Test.is_safe_step(
                wrld,
                start,
                (start[0] + ACTION_SPACE[action_idx][0],
                 start[1] + ACTION_SPACE[action_idx][1]),
            )
        ]
        if not safe_actions:
            return None

        def route_cost(action_idx):
            dx, dy, _ = ACTION_SPACE[action_idx]
            destination = (start[0] + dx, start[1] + dy)
            path = self.Test.astar(wrld, destination, site)
            return len(path) if path else float("inf")

        action_idx = min(safe_actions, key=route_cost)
        return action_idx if route_cost(action_idx) != float("inf") else None

    def _avoid_monster_idle(self, wrld, action_idx):
        monsters = self.Test.monster_list(wrld)
        me = wrld.me(self)
        if not monsters or me is None or action_idx == len(ACTION_SPACE) - 1:
            return action_idx

        start = (me.x, me.y)
        dx, dy, _ = ACTION_SPACE[action_idx]
        destination = (start[0] + dx, start[1] + dy)
        selected_moves = (dx, dy) != (0, 0)
        selected_is_legal = (
            selected_moves
            and self.Test.in_bounds(wrld, *destination)
            and self.Test.valid_spot(wrld, *destination)
        )
        if selected_is_legal:
            return action_idx

        escape_actions = []
        for candidate_idx, (move_x, move_y, place_bomb) in enumerate(ACTION_SPACE[:-1]):
            if place_bomb or (move_x, move_y) == (0, 0):
                continue
            candidate = (start[0] + move_x, start[1] + move_y)
            if (not self.Test.in_bounds(wrld, *candidate)
                    or not self.Test.valid_spot(wrld, *candidate)):
                continue
            clearance = self.Test.monster_clearance(wrld, candidate)
            threatened = self.Test.is_threatened(wrld, candidate)
            escape_actions.append(
                (clearance, not threatened, candidate_idx)
            )

        if escape_actions:
            return max(escape_actions)[2]
        return action_idx

    def _get_action_index(self, state):
        available_actions = self._available_action_indices()
        wrld = getattr(self, 'world', None)
        if wrld is not None:
            trapped_action = self._trapped_monster_action_index(
                wrld, available_actions
            )
            if trapped_action is not None:
                return trapped_action

            goal_action = self._no_monster_action_index(
                wrld, available_actions
            )
            if goal_action is not None:
                return goal_action

        if self.training == 4:
            self.epsilon = max(0.05, 0.40 * (0.995 ** self.n_games))
            if random.random() < self.epsilon:
                action_idx = random.choice(available_actions)
                if wrld is not None:
                    return self._avoid_monster_idle(wrld, action_idx)
                return action_idx
        else:
            self.epsilon = 0.0

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor)
            unavailable_actions = set(range(len(ACTION_SPACE))) - set(available_actions)
            if unavailable_actions:
                q_values[list(unavailable_actions)] = -torch.inf
            action_idx = int(q_values.argmax().item())
        if wrld is not None:
            return self._avoid_monster_idle(wrld, action_idx)
        return action_idx

    # Actually tell character to take the action found
    def do_action(self, action):
        if action[2]:
            self.place_bomb()

        self.move(action[0], action[1])

    # Tries to keep a handle on what the bot is doing, if it is going to die or have game end, and give it proper incentives. Another area that could've been improved since the learning was so poor.
    def scoring(self, wrld, action):
        score = wrld.scores["me"]
        next_wrld = SensedWorld.from_world(wrld)
        next_wrld.me = next_wrld.me(self)
        next_wrld.me.maybe_place_bomb = bool(action[2])
        next_wrld.me.move(action[0], action[1])
        next_wrld, events = next_wrld.next()
        reward = -1
        game_over = False

        goal = self.Test.get_exit(wrld)
        me = wrld.me(self)
        next_me = next_wrld.me(self)
        if goal is not None and me is not None and next_me is not None:
            path_before = self.Test.astar(wrld, (me.x, me.y), goal)
            path_after = self.Test.astar(
                next_wrld, (next_me.x, next_me.y), goal
            )
            if path_before and path_after:
                distance_before = len(path_before) - 1
                distance_after = len(path_after) - 1
                reward += PROGRESS_REWARD * (distance_before - distance_after)

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

        state_new = self.get_state(next_wrld)
        return state_new, reward, score, game_over

    # When the character detects that the game will end next turn it saves the current data before it is too late.
    def game_over(self, score):
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
        
    # Core loop
    def trainself(self, wrld):
        state_old = self.get_state(wrld)
        action_idx = self._get_action_index(state_old)
        action = self._action_for_index(action_idx, wrld)
        self.do_action(action)
        state_new, reward, score, game_over = self.scoring(wrld, action)

        self.train_short_memory(state_old, action_idx, reward, state_new, game_over)
        self.remember(state_old, action_idx, reward, state_new, game_over)

        if game_over:
            self.game_over(score)

    def trainguided(self, wrld):
        
        state_old = self.get_state(wrld)
        dx, dy = 0, 0
        bomb = False

        # Handle input
        inp = self.bankedGames[self.guide_game][self.guided_turn]
        for c in inp:
            if 'w' == c:
                dy -= 1
            if 'a' == c:
                dx -= 1
            if 's' == c:
                dy += 1
            if 'd' == c:
                dx += 1
            if 'b' == c:
                bomb = True
        self.move(dx, dy)
        #print(f"Guided move: {inp} -> dx: {dx}, dy: {dy}, bomb: {bomb}")
            
        if bomb:
            self.place_bomb()
        action = (dx, dy, 1 if bomb else 0)
        action_idx = ACTION_SPACE.index(action)
        state_new, reward, score, game_over = self.scoring(wrld, action)

        self.train_short_memory(state_old, action_idx, reward, state_new, game_over)
        self.remember(state_old, action_idx, reward, state_new, game_over)
        self.guided_turn += 1

        if game_over:
            self.game_over(score)
            self.guide_game = (self.guide_game + 1) % len(self.bankedGames)
            self.guided_turn = 0
        elif self.guided_turn >= len(self.bankedGames[self.guide_game]):
            self.guide_game = (self.guide_game + 1) % len(self.bankedGames)
            self.guided_turn = 0

        # Commands

    def recordguided(self, wrld):   
        dx, dy = 0, 0
        bomb = False
        inp = input("How would you like to move (w=up,a=left,s=down,d=right,b=bomb)?")   
        for c in inp:
            if 'w' == c:
                dy -= 1
            if 'a' == c:
                dx -= 1
            if 's' == c:
                dy += 1
            if 'd' == c:
                dx += 1
            if 'b' == c:
                bomb = True
        self.move(dx, dy)

        if bomb:
            self.place_bomb()

        action = (dx, dy, 1 if bomb else 0)
        state_new, reward, score, game_over = self.scoring(wrld, action)
        self.guide_string = self.guide_string + inp

        if game_over:
            print(self.guide_string)

    def do(self, wrld):
        self.world = wrld
        self.me = wrld.me(self)
        self.start = (self.me.x, self.me.y)
        self.goal = self.Test.get_exit(wrld)

        # Selection of mode, sorta defunct. 0-just use current model | 1-try to have the model train itself | 2-Follow guided paths and learn from it; paired with 3-manuelly play the game and get a string of that game to teach the bot in 2
        if self.training == 1:
            self.trainself(wrld)
        elif self.training == 2:
            self.trainguided(wrld)
        elif self.training == 3:
            self.recordguided(wrld)
        else:
            state_old = self.get_state(wrld)
            action_idx = self._get_action_index(state_old)
            action = self._action_for_index(action_idx, wrld)
            self.do_action(action)

# This is necessary to find the main code
import heapq
import sys

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self,capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)

class TestCharacter(CharacterEntity):
    # List of standard moves that can be made by any entity at any time
    moves = [(-1, -1), (0, -1), (1, -1), (-1,  0), (0, 0), (1,  0), (-1,  1), (0,  1), (1,  1)]

    weights = [1.0, -1.0, -1.0]          # exit, monster, bomb
    alpha, gamma, epsilon = 0.1, 0.9, 0.1
    prev_features = None
    prev_q = None
    prev_exit_dist = None
    safety_margin = 1.0
    
    # Check if a spot can be moved into by the character
    def valid_spot(self, wrld, x, y):
        if not (0 <= x < wrld.width() and 0 <= y < wrld.height()):
            return False
        return wrld.empty_at(x, y) or wrld.exit_at(x, y)

    # Get all empty spaces neighboring a position
    def get_neighbors(self, wrld, current):
        x, y = current
        neighbors = []

        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if self.valid_spot(wrld, nx, ny):
                neighbors.append(((nx, ny), 1))
        return neighbors

    # Find what cells an explosion will affect
    def blast_cells(self, wrld, bx, by):
        """Cells a bomb at (bx, by) will hit (stops at walls)."""
        rng = getattr(wrld, "expl_range", 4)
        cells = {(bx, by)}
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, rng + 1):
                x, y = bx + dx * i, by + dy * i
                if not (0 <= x < wrld.width() and 0 <= y < wrld.height()):
                    break
                if wrld.wall_at(x, y):
                    break
                cells.add((x, y))
        return cells

    #helper funtion for finding the damger of cells

    def build_danger(self, wrld):
        """
        Returns:
          blast_time: {cell: [(start_tick, end_tick), ...]} windows when cell is deadly
          monsters:   list of (x, y)
        """
        duration = getattr(wrld, "expl_duration", 2)
        blast_time = {}

        # Bombs: deadly from tick `timer` for `duration` ticks
        for bomb in wrld.bombs.values():
            for cell in self.blast_cells(wrld, bomb.x, bomb.y):
                blast_time.setdefault(cell, []).append(
                    (bomb.timer, bomb.timer + duration))

        # Already-active explosions: deadly right now until they expire
        for expl in wrld.explosions.values():
            blast_time.setdefault((expl.x, expl.y), []).append((0, expl.timer))

        monsters = [(m.x, m.y) for ms in wrld.monsters.values() for m in ms]
        return blast_time, monsters

    #helper for if a cell is deadly to avoid

    def cell_deadly(self, cell, t, blast_time, monsters):
        for start, end in blast_time.get(cell, []):
            if start <= t < end:
                return True
        radius = 1 + int(self.safety_margin)   # monsters assumed to close in
        for mx, my in monsters:
            if max(abs(cell[0] - mx), abs(cell[1] - my)) <= radius - t // 2 * 0:
                return True
        return False

    def is_threatened(self, wrld, pos, horizon=4):
        """True if standing still at pos gets you hurt within `horizon` ticks."""
        blast_time, monsters = self.build_danger(wrld)
        return any(self.cell_deadly(pos, t, blast_time, monsters)
                   for t in range(horizon + 1))

    # Find any monsters adjacent to current position
    def get_neighbors_monsters(self, wrld, current):
        x, y = current
        monsters = []

        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
                if wrld.monsters_at(nx, ny):
                    monsters.append(((nx, ny), 1))
        return monsters

    # Primary golden path to exit with Astar
    def astar_path_to_exit(self, wrld, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        
        came_from = {}
        cost_so_far = {start: 0}
        counter = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)

            if current == goal:
                path = []
                curr = current
                while curr != start:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                return path

            for next_node, step_cost in self.get_neighbors(wrld, current):
                new_cost = cost_so_far[current] + step_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    # Astar priorit based upon cost to get their + monster hindrance + heuristic (max(abs(next_node[0] - goal[0]), abs(next_node[1] - goal[1])))
                    new_cost += len(
                        self.get_neighbors_monsters(
                            wrld, (next_node[0], next_node[1])
                        )
                    )
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + max(abs(next_node[0] - goal[0]), abs(next_node[1] - goal[1]))
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, next_node))
                    came_from[next_node] = current
        return path


    
    # Primary control of movement through expectimax
    # Takes Astar path as the main movement but allows further deviation to avoid monsters
    def expectimax_move(self, wrld, start, goal, depth=3):
        path = self.astar_path_to_exit(wrld, start, goal)
        next_pos = path[1] if path and len(path) > 1 else None

        # Get all monsters and their positions
        monsters = []
        for objects in wrld.monsters.values():
            for monster in objects:
                monsters.append((monster.x, monster.y))

        # Used for checking all possible choices that a monster can make
        def monster_choices(position):
            choices = [position]
            x, y = position
            for dx, dy in self.moves:
                nx, ny = x + dx, y + dy
                if self.valid_spot(wrld, nx, ny):
                    choices.append((nx, ny))
            return choices

        # Try to quantify the danger of any path with a slight preference for staying on the current route
        def danger(position, monster_positions):
            dang = 0
            for monster in monster_positions:
                distance = min(abs(position[0] - monster[0]), abs(position[1] - monster[1]))
                dang = max(dang, 3 - distance)
            distance_to_exit = max(abs(position[0] - goal[0]), abs(position[1] - goal[1]))
            route_penalty = 0 if next_pos == position else 2
            return dang + distance_to_exit + route_penalty

        # Check all positions that monsters can move into and their danger scores
        def chance(position, monster_positions, turns):
            if turns == 0 or not monster_positions:
                return danger(position, monster_positions)
            outcomes = [[]]
            for monster in monster_positions:
                outcomes = [prefix + [next_position]
                            for prefix in outcomes
                            for next_position in monster_choices(monster)]
                if len(outcomes) > 256:
                    outcomes = outcomes[:256]
            return sum(danger(position, outcome) for outcome in outcomes) / len(outcomes)


        # Actually use all the functions to try and find the best move
        candidates = []
        for dx, dy in self.moves:
            position = (start[0] + dx, start[1] + dy)
            if self.valid_spot(wrld, *position):
                candidates.append((chance(position, monsters, depth - 1), dx, dy))
        return candidates

    # Needed to actually find where the goal is
    def get_exit(self, wrld):
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None


    #Q learning Implimation helper functions

    def features(self, wrld, pos, goal):
        f_exit = 1.0 / (1 + self.exit_distance(wrld, pos, goal))

        m_d = [max(abs(pos[0] - m.x), abs(pos[1] - m.y))
                for ms in wrld.monsters.values() for m in ms]
        f_monster = 1.0 / (1 + min(m_d)) if m_d else 0.0

        b_d = [abs(pos[0] - b.x) + abs(pos[1] - b.y)
                for bs in wrld.bombs.values() for b in bs]
        f_bomb = 1.0 / (1 + min(b_d)) if b_d else 0.0

        return [f_exit, f_monster, f_bomb]

    def Q_value(self, features, weights=None):
        if weights is None:
            weights = self.weights
        return sum(w * f for w, f in zip(weights, features))

    # Evaluate every legal move from pos -> list of (q, (dx, dy), feats)
    def action_values(self, wrld, pos, goal):
        results = []
        for dx, dy in self.moves:
            p = (pos[0] + dx, pos[1] + dy)
            if (dx, dy) == (0, 0) or self.valid_spot(wrld, *p):
                feats = self.features(wrld, p, goal)
                results.append((self.q_value(feats), (dx, dy), feats))
        return results

    def reward(self, wrld, pos, goal):
        dist = self.exit_distance(wrld, pos, goal)
        r = (self.prev_exit_dist - dist) if self.prev_exit_dist is not None else 0
        if self.get_neighbors_monsters(wrld, pos):
            r -= 5          # adjacent to a monster
        if pos == goal:
            r += 100
        return r, dist

    def update_q(self, reward, next_features=None, terminal=False):
        if self.previous_features is None or self.previous_action is None:
            return
        old_q = self.q_value(self.previous_features)
        future = 0.0 if terminal or next_features is None else self.Q_value(next_features)
        delta = (reward + self.gamma * next_best_q) - self.prev_q
        self.weights = [w + self.alpha * delta * f
                        for w, f in zip(self.weights, self.prev_features)]


    def select_action(self, wrld, start, goal):
        # Get the current state features
        features = self.get_features(wrld, start, goal)
        q_values = self.get_q_values(features)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.moves) - 1)
        else:
            action_index = torch.argmax(q_values).item()

        return action_index

    #escape move to avoid danger can be  mreo itnegrated later 
    def escape_move(self, wrld, start, max_steps=10):
        """
        BFS over (cell, tick). Finds the shortest sequence of moves to a cell
        that is out of every blast zone and away from monsters.
        Returns (dx, dy), or None if already safe / nothing reachable.
        """
        blast_time, monsters = self.build_danger(wrld)

        def passable(x, y):
            return (0 <= x < wrld.width() and 0 <= y < wrld.height()
                    and not wrld.wall_at(x, y)
                    and not wrld.bombs_at(x, y))

        def permanently_safe(cell):
            # outside every blast zone and not near a monster
            if cell in blast_time:
                return False
            return not self.cell_deadly(cell, 0, {}, monsters)

        if permanently_safe(start) and not self.is_threatened(wrld, start):
            return None

        queue = deque([(start, 0, None)])        # (cell, tick, first_move)
        seen = {(start, 0)}
        fallback = None                          # best "survive a bit longer" move

        while queue:
            (x, y), t, first = queue.popleft()

            if first is not None and permanently_safe((x, y)):
                return first

            if t >= max_steps:
                continue

            for dx, dy in self.moves:            # includes (0, 0) = wait
                nx, ny = x + dx, y + dy
                nt = t + 1
                if not passable(nx, ny) or ((nx, ny), nt) in seen:
                    continue
                if self.cell_deadly((nx, ny), nt, blast_time, monsters):
                    continue
                seen.add(((nx, ny), nt))
                move = first if first is not None else (dx, dy)
                if fallback is None and nt >= 3:
                    fallback = move
                queue.append(((nx, ny), nt, move))

        return fallback

    # Primary function where information is given to the expectimax algorithm, runs it, and declares the move
    def do(self, wrld):
        me = wrld.me(self)
        start = (me.x, me.y)
        goal = self.get_exit(wrld)

        options = self.expectimax_move(wrld, start, goal)
        options.sort()
            
        self.move(options[0][1], options[0][2])


        #initialize replay memory
        D = ReplayMemory(10000)

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50
        #initialize action-value function Q with random weights

        #initialize target value function Q' with weights θ− = θ
        
        for i in range(num_episodes):
            #initialize sequence s1 = {x1} and preprocessed sequence φ1 = φ(s1)
            for t in count():
                action = select_action(wrld, start, goal)
                #with probability ε select a random action at
                #otherwise select at = argmaxa Q(φ(st), a; θ)
                #execute action at in emulator and observe reward rt and image xt+1
                #set st+1 = st, at, xt+1 and preprocess φt+
                #store transition (φt, at, rt, φt+1) in D
                #sample random minibatch of transitions (φj, aj, rj, φj+1) from D
                #set yj = { rj for terminal φj+1
                #otherwise set yj = rj + γ maxa' Q'(φj+1, a'; θ−)
                #perform a gradient descent step on (yj − Q(φj, aj; θ))2 with respect to the network parameters θ
                #Every C steps reset Q' = Q
        

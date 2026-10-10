# This is necessary to find the main code
import heapq
import sys

import collections
import math
import json
import os
import random
from collections import deque

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back

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


    # Q-learning configuration
    FEATURE_NAMES = (
        "bias", "progress", "monster_d", "adj_monsters",
        "safe_neighbors", "bomb_d", "explosion_d", "exit"
    )
    INITIAL_WEIGHTS = [0.0, 2.0, -5.0, -4.0, 0.6, -8.0, -12.0, 20.0]
    
    ALPHA = 0.08      # learning rate
    GAMMA = 0.90      # discount
    EPSILON = 0.03    # exploration rate

    WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Qweights.json")

    safety_margin = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = dict(zip(self.FEATURE_NAMES, self.INITIAL_WEIGHTS))
        self.previous_features = None
        self.previous_action = None
        self.rng = random.Random()
        self.load_weights()

    #load weights from file if they exist
    def load_weights(self):
        try:
            with open(self.WEIGHTS_FILE, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            for name in self.FEATURE_NAMES:
                if name in saved and isinstance(saved[name], (int, float)):
                    self.weights[name] = float(saved[name])
        except (OSError, ValueError, TypeError):
            pass

    def save_weights(self):
        try:
            temporary = self.WEIGHTS_FILE + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self.weights, handle, indent=2, sort_keys=True)
            os.replace(temporary, self.WEIGHTS_FILE)
        except OSError:
            pass  # still run in read-only environments


    # Q-learning
    def reward_events(self, wrld):
        reward = -0.05  # small per-step cost encourages reaching the exit
        for event in getattr(wrld, "events", []):
            if event.tpe in (Event.BOMB_HIT_CHARACTER,
                             Event.CHARACTER_KILLED_BY_MONSTER):
                reward -= 100.0
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 100.0
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 2.0
        return reward
    
    def Q_value(self, features):
        return sum(self.weights[name] * features.get(name, 0.0) for name in self.FEATURE_NAMES)

    def update_Q(self, reward, next_features=None, terminal=False):
        if self.previous_features is None or self.previous_action is None:
                    return
        old_q = self.q_value(self.previous_features)
        future = 0.0 if terminal or next_features is None else self.q_value(next_features)
        td_error = reward + self.GAMMA * future - old_q
        for name in self.FEATURE_NAMES:
            self.weights[name] += self.ALPHA * td_error * self.previous_features.get(name, 0.0)
        self._save_weights()

    def features(self, wrld, start, destination, goal):
        distance_before = self.exitdistance(start, goal)
        distance_after = self.exitdistance(destination, goal)
        progress = max(-1.0, min(1.0, distance_before - distance_after))
        monsters = self.monsters(wrld)
        nearest = 1.0
        adjacent = 0.0
        for monster in monsters:
            distance = max(abs(destination[0] - monster.x),
                           abs(destination[1] - monster.y))
            nearest = min(nearest, min(distance / 5.0, 1.0))
            if distance <= 1:
                adjacent += 1.0
        safe_neighbors = sum(
            1 for pos in self.neighbors(wrld, destination)
            if not wrld.bomb_at(pos[0], pos[1])
            and not wrld.explosion_at(pos[0], pos[1])
        )
        bomb_danger = 1.0 if wrld.bomb_at(*destination) else 0.0
        explosion_danger = 1.0 if wrld.explosion_at(*destination) else 0.0
        return {
            "bias": 1.0,
            "progress": progress,
            "monster_d": 1.0 - nearest if monsters else 0.0,
            "adj_monsters": min(adjacent, 3.0),
            "safe_neighbors": min(safe_neighbors / 8.0, 1.0),
            "bomb_d": bomb_danger,
            "explosion_d": explosion_danger,
            "exit": 1.0 if destination == goal else 0.0,
        }

    def in_bounds(self, wrld, x, y):
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()

    # Check if a spot can be moved into by the character
    def valid_spot(self, wrld, x, y):
        if not self.in_bounds(wrld, x, y):
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
        return 0  # No path found
    
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

    # Lookahead helpers
    def simulate_move(self, wrld, start, destination):
        """Test one player move in a copied world using the engine's next()."""
        test_world = SensedWorld.from_world(wrld)
        me = self.me_from_world(test_world)
        if me is None:
            return None, []
        me.move(destination[0] - start[0], destination[1] - start[1])
        return test_world.next()

    def future_selfpreserving_states(self, wrld, monster_state, player_pos,
                                        radius, walls, exitcell):
        """
        Possible next states of a chasing monster as (x, y, dx, dy, probability).
        Random choices are returned with equal probability.
        """
        mx, my, mdx, mdy = monster_state
        dx_to_player = player_pos[0] - mx
        dy_to_player = player_pos[1] - my
        sees_player = max(abs(dx_to_player), abs(dy_to_player)) <= radius

        nx, ny = mx + mdx, my + mdy
        must_change = (
            not self.in_bounds(wrld, nx, ny) or
            walls[nx][ny] or
            exitcell == (nx, ny)
        )

        # Chase behavior
        if sees_player and not must_change:
            move_x = (dx_to_player > 0) - (dx_to_player < 0)
            move_y = (dy_to_player > 0) - (dy_to_player < 0)
            tx, ty = mx + move_x, my + move_y
            if self.in_bounds(wrld, tx, ty) and not walls[tx][ty]:
                return [(tx, ty, move_x, move_y, 1.0)]
            return [(mx, my, move_x, move_y, 1.0)]

        # Random choice when idle or blocked
        if (mdx == 0 and mdy == 0) or must_change:
            safe_moves = []
            for move_x, move_y in self.MOVES:
                tx, ty = mx + move_x, my + move_y
                if not self.in_bounds(wrld, tx, ty) or walls[tx][ty]:
                    continue
                if (tx, ty) == player_pos:
                    continue
                safe_moves.append((move_x, move_y))

            if not safe_moves:
                return [(mx, my, 0, 0, 1.0)]

            probability = 1.0 / len(safe_moves)
            return [(mx + mx_, my + my_, mx_, my_, probability)
                    for mx_, my_ in safe_moves]

        # Otherwise keep moving in the current direction
        return [(nx, ny, mdx, mdy, 1.0)]

    def route_risk(self, wrld, player_start, path, monster):
        """Look several moves down the A* route and estimate collision risk."""
        depth = min(8, len(path) - 1)
        if depth <= 0:
            return 0.0

        walls = [[wrld.wall_at(x, y) for y in range(wrld.height())]
                    for x in range(wrld.width())]

        radius = 2 if self.monster_kind(monster) == "aggressive" else 1
        states = {(monster.x, monster.y, monster.dx, monster.dy): 1.0}
        player_pos = player_start
        total_risk = 0.0

        for i in range(depth):
            next_player = path[i + 1]
            new_states = collections.defaultdict(float)

            for monster_state, p_so_far in states.items():
                for nx, ny, ndx, ndy, p in self.future_selfpreserving_states(
                        wrld, monster_state, player_pos, radius, walls,
                        wrld.exitcell):
                    p_now = p_so_far * p

                    # The monster moves before the character.
                    if (nx, ny) == player_pos or (nx, ny) == next_player:
                        total_risk += p_now * 25000
                        continue

                    distance = max(abs(next_player[0] - nx),
                                    abs(next_player[1] - ny))
                    if distance <= 1:
                        total_risk += p_now * 3000
                    elif distance == 2:
                        total_risk += p_now * 800
                    elif distance == 3:
                        total_risk += p_now * 200

                    new_states[(nx, ny, ndx, ndy)] += p_now

            states = new_states
            player_pos = next_player
            if not states:
                break

        return total_risk

    def immediate_danger(self, player_pos, monsters):
        value = 0
        for monster in monsters:
            distance = max(abs(player_pos[0] - monster.x),
                            abs(player_pos[1] - monster.y))
            if self.monster_kind(monster) == "aggressive":
                if distance <= 1:
                    value -= 10000
                elif distance == 2:
                    value -= 5000
                elif distance == 3:
                    value -= 1500
                elif distance == 4:
                    value -= 500
            else:
                if distance == 0:
                    value -= 1000
                elif distance == 1:
                    value -= 500
        return value

    def search_score(self, simulated, goal):
        """Lookahead score for the world after a candidate move."""
        new_me = self.me_from_world(simulated)
        if new_me is None:
            return 0.0

        player_pos = (new_me.x, new_me.y)
        path = self.astar(simulated, player_pos, goal)
        score = -0.15 * len(path) if path else -2.0

        monsters = self.monster_list(simulated)
        score += 0.15 * self.immediate_danger(player_pos, monsters) / 1000.0

        # Penalize routes likely to run into a chasing monster over the next
        # several turns, not just on the next step.
        if path:
            risk = sum(self.route_risk(simulated, player_pos, path, m)
                        for m in monsters
                        if self.monster_kind(m) in ("aggressive", "selfpreserving"))
            score -= 0.005 * risk
        return score

    
    #Decide what is the best move

    def choose_move(self, wrld, start, goal):
        """Returns (dx, dy). Also performs the Q-learning update."""
        if goal is None:
            self._update_q(self._reward_from_events(wrld), terminal=True)
            self.previous_features = None
            self.previous_action = None
            return (0, 0)

        # Reward/events generated by the previous action
        reward = self._reward_from_events(wrld)

        candidates = self.get_neighbors(wrld, start)
        candidates.append(start)  # standing still can be safer than moving

        blast_time, _ = self.build_danger(wrld)

        scored = []
        for destination in candidates:
            # Reject moves the engine's one-step simulation says are fatal.
            simulated, events = self.simulate_move(wrld, start, destination)
            if any(e.tpe in (Event.BOMB_HIT_CHARACTER,
                             Event.CHARACTER_KILLED_BY_MONSTER)
                   for e in events):
                continue

            features = self.action_features(wrld, start, destination, goal, blast_time)
            lookahead = self.search_score(simulated, goal) if simulated is not None else 0.0
            scored.append((self.q_value(features) + lookahead, destination, features))

        terminal = any(e.tpe in (Event.BOMB_HIT_CHARACTER,
                                 Event.CHARACTER_KILLED_BY_MONSTER,
                                 Event.CHARACTER_FOUND_EXIT)
                       for e in getattr(wrld, "events", []))
        next_features = max(scored, key=lambda item: item[0])[2] if scored else None
        self._update_q(reward, next_features=next_features, terminal=terminal)

        if not scored:
            # Everything looks fatal: take the least dangerous move.
            def danger(pos):
                f = self.action_features(wrld, start, pos, goal, blast_time)
                return (f["monster_danger"] + 2 * f["bomb_danger"]
                        + 3 * f["explosion_danger"])
            destination = min(candidates, key=danger)
            features = self.action_features(wrld, start, destination, goal, blast_time)
        elif self.rng.random() < self.EPSILON:
            _, destination, features = self.rng.choice(scored)
        else:
            _, destination, features = max(scored, key=lambda item: item[0])

        self.previous_features = features
        self.previous_action = destination
        return (destination[0] - start[0], destination[1] - start[1])


    # Primary function where information is given to the expectimax algorithm, runs it, and declares the move
    def do(self, wrld):
        print("Hi" + str(wrld.scores["me"]))
        me = wrld.me(self)
        start = (me.x, me.y)
        goal = self.get_exit(wrld)

        dx, dy = self.choose_move(wrld, start, goal)
        self.move(dx, dy)

# This is necessary to find the main code
import heapq
from random import random
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back

class TestCharacter(CharacterEntity):
    # List of standard moves that can be made by any entity at any time
    smoves = [(-1, -1), (0, -1), (1, -1), (-1,  0), (0, 0), (1,  0), (-1,  1), (0,  1), (1,  1)]

    def __init__(self):
        self.weights = [1.0, -1.0, -1.0]          # exit, monster, bomb
        self.alpha, self.gamma, self.epsilon = 0.1, 0.9, 0.1
        self.prev_features = None
        self.prev_q = None
        self.prev_exit_dist = None
        self.safety_margin = 1.0

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

    
    def exit_distance(self, wrld, pos, goal):
        path = self.astar_path_to_exit(wrld, pos, goal)
        return len(path) if path else wrld.width() + wrld.height()

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

    def Q_value(wrld, features, weights):
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

    def update_weights(self, reward, next_best_q):
        delta = (reward + self.gamma * next_best_q) - self.prev_q
        self.weights = [w + self.alpha * delta * f
                        for w, f in zip(self.weights, self.prev_features)]




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

    # Primary function where information is given to the expectimax algorithm, runs it, and declares the move
    def do(self, wrld):
        me = wrld.me(self)
        start = (me.x, me.y)
        goal = self.get_exit(wrld)

        options = self.action_values(wrld, start, goal)
        if not options:
            return

        #1) learn from previous step

        if self. prev_features is not None:
            r, exit_dist = self.reward(wrld, start, goal)
            self.update_weights(r, max(o[0] for o in options))
        else:
            dist = self.exit_distance(wrld, start, goal)

        #2) chose an action, using epsilon-greedy exploration
        danger = {(dx,dy): d for d, dx, dy in self.expectimax_move(wrld, start, goal)}

        if danger:
            best_danger = min(danger.values())
            safe = [o for o in options 
                    if o[1] in danger and danger[o[1]] <= best_danger + self.safety_margin]
        if not safe:
            safe = options

        if random() < self.epsilon:
            q, (dx, dy), feats = random.choice(safe)
        else:
            q, (dx, dy), feats = max(safe, key=lambda o: o[0])

        #3) store features and q for next step
        self.prev_features, self.prev_q, self.prev_exit_dist = feats, q, exit_dist
    
        self.move(dx,dy)
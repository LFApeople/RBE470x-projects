# This is necessary to find the main code
import heapq
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back

class TestCharacter(CharacterEntity):
    moves = [(-1, -1), (0, -1), (1, -1), (-1,  0), (1,  0), (-1,  1), (0,  1), (1,  1)]

    def valid_spot(self, wrld, x, y):
        if not (0 <= x < wrld.width() and 0 <= y < wrld.height()):
            return False
        return wrld.empty_at(x, y) or wrld.exit_at(x, y)

    def get_neighbors(self, wrld, current):
        x, y = current
        neighbors = []

        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if self.valid_spot(wrld, nx, ny):
                neighbors.append(((nx, ny), 1))
        return neighbors

    def get_neighbors_monsters(self, wrld, current):
        x, y = current
        monsters = []

        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < wrld.width() and 0 <= ny < wrld.height()):
                if wrld.monsters_at(nx, ny):
                    monsters.append(((nx, ny), 1))
        return monsters

    
    def astar_path_to_exit(self, wrld, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        
        came_from = {}
        cost_so_far = {start: 0}
        entry_counter = 0

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
                    # Penalize every monster near a possible path node so the
                    # route leaves a safety buffer instead of merely avoiding
                    # the monster's current square.
                    new_cost += len(
                        self.get_neighbors_monsters(
                            wrld, (next_node[0], next_node[1])
                        )
                    )
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + max(abs(next_node[0] - goal[0]), abs(next_node[1] - goal[1]))
                    entry_counter += 1
                    heapq.heappush(frontier, (priority, entry_counter, next_node))
                    came_from[next_node] = current

        return None

    def expectimax_move(self, wrld, start, goal, depth=2):
        path = self.astar_path_to_exit(wrld, start, goal)
        route_next = path[1] if path and len(path) > 1 else None

        monsters = [(monster.x, monster.y)
                    for cells in wrld.monsters.values()
                    for monster in cells]

        def monster_choices(position):
            choices = [position]
            x, y = position
            for dx, dy in self.moves:
                nx, ny = x + dx, y + dy
                if self.valid_spot(wrld, nx, ny):
                    choices.append((nx, ny))
            return choices

        def score(position, monster_positions):
            danger = 0
            for monster in monster_positions:
                distance = min(abs(position[0] - monster[0]),
                               abs(position[1] - monster[1]))
                danger = max(danger, 3 - distance)
            distance_to_exit = abs(position[0] - goal[0]) + abs(position[1] - goal[1])
            route_penalty = 0 if route_next == position else 2
            return danger + distance_to_exit + route_penalty

        def chance(position, monster_positions, turns):
            if turns == 0 or not monster_positions:
                return score(position, monster_positions)
            outcomes = [[]]
            for monster in monster_positions:
                outcomes = [prefix + [next_position]
                            for prefix in outcomes
                            for next_position in monster_choices(monster)]
                if len(outcomes) > 128:
                    outcomes = outcomes[:128]
            return sum(score(position, outcome) for outcome in outcomes) / len(outcomes)

        candidates = []
        for dx, dy in self.moves:
            position = (start[0] + dx, start[1] + dy)
            if self.valid_spot(wrld, *position):
                candidates.append((chance(position, monsters, depth - 1), dx, dy))
        if not candidates:
            return None
        _, dx, dy = min(candidates)
        return dx, dy

    def get_exit(self, wrld):
        for x in range(wrld.width()):
            for y in range(wrld.height()):
                if wrld.exit_at(x, y):
                    return (x, y)
        return None

    def do(self, wrld):
        me = wrld.me(self)
        start = (me.x, me.y)
        goal = self.get_exit(wrld)


        action = self.expectimax_move(wrld, start, goal)
        if action:
            self.move(*action)
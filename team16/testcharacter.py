# This is necessary to find the main code
import heapq
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from colorama import Fore, Back

class TestCharacter(CharacterEntity):
    # List of standard moves that can be made by any entity at any time
    moves = [(-1, -1), (0, -1), (1, -1), (-1,  0), (0, 0), (1,  0), (-1,  1), (0,  1), (1,  1)]

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

        options = self.expectimax_move(wrld, start, goal)
        options.sort()
            
        self.move(options[0][1], options[0][2])
# This is necessary to find the main code
import heapq
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back

class TestCharacter(CharacterEntity):

    def valid_spot(self, wrld, x, y):
        if wrld.empty_at(x, y) and (0 <= x < wrld.width() and 0 <= y < wrld.height()):
            return True
        return False
    
    def get_neighbors(self, wrld, current):
        x, y = current
        moves = [(-1, -1), (0, -1), (1, -1), (-1,  0), (1,  0), (-1,  1), (0,  1), (1,  1)]  
        neighbors = []

        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if self.valid_spot(wrld, nx, ny):
                neighbors.append(((nx, ny), 1))
                
        return neighbors

    def pathfinding(self, wrld, start, goal):
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
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + max(abs(next_node[0] - goal[0]), abs(next_node[1] - goal[1]))
                    entry_counter += 1
                    heapq.heappush(frontier, (priority, entry_counter, next_node))
                    came_from[next_node] = current

        return None

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
        path = self.pathfinding(wrld, start, goal)

        if len(path) > 1:
            nx, ny = path[1]
            dx = nx - start[0]
            dy = ny - start[1]
            self.move(dx, dy)
        pass
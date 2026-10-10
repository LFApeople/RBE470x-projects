# This is necessary to find the main code
import heapq
import collections
import math
import json
import os
import random
import sys
from collections import deque

sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from sensed_world import SensedWorld
from events import Event
from colorama import Fore, Back


class TestCharacter(CharacterEntity):
    """A* + threat-aware Bomberman agent with linear Approximate Q-Learning.

    Each turn every candidate move is scored with
        Q(s, a) = sum_i w_i * f_i(s, a)
    plus a lookahead term (A* path length, monster proximity, route risk).
    The weights are updated from the reward observed on the following turn
    (TD update) and saved to disk so learning persists between games.
    """

    # ------------------------------------------------------------------
    # Q-learning configuration
    # ------------------------------------------------------------------
    FEATURE_NAMES = (
        "bias", "progress", "monster_danger", "adjacent_monsters",
        "safe_neighbors", "bomb_danger", "explosion_danger", "exit"
    )
    INITIAL_WEIGHTS = [0.0, 2.0, -5.0, -4.0, 0.6, -8.0, -12.0, 20.0]

    LEARNING = True   # set False to freeze the weights (e.g. when evaluating)
    TD_CLIP = 5.0     # clip TD error so one +/-100 reward can't wreck the weights
    WEIGHT_CLIP = 50.0
    ALPHA = 0.02      # learning rate
    GAMMA = 0.90      # discount
    EPSILON = 0.03    # exploration rate
    WEIGHTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "q_weights.json")

    # Moves WITHOUT standing still (standing still is added separately)
    MOVES = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1)
    ]

    safety_margin = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = dict(zip(self.FEATURE_NAMES, self.INITIAL_WEIGHTS))
        self.previous_features = None
        self.previous_action = None
        self.rng = random.Random()
        self._load_weights()

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------
    def _load_weights(self):
        try:
            with open(self.WEIGHTS_FILE, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            for name in self.FEATURE_NAMES:
                if name in saved and isinstance(saved[name], (int, float)):
                    self.weights[name] = float(saved[name])
        except (OSError, ValueError, TypeError):
            pass

    def _save_weights(self):
        try:
            temporary = self.WEIGHTS_FILE + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(self.weights, handle, indent=2, sort_keys=True)
            os.replace(temporary, self.WEIGHTS_FILE)
        except OSError:
            pass  # still run in read-only environments

    # ------------------------------------------------------------------
    # Q-learning core
    # ------------------------------------------------------------------
    def _reward_from_events(self, wrld):
        reward = -0.05  # small per-step cost encourages reaching the exit
        for event in getattr(wrld, "events", []):
            if event.tpe in (Event.BOMB_HIT_CHARACTER,
                             Event.CHARACTER_KILLED_BY_MONSTER):
                reward -= 100.0
            elif event.tpe == Event.CHARACTER_FOUND_EXIT:
                reward += 100.0
            elif event.tpe == Event.BOMB_HIT_MONSTER:
                reward += 2.0
            elif event.tpe == getattr(Event, "BOMB_HIT_WALL", None):
                reward += 1.0   # opened up the map
        return reward

    def q_value(self, features):
        return sum(self.weights[name] * features.get(name, 0.0)
                   for name in self.FEATURE_NAMES)

    def _update_q(self, reward, next_features=None, terminal=False):
        if not self.LEARNING:
            return
        if self.previous_features is None or self.previous_action is None:
            return
        old_q = self.q_value(self.previous_features)
        future = 0.0 if terminal or next_features is None else self.q_value(next_features)
        td_error = reward + self.GAMMA * future - old_q
        td_error = max(-self.TD_CLIP, min(self.TD_CLIP, td_error))
        for name in self.FEATURE_NAMES:
            w = self.weights[name] + self.ALPHA * td_error * self.previous_features.get(name, 0.0)
            self.weights[name] = max(-self.WEIGHT_CLIP, min(self.WEIGHT_CLIP, w))
        self._save_weights()

    def action_features(self, wrld, start, destination, goal, blast_time=None):
        """Normalized state-action features for one candidate destination."""
        if blast_time is None:
            blast_time, _ = self.build_danger(wrld)

        distance_before = self.exit_distance(start, goal)
        distance_after = self.exit_distance(destination, goal)
        progress = max(-1.0, min(1.0, distance_before - distance_after))

        monsters = self.monster_list(wrld)
        nearest = 1.0
        adjacent = 0.0
        for monster in monsters:
            distance = max(abs(destination[0] - monster.x),
                           abs(destination[1] - monster.y))
            nearest = min(nearest, min(distance / 5.0, 1.0))
            if distance <= 1:
                adjacent += 1.0

        safe_neighbors = sum(
            1 for pos in self.get_neighbors(wrld, destination)
            if pos not in blast_time
        )

        # Blast windows come from build_danger (bomb timers + live explosions)
        windows = blast_time.get(destination, [])
        bomb_danger = 1.0 if windows else 0.0
        explosion_danger = 1.0 if any(s <= 1 < e for s, e in windows) else 0.0

        return {
            "bias": 1.0,
            "progress": progress,
            "monster_danger": 1.0 - nearest if monsters else 0.0,
            "adjacent_monsters": min(adjacent, 3.0),
            "safe_neighbors": min(safe_neighbors / 8.0, 1.0),
            "bomb_danger": bomb_danger,
            "explosion_danger": explosion_danger,
            "exit": 1.0 if destination == goal else 0.0,
        }

    # ------------------------------------------------------------------
    # Map helpers
    # ------------------------------------------------------------------
    def in_bounds(self, wrld, x, y):
        return 0 <= x < wrld.width() and 0 <= y < wrld.height()

    # Check if a spot can be moved into by the character
    def valid_spot(self, wrld, x, y):
        if not self.in_bounds(wrld, x, y):
            return False
        return wrld.empty_at(x, y) or wrld.exit_at(x, y)

    # Get all open spaces neighboring a position (does not include staying put)
    def get_neighbors(self, wrld, current):
        x, y = current
        return [(x + dx, y + dy) for dx, dy in self.MOVES
                if self.valid_spot(wrld, x + dx, y + dy)]

    def exit_distance(self, pos, goal):
        """Straight-line distance to the goal."""
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    # Needed to actually find where the goal is
    def get_exit(self, wrld):
        return getattr(wrld, "exitcell", None)

    def monster_list(self, wrld):
        return [m for cells in wrld.monsters.values() for m in cells]

    def me_from_world(self, wrld):
        for cells in wrld.characters.values():
            for character in cells:
                if character.name == self.name:
                    return character
        return None

    # ------------------------------------------------------------------
    # Danger helpers (explosions)
    # ------------------------------------------------------------------
    def blast_cells(self, wrld, bx, by):
        """Cells a bomb at (bx, by) will hit (stops at walls)."""
        rng = getattr(wrld, "expl_range", 4)
        cells = {(bx, by)}
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, rng + 1):
                x, y = bx + dx * i, by + dy * i
                if not self.in_bounds(wrld, x, y) or wrld.wall_at(x, y):
                    break
                cells.add((x, y))
        return cells

    def build_danger(self, wrld):
        """
        Returns:
          blast_time: {cell: [(start_tick, end_tick), ...]} windows when cell is deadly
          monsters:   list of (x, y)
        """
        duration = getattr(wrld, "expl_duration", 2)
        blast_time = {}

        for bomb in wrld.bombs.values():
            for cell in self.blast_cells(wrld, bomb.x, bomb.y):
                blast_time.setdefault(cell, []).append(
                    (bomb.timer, bomb.timer + duration))

        for expl in wrld.explosions.values():
            blast_time.setdefault((expl.x, expl.y), []).append((0, expl.timer))

        monsters = [(m.x, m.y) for m in self.monster_list(wrld)]
        return blast_time, monsters

    def cell_deadly(self, cell, t, blast_time, monsters):
        for start, end in blast_time.get(cell, []):
            if start <= t < end:
                return True
        radius = 1 + int(self.safety_margin)
        for mx, my in monsters:
            if max(abs(cell[0] - mx), abs(cell[1] - my)) <= radius:
                return True
        return False

    def is_threatened(self, wrld, pos, horizon=4):
        """True if standing still at pos gets you hurt within `horizon` ticks."""
        blast_time, monsters = self.build_danger(wrld)
        return any(self.cell_deadly(pos, t, blast_time, monsters)
                   for t in range(horizon + 1))

    # Escape move to avoid danger (not wired into choose_move yet)
    def escape_move(self, wrld, start, max_steps=10):
        """
        BFS over (cell, tick). Finds the shortest sequence of moves to a cell
        that is out of every blast zone and away from monsters.
        Returns (dx, dy), or None if already safe / nothing reachable.
        """
        blast_time, monsters = self.build_danger(wrld)

        def passable(x, y):
            return (self.in_bounds(wrld, x, y)
                    and not wrld.wall_at(x, y)
                    and not wrld.bombs_at(x, y))

        def permanently_safe(cell):
            if cell in blast_time:
                return False
            return not self.cell_deadly(cell, 0, {}, monsters)

        if permanently_safe(start) and not self.is_threatened(wrld, start):
            return None

        queue = deque([(start, 0, None)])
        seen = {(start, 0)}
        fallback = None

        while queue:
            (x, y), t, first = queue.popleft()

            if first is not None and permanently_safe((x, y)):
                return first
            if t >= max_steps:
                continue

            for dx, dy in self.MOVES + [(0, 0)]:   # (0, 0) = wait
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

    # ------------------------------------------------------------------
    # A* path to the exit
    # ------------------------------------------------------------------
    def astar(self, wrld, start, goal):
        """Returns [start, ..., goal], or [] if the goal is unreachable."""
        if start == goal:
            return [start]

        frontier = []
        count = 0
        heapq.heappush(frontier, (0, count, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for nxt in self.get_neighbors(wrld, current):
                new_cost = cost_so_far[current] + 1
                if nxt in cost_so_far and new_cost >= cost_so_far[nxt]:
                    continue
                cost_so_far[nxt] = new_cost
                f = new_cost + self.exit_distance(nxt, goal)
                count += 1
                heapq.heappush(frontier, (f, count, nxt))
                came_from[nxt] = current

        if goal not in came_from:
            return []

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Lookahead helpers
    # ------------------------------------------------------------------
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

        radius = self.MONSTER_RADIUS
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
            r = self.MONSTER_RADIUS
            if distance == 0:
                value -= 10000
            elif distance <= r:
                value -= 5000
            elif distance == r + 1:
                value -= 1500
            elif distance == r + 2:
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
                       for m in monsters)
            score -= 0.005 * risk
        return score

    # ------------------------------------------------------------------
    # Expectimax against chasing monsters
    #   MAX nodes  = our move
    #   CHANCE     = the monsters' possible moves (future_selfpreserving_states)
    # Each tick the monsters move first, then we move.
    # ------------------------------------------------------------------
    MONSTER_RADIUS = 1   # attack range assumed for every monster
    EM_DEPTH = 3
    EM_WEIGHT = 0.05
    DEATH = -1000.0
    WIN = 500.0

    def abstract_moves(self, wrld, walls, pos):
        x, y = pos
        moves = [pos]   # standing still
        for dx, dy in self.MOVES:
            nx, ny = x + dx, y + dy
            if self.in_bounds(wrld, nx, ny) and not walls[nx][ny]:
                moves.append((nx, ny))
        return moves

    def monster_outcomes(self, wrld, monsters, player, walls):
        """Joint outcomes of all chasing monsters moving: [(monsters, prob)]."""
        joint = [((), 1.0)]
        for (x, y, dx, dy, radius) in monsters:
            options = self.future_selfpreserving_states(
                wrld, (x, y, dx, dy), player, radius, walls, wrld.exitcell)
            joint = [(prefix + ((nx, ny, ndx, ndy, radius),), p * q)
                     for prefix, p in joint
                     for nx, ny, ndx, ndy, q in options]
            if len(joint) > 64:                       # keep the search cheap
                joint = sorted(joint, key=lambda j: -j[1])[:64]
        total = sum(p for _, p in joint)
        return [(m, p / total) for m, p in joint]

    def em_heuristic(self, wrld, walls, player, monsters, goal):
        value = -self.exit_distance(player, goal)
        for m in monsters:
            d = max(abs(player[0] - m[0]), abs(player[1] - m[1]))
            if d <= 2:
                value -= (3 - d) * 5
        # Dead ends are where chasing monsters get you: reward open space
        value += 0.5 * (len(self.abstract_moves(wrld, walls, player)) - 1)
        return value

    def em_max(self, wrld, walls, player, monsters, goal, depth):
        if depth <= 0:
            return self.em_heuristic(wrld, walls, player, monsters, goal)
        key = (player, monsters, depth)
        if key in self._em_memo:
            return self._em_memo[key]
        best = max(self.em_step(wrld, walls, player, monsters, nxt, goal, depth)
                   for nxt in self.abstract_moves(wrld, walls, player))
        self._em_memo[key] = best
        return best

    def em_step(self, wrld, walls, player, monsters, nxt, goal, depth):
        """Expected value of one tick: monsters move (seeing `player`), then we go to `nxt`."""
        value = 0.0
        for new_m, p in self.monster_outcomes(wrld, monsters, player, walls):
            cells = {(m[0], m[1]) for m in new_m}
            if player in cells or nxt in cells:
                value += p * self.DEATH
            elif nxt == goal:
                value += p * self.WIN
            else:
                value += p * self.em_max(wrld, walls, nxt, new_m, goal, depth - 1)
        return value

    def monster_need(self):
        """
        Distance we must keep from a monster's CURRENT cell after our move.
        A monster attacks anything within MONSTER_RADIUS at the start of its
        turn and may first step 1 closer at random, so we need radius + 2.
        """
        return self.MONSTER_RADIUS + 2

    def monster_clearance(self, wrld, cell):
        """Smallest (distance - needed distance) over all monsters. >= 0 means safe."""
        margin = 99
        for m in self.monster_list(wrld):
            need = self.monster_need()
            for mx, my in ((m.x, m.y), (m.x + m.dx, m.y + m.dy)):
                d = max(abs(cell[0] - mx), abs(cell[1] - my))
                margin = min(margin, d - need)
        return margin

    # ------------------------------------------------------------------
    # Bomb placement
    # ------------------------------------------------------------------
    CARDINALS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def can_place_bomb(self, wrld):
        """The game allows one active bomb at a time."""
        return not wrld.bombs

    def walls_hit(self, wrld, bx, by):
        """Walls a bomb at (bx, by) would destroy (first wall in each direction)."""
        rng = getattr(wrld, "expl_range", 4)
        hit = set()
        for dx, dy in self.CARDINALS:
            for i in range(1, rng + 1):
                x, y = bx + dx * i, by + dy * i
                if not self.in_bounds(wrld, x, y):
                    break
                if wrld.wall_at(x, y):
                    hit.add((x, y))
                    break
        return hit

    def astar_through_walls(self, wrld, start, goal, wall_cost=6):
        """A* that may cross walls at extra cost, to see which walls to blow up."""
        frontier = [(0, 0, start)]
        count = 0
        came_from = {start: None}
        cost = {start: 0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dx, dy in self.MOVES:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.in_bounds(wrld, *nxt):
                    continue
                step = wall_cost if wrld.wall_at(*nxt) else 1
                new_cost = cost[current] + step
                if nxt in cost and new_cost >= cost[nxt]:
                    continue
                cost[nxt] = new_cost
                count += 1
                heapq.heappush(frontier,
                               (new_cost + self.exit_distance(nxt, goal), count, nxt))
                came_from[nxt] = current

        if goal not in came_from:
            return []
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def breach_move(self, wrld, start, goal):
        """
        When walls cut off the exit: return the next cell on the way to a spot
        from which a bomb opens the route (or `start` if already at such a spot).
        """
        wpath = self.astar_through_walls(wrld, start, goal)
        first_wall = next((c for c in wpath if wrld.wall_at(*c)), None)
        if first_wall is None:
            return None

        rng = getattr(wrld, "expl_range", 4)
        best = None
        for dx, dy in self.CARDINALS:
            for i in range(1, rng + 1):
                site = (first_wall[0] + dx * i, first_wall[1] + dy * i)
                if not self.in_bounds(wrld, *site) or wrld.wall_at(*site):
                    break
                if site == start:
                    return start
                if not self.valid_spot(wrld, *site):
                    continue
                path = self.astar(wrld, start, site)
                if path and (best is None or len(path) < len(best)):
                    best = path

        if best is None or len(best) < 2:
            return None
        return best[1]

    def bomb_escape_move(self, wrld, start):
        """
        Pretend a bomb is dropped at `start` now. BFS over (cell, tick) for a
        way out of the new blast zone before it detonates. Returns the first
        (dx, dy) of the escape, or None if there is no safe escape.
        """
        timer = getattr(wrld, "bomb_time", 10)
        duration = getattr(wrld, "expl_duration", 2)
        new_blast = self.blast_cells(wrld, *start)
        blast_time, _ = self.build_danger(wrld)
        monsters = [(m.x, m.y) for m in self.monster_list(wrld)]

        def deadly(cell, t):
            if cell in new_blast and timer <= t < timer + duration:
                return True
            if any(s <= t < e for s, e in blast_time.get(cell, [])):
                return True
            return self.monster_clearance(wrld, cell) < 0

        queue = deque([(start, 0, None)])
        seen = {(start, 0)}
        while queue:
            cell, t, first = queue.popleft()
            if first is not None and cell not in new_blast:
                return first
            if t >= timer - 1:
                continue
            for dx, dy in self.MOVES:
                nxt = (cell[0] + dx, cell[1] + dy)
                nt = t + 1
                # the bomb sits on `start`, so it can't be re-entered
                if nxt == start or (nxt, nt) in seen:
                    continue
                if not self.valid_spot(wrld, *nxt) or deadly(nxt, nt):
                    continue
                seen.add((nxt, nt))
                queue.append((nxt, nt, first if first is not None else (dx, dy)))
        return None

    def plan_bomb(self, wrld, start, goal):
        """
        Decide whether to drop a bomb here. Returns the escape move (dx, dy)
        to make this same turn, or None for "don't bomb".
        A bomb is only placed if it is useful AND we have a safe way out.
        """
        if goal is None or not self.can_place_bomb(wrld):
            return None

        useful = False

        # 1. A monster is close and in the blast lines
        blast = self.blast_cells(wrld, *start)
        for m in self.monster_list(wrld):
            if ((m.x, m.y) in blast
                    and max(abs(m.x - start[0]), abs(m.y - start[1])) <= 3):
                useful = True

        # 2. Walls cut off the exit and this bomb breaks one on the route
        if not useful and not self.astar(wrld, start, goal):
            wpath = self.astar_through_walls(wrld, start, goal)
            route_walls = {c for c in wpath if wrld.wall_at(*c)}
            if route_walls & self.walls_hit(wrld, *start):
                useful = True

        if not useful:
            return None
        return self.bomb_escape_move(wrld, start)

    def is_safe_step(self, wrld, start, step):
        _, events = self.simulate_move(wrld, start, step)
        if any(e.tpe in (Event.BOMB_HIT_CHARACTER,
                         Event.CHARACTER_KILLED_BY_MONSTER) for e in events):
            return False
        return self.monster_clearance(wrld, step) >= 0

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------
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

        # Abstract model of the chasing monsters for the expectimax lookahead
        walls = [[wrld.wall_at(x, y) for y in range(wrld.height())]
                 for x in range(wrld.width())]
        # Treat every monster as a chaser with the same attack radius
        chasers = tuple((m.x, m.y, m.dx, m.dy, self.MONSTER_RADIUS)
                        for m in self.monster_list(wrld))
        self._em_memo = {}

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
            if chasers:
                lookahead += self.EM_WEIGHT * self.em_step(
                    wrld, walls, start, chasers, destination, goal, self.EM_DEPTH)
            scored.append((self.q_value(features) + lookahead, destination, features))

        # Keep a safe gap to every monster. If no move has one, keep only the
        # moves with the best (largest) gap instead of ignoring the rule.
        if scored:
            margins = [99 if s[1] == goal else self.monster_clearance(wrld, s[1])
                       for s in scored]
            cutoff = 0 if max(margins) >= 0 else max(margins)
            scored = [s for s, m in zip(scored, margins) if m >= cutoff]

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

    # Primary function called by the game each turn
    def do(self, wrld):
        print("Hi" + str(wrld.scores["me"]))
        me = wrld.me(self)
        start = (me.x, me.y)
        goal = self.get_exit(wrld)

        dx, dy = self.choose_move(wrld, start, goal)
        destination = (start[0] + dx, start[1] + dy)

        if goal is not None:
            # 1. Exit is walled off: head to a spot where a bomb opens the route
            if not wrld.bombs and not self.astar(wrld, start, goal):
                step = self.breach_move(wrld, start, goal)
                if step is not None and self.is_safe_step(wrld, start, step):
                    destination = step

            # 2. Drop a bomb if it is useful and we can escape the blast.
            #    The bomb goes on the current cell, then we move away this turn.
            escape = self.plan_bomb(wrld, start, goal)
            if escape is not None:
                self.place_bomb()
                destination = (start[0] + escape[0], start[1] + escape[1])

            # Keep the Q-learning update consistent with the move actually taken
            if self.previous_action != destination:
                self.previous_action = destination
                self.previous_features = self.action_features(
                    wrld, start, destination, goal)

            # We will reach the exit this tick: the game ends before the next
            # do() call, so apply the terminal reward now.
            if destination == goal:
                self._update_q(100.0, terminal=True)

        self.move(destination[0] - start[0], destination[1] - start[1])

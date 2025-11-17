import pygame
import sys
import math
import heapq
import random
import numpy as np
from collections import deque, namedtuple
from visualizer_sprites import Visualizer

TerrainCell = namedtuple('TerrainCell', 'type cost visible explored has_sample')

class Config:
    def __init__(self):
        self.map_w = 1200
        self.map_h = 938
        self.sidebar_width = 240
        self.screen_w = self.map_w + self.sidebar_width
        self.screen_h = self.map_h
        self.tile = 10
        self.map_size = (self.map_h, self.map_w)
        self.number_rows = 3
        self.number_columns = 3
        self.noise = (3, 3)
        self.margins = (0.15, 0.15)
        self.obstacle_size = (1, 3)
        self.obstacles = 100
        self.p_rock = 0.12
        self.p_crater = 0.06
        self.p_sample = 0.04
        self.seed = 42
        self.rover_energy = 1200
        self.move_cost = 1
        self.crater_cost = 100
        self.scan_range = 6
        self.max_steps = 50000
        self.fps = 60
        self.return_threshold = 0.5
        self.sprite_scale = 2.0

class MapGenerator:
    def __init__(self, size=(938,1200), obstacles=100,
                 number_rows=3, number_columns=3,
                 noise=(3,3), margins=(0.15,0.15),
                 obstacle_size=(1,3), seed=42):
        self.height = int(size[0])
        self.width = int(size[1])
        self.number_rows = number_rows
        self.number_columns = number_columns
        self.noise = (int(noise[0]), int(noise[1]))
        self.margins = margins
        self.obstacle_size = obstacle_size
        self.obstacles = obstacles
        self.size = (self.height, self.width)
        self.rng = np.random.RandomState(seed)
        self.map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.hv = np.array([], dtype=int)
        self.wv = np.array([], dtype=int)
        self._obstacles_initial_positions()
        if self.number_rows is not None and self.number_columns is not None:
            self._noise_obstacles_positions()
            self._random_obstacle_size()
        else:
            self._random_obstacle_size_cell()
        if self.hv.size and self.wv.size:
            self.hv = np.clip(self.hv, 0, self.height - 1)
            self.wv = np.clip(self.wv, 0, self.width - 1)
            self.map[self.hv, self.wv] = 1

    def _obstacles_initial_positions(self):
        margin_h = int(self.margins[0] * self.height)
        margin_w = int(self.margins[1] * self.width)
        if self.number_rows is None or self.number_columns is None:
            if self.obstacles <= 0:
                self.hv = np.array([], dtype=int); self.wv = np.array([], dtype=int); return
            h_obstacles = self.rng.randint(margin_h, max(margin_h+1, self.height - margin_h), size=self.obstacles)
            w_obstacles = self.rng.randint(margin_w, max(margin_w+1, self.width - margin_w), size=self.obstacles)
            self.hv = h_obstacles; self.wv = w_obstacles
        else:
            h_positions = np.linspace(margin_h, max(margin_h, self.height - 1 - margin_h),
                                      self.number_rows, dtype=int)
            w_positions = np.linspace(margin_w, max(margin_w, self.width - 1 - margin_w),
                                      self.number_columns, dtype=int)
            hv_grid, wv_grid = np.meshgrid(h_positions, w_positions)
            self.hv = hv_grid.reshape(-1); self.wv = wv_grid.reshape(-1)

    def _noise_obstacles_positions(self):
        h_noise = abs(int(self.noise[0])); w_noise = abs(int(self.noise[1]))
        if self.hv.size == 0: return
        if h_noise > 0:
            offsets_h = self.rng.randint(-h_noise, h_noise + 1, size=self.hv.shape[0]); self.hv = self.hv + offsets_h
        if w_noise > 0:
            offsets_w = self.rng.randint(-w_noise, w_noise + 1, size=self.wv.shape[0]); self.wv = self.wv + offsets_w

    def _random_obstacle_size(self):
        new_h = []; new_w = []
        min_h, max_h = int(self.obstacle_size[0]), int(self.obstacle_size[1])
        min_w, max_w = min_h, max_h
        for i in range(self.hv.shape[0]):
            ob_h = self.rng.randint(max(1, min_h), max(1, max_h) + 1)
            ob_w = self.rng.randint(max(1, min_w), max(1, max_w) + 1)
            base_h = self.hv[i]; base_w = self.wv[i]
            h_inds = np.arange(base_h, base_h + ob_h); w_inds = np.arange(base_w, base_w + ob_w)
            if h_inds.size and w_inds.size:
                hh, ww = np.meshgrid(h_inds, w_inds); new_h.append(hh.reshape(-1)); new_w.append(ww.reshape(-1))
        if new_h:
            self.hv = np.concatenate(new_h); self.wv = np.concatenate(new_w)
        else:
            self.hv = np.array([], dtype=int); self.wv = np.array([], dtype=int)

    def _random_obstacle_size_cell(self):
        new_h = []; new_w = []
        if self.hv.size == 0: return
        min_h, max_h = int(self.obstacle_size[0]), int(self.obstacle_size[1])
        min_w, max_w = min_h, max_h
        for i in range(self.hv.shape[0]):
            ob_h = self.rng.randint(max(1, min_h), max(1, max_h) + 1)
            ob_w = self.rng.randint(max(1, min_w), max(1, max_w) + 1)
            base_h = self.hv[i]; base_w = self.wv[i]
            h_inds = np.arange(base_h, base_h + ob_h); w_inds = np.arange(base_w, base_w + ob_w)
            if h_inds.size and w_inds.size:
                hh, ww = np.meshgrid(h_inds, w_inds); new_h.append(hh.reshape(-1)); new_w.append(ww.reshape(-1))
        if new_h:
            self.hv = np.concatenate(new_h); self.wv = np.concatenate(new_w)
        else:
            self.hv = np.array([], dtype=int); self.wv = np.array([], dtype=int)

    def get_map(self):
        return self.map.copy()

class LazyGrid:
    def __init__(self, cfg: Config, map_array: np.ndarray):
        self.cfg = cfg; self.map = map_array
        self.height, self.width = self.map.shape
        self.cells = {}; self.seed = cfg.seed
        self.rng = np.random.RandomState(self.seed)

    def get_cell(self, x, y):
        if (x, y) in self.cells: return self.cells[(x, y)]
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            cell = TerrainCell('C', math.inf, False, False, False); self.cells[(x,y)] = cell; return cell
        if self.map[y, x] == 1:
            cell = TerrainCell('R', math.inf, False, False, False)
        else:
            has_sample = (self.rng.rand() < self.cfg.p_sample)
            cell = TerrainCell('E', 1, False, False, has_sample)
        self.cells[(x,y)] = cell; return cell

    def neighbors(self, pos):
        x, y = pos
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            yield (x+dx, y+dy)

class Rover:
    def __init__(self, cfg: Config, grid: LazyGrid):
        self.cfg = cfg; self.grid = grid
        cx = grid.width // 2; cy = grid.height // 2
        self.pos = (cx, cy); self.base = (cx, cy)
        self.energy = cfg.rover_energy; self.samples = 0
        self.known = {}; self.path = []; self.scan_range = cfg.scan_range
        self.steps_taken = 0; self.revealed_centers = set()
        self.returning = False; self.second_pass = False
        self.trace = []
        self.mode_trace_color = (255,255,0)
        self.history = [self.pos]
        self.update_known(self.pos)

    def update_known(self, center):
        cx, cy = center; r = self.scan_range
        self.revealed_centers.add(center)
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                truecell = self.grid.get_cell(x, y)
                self.known[(x,y)] = TerrainCell(truecell.type, truecell.cost, True, True, truecell.has_sample)

    def visible_targets(self):
        return [p for p,c in self.known.items() if c.visible and c.has_sample]

    def collect(self):
        x, y = self.pos
        cell = self.grid.get_cell(x, y)
        if cell.has_sample:
            self.samples += 1
            self.grid.cells[(x,y)] = TerrainCell(cell.type, cell.cost, True, True, False)
            self.known[(x,y)] = TerrainCell(self.known.get((x,y), cell).type, cell.cost, True, True, False)

    def move_step(self):
        if not self.path:
            return
        prev = self.pos
        nextpos = self.path.pop(0)
        nextcell = self.grid.get_cell(*nextpos)
        if nextcell.type in ('R','C'):
            self.path = []; return
        self.pos = nextpos; self.energy -= self.cfg.move_cost; self.steps_taken += 1
        self.trace.append((prev, self.pos, self.mode_trace_color))
        self.history.append(self.pos)
        self.update_known(self.pos); self.collect()

    def remaining_energy_fraction(self):
        return max(0, self.energy) / self.cfg.rover_energy

class Pathfinder:
    def __init__(self, grid): self.grid = grid
    def heuristic(self, a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
    def astar(self, start, goal, known):
        self.last_expanded = 0
        if start==goal: return []
        frontier=[]; heapq.heappush(frontier,(0,start))
        came_from={start:None}; cost_so_far={start:0}
        while frontier:
            _,current = heapq.heappop(frontier)
            self.last_expanded += 1
            if current==goal: break
            for nxt in self.grid.neighbors(current):
                realcell = self.grid.get_cell(*nxt)
                if realcell.type in ('R','C'): continue
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic(goal, nxt)
                    heapq.heappush(frontier,(priority,nxt)); came_from[nxt]=current
        if goal not in came_from: return []
        path=[]; cur=goal
        while cur!=start:
            path.append(cur); cur=came_from[cur]
        path.reverse(); return path
    def bfs_frontier(self, known, start):
        if start not in known: return None
        visited = set()
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur in visited: continue
            visited.add(cur)
            for nbr in self.grid.neighbors(cur):
                if nbr not in known:
                    return cur
            for nbr in self.grid.neighbors(cur):
                if nbr in known and known[nbr].explored and nbr not in visited:
                    q.append(nbr)
        return None

class Utility:
    def __init__(self, rover, grid, pathfinder): self.rover=rover; self.grid=grid; self.pathfinder=pathfinder
    def score(self, target):
        path = self.pathfinder.astar(self.rover.pos, target, self.rover.known)
        if not path: return -math.inf
        return 1.0 / (len(path) + 1e-6)

class Simulation:
    def __init__(self):
        self.cfg = Config()
        gen = MapGenerator(size=self.cfg.map_size, obstacles=self.cfg.obstacles,
                           number_rows=self.cfg.number_rows, number_columns=self.cfg.number_columns,
                           noise=self.cfg.noise, margins=self.cfg.margins,
                           obstacle_size=self.cfg.obstacle_size, seed=self.cfg.seed)
        map_array = gen.get_map()
        self.grid = LazyGrid(self.cfg, map_array)
        self.rover = Rover(self.cfg, self.grid)
        self.pathfinder = Pathfinder(self.grid)
        self.utility = Utility(self.rover, self.grid, self.pathfinder)
        self.visual = Visualizer(self.cfg, self.grid, self.rover)
        self.buttons = []
        self._create_buttons()
        self.running = True
        self.paused = True
        self.cfg.draw_grid = False
        self.cfg.draw_lidar = False
        self.cfg.draw_traceline = True
        self.first_pass_last = None
        self.in_first_pass = True
        self.first_pass_breadcrumbs = []
        self.breadcrumb_min_dist = 20

    def _create_buttons(self):
        btn_w, btn_h = 200, 34
        sx = self.cfg.map_w + 16
        sy = 20
        labels = ["Start", "Return to Base", "Second Pass", "Recharge", "Zoom In", "Zoom Out"]
        self.buttons = []
        for i, label in enumerate(labels):
            rect = pygame.Rect(sx, sy + i * (btn_h + 8), btn_w, btn_h)
            self.buttons.append((label, rect))

    def _draw_buttons(self):
        for label, rect in self.buttons:
            color = (60,60,60)
            pygame.draw.rect(self.visual.screen, color, rect)
            pygame.draw.rect(self.visual.screen, (180,180,180), rect, 1)
            font = self.visual.font
            txt = font.render(label, True, (220,220,220))
            self.visual.screen.blit(txt, (rect.x + 8, rect.y + 6))

    def handle_button_click(self, pos):
        for label, rect in self.buttons:
            if rect.collidepoint(pos):
                self._on_button(label)
                break

    def _on_button(self, label):
        if label == "Start":
            self.paused = False
            self.rover.second_pass = False
            self.in_first_pass = True
            self.rover.mode_trace_color = (255,255,0)

            # FULL reset of exploration state
            self.first_pass_last = None
            self.first_pass_breadcrumbs = []
            self.rover.history = [self.rover.base]
            self.rover.trace = []
            self.rover.path = []
            self.rover.returning = False
        elif label == "Return to Base":
            self.rover.returning = True
            return
        elif label == "Second Pass":
            self.rover.second_pass = not self.rover.second_pass
            if self.rover.second_pass:
                self.in_first_pass = False
                self.rover.mode_trace_color = (0,255,255)
                print("SECOND PASS TARGET =", self.first_pass_last)
                if self.first_pass_last is not None:
                    self._plan_second_pass_chain()
                    self.evaluate_passes_advanced(self.rover.path)
            else:
                self.in_first_pass = True
                self.rover.mode_trace_color = (255,255,0)
        elif label == "Recharge":
            if self.rover.pos == self.rover.base:
                self.rover.energy = self.cfg.rover_energy
                self.rover.returning = False
                self.rover.path = []
                self.paused = True
        elif label == "Zoom In":
            self.visual.zoom_to_rover(1.25)
        elif label == "Zoom Out":
            self.visual.zoom_to_rover(1/1.25)

    def select_best_target(self):
        targets = self.rover.visible_targets()
        best=None; best_score=-math.inf
        for t in targets:
            s=self.utility.score(t)
            if s>best_score: best=t; best_score=s
        return best

    def plan_and_set_path(self, target):
        path = self.pathfinder.astar(self.rover.pos, target, self.rover.known)
        self.rover.path = path

    def explore_step(self):
        f = self.pathfinder.bfs_frontier(self.rover.known, self.rover.pos)
        if f is None:
            return
        path = self.pathfinder.astar(self.rover.pos, f, self.rover.known)
        self.rover.path = path

    def _plan_second_pass_chain(self):
        crumbs = self.first_pass_breadcrumbs[:]
        if not crumbs:
            return
        chain = []
        start = self.rover.base
        for c in crumbs:
            seg = self.pathfinder.astar(start, c, self.rover.known)
            if seg:
                chain.extend(seg)
                start = c
            else:
                # if a segment fails, fallback to direct astar to the final target
                fallback = self.pathfinder.astar(self.rover.pos, self.first_pass_last, self.rover.known)
                if fallback:
                    chain = fallback
                break
        # finally add a* from last crumb to final first_pass_last if it's not the same
        if crumbs and crumbs[-1] != self.first_pass_last:
            seg = self.pathfinder.astar(start, self.first_pass_last, self.rover.known)
            if seg:
                chain.extend(seg)
        if chain:
            self.rover.path = chain

    def second_pass_step(self):
        if self.first_pass_last is not None:
            path = self.pathfinder.astar(self.rover.pos, self.first_pass_last, self.rover.known)
            if path:
                self.rover.path = path
                return
        f = self.pathfinder.bfs_frontier(self.rover.known, self.rover.pos)
        if f is None:
            return
        path = self.pathfinder.astar(self.rover.pos, f, self.rover.known)
        self.rover.path = path

    def _maybe_add_breadcrumb(self, pos):
        if not self.first_pass_breadcrumbs:
            self.first_pass_breadcrumbs.append(pos)
            return
        last = self.first_pass_breadcrumbs[-1]
        dx = pos[0]-last[0]; dy = pos[1]-last[1]
        dist = math.hypot(dx, dy)
        if dist >= self.breadcrumb_min_dist:
            if len(self.first_pass_breadcrumbs) >= 2:
                prev = self.first_pass_breadcrumbs[-2]
                v1 = (last[0]-prev[0], last[1]-prev[1])
                v2 = (pos[0]-last[0], pos[1]-last[1])
                if v1 != v2:
                    self.first_pass_breadcrumbs.append(pos)
            else:
                self.first_pass_breadcrumbs.append(pos)

    def step(self):
        if self.rover.steps_taken > self.cfg.max_steps:
            self.running = False; return

        if self.rover.path:
            before = self.rover.pos
            self.rover.move_step()
            after = self.rover.pos
            if not self.rover.path and self.in_first_pass:
                self.first_pass_last = after
                self._maybe_add_breadcrumb(after)
            return

        if (not self.rover.returning) and (self.rover.remaining_energy_fraction() <= self.cfg.return_threshold):
            self.first_pass_last = self.rover.pos
            print("FIRST PASS TRUE END =", self.first_pass_last)
            self.in_first_pass = False
            self.rover.returning = True

            crumbs = self.first_pass_breadcrumbs[:]
            # ensure breadcrumbs don't contain the rover.pos to avoid a giant jump
            if crumbs and crumbs[-1] == self.rover.pos:
                crumbs = crumbs[:-1]

            # return chain = reversed crumbs then base
            return_points = crumbs[::-1] + [self.rover.base]

            # build return path by chaining A* between consecutive return points
            full_path = []
            for i in range(len(return_points)-1):
                s = return_points[i]
                g = return_points[i+1]
                seg = self.pathfinder.astar(s, g, self.rover.known)
                if seg:
                    full_path.extend(seg)
                else:
                    # fallback: if any segment fails, use exact reversed history
                    history = self.rover.history
                    back = history[:-1][::-1]
                    full_path = back
                    break

            if full_path:
                self.rover.path = full_path
            return

        if self.rover.returning:
            if self.rover.pos == self.rover.base:
                self.rover.path=[]; return
            if not self.rover.path:
                # if path exhausted while returning, try to continue along reversed history as fallback
                history = self.rover.history
                remaining = history[:-1][::-1]
                if remaining:
                    self.rover.path = remaining
                return

        if self.rover.second_pass and (not self.rover.returning):
            targets = self.rover.visible_targets()
            if targets:
                best=self.select_best_target()
                if best:
                    self.plan_and_set_path(best); return
            self.second_pass_step(); return

        targets=self.rover.visible_targets()
        if targets and (not self.rover.returning):
            best=self.select_best_target()
            if best:
                self.plan_and_set_path(best); return

        self.explore_step()
    def evaluate_passes_advanced(self, second_pass_path):
        fp_hist = self.rover.history
        first_pass_dist = len(fp_hist)
        second_pass_dist = len(second_pass_path)

        time_saved = max(0, first_pass_dist - second_pass_dist)

        straight = 0
        for i in range(len(second_pass_path)-1):
            dx = second_pass_path[i+1][0] - second_pass_path[i][0]
            dy = second_pass_path[i+1][1] - second_pass_path[i][1]
            straight += math.hypot(dx, dy)

        branch_points = 0
        for i in range(2, len(fp_hist)):
            v1 = (fp_hist[i-1][0] - fp_hist[i-2][0],
                  fp_hist[i-1][1] - fp_hist[i-2][1])
            v2 = (fp_hist[i][0] - fp_hist[i-1][0],
                  fp_hist[i][1] - fp_hist[i-1][1])
            if v1 != v2:
                branch_points += 1

        turns_2p = 0
        for i in range(2, len(second_pass_path)):
            v1 = (second_pass_path[i-1][0] - second_pass_path[i-2][0],
                  second_pass_path[i-1][1] - second_pass_path[i-2][1])
            v2 = (second_pass_path[i][0] - second_pass_path[i-1][0],
                  second_pass_path[i][1] - second_pass_path[i-1][1])
            if v1 != v2:
                turns_2p += 1

        explored_cells = len(self.rover.known)
        total_cells = self.cfg.map_w * self.cfg.map_h
        coverage = explored_cells / total_cells

        entropy = 0
        for i in range(1, len(fp_hist)):
            dx = abs(fp_hist[i][0] - fp_hist[i-1][0])
            dy = abs(fp_hist[i][1] - fp_hist[i-1][1])
            if dx + dy > 0:
                entropy += 1
        if len(fp_hist) > 0:
            entropy /= len(fp_hist)

        straightness_factor = straight / max(second_pass_dist, 1)

        redundancy = first_pass_dist / max(branch_points+1, 1)

        frontier_efficiency = (branch_points+1) / max(first_pass_dist, 1)

        if first_pass_dist > 0:
            heuristic_efficiency = second_pass_dist / first_pass_dist
        else:
            heuristic_efficiency = 1

        if hasattr(self.pathfinder, 'last_expanded'):
            expanded = self.pathfinder.last_expanded
        else:
            expanded = -1

        opt_dev = abs(second_pass_dist - straight)

        curvature = turns_2p / max(len(second_pass_path),1)

        print("\n------ ADVANCED PASS ANALYSIS ------")
        print("First Pass Distance:", first_pass_dist)
        print("Second Pass Distance:", second_pass_dist)
        print("Distance Saved:", time_saved)
        print("Straight-Line Second-Pass Path:", round(straight,2))
        print("Straightness Factor:", round(straightness_factor,3))
        print("Curvature (Second Pass):", round(curvature,3))
        print("Branchiness (First Pass):", branch_points)
        print("Redundancy Ratio:", round(redundancy,3))
        print("Frontier Efficiency:", round(frontier_efficiency,3))
        print("Exploration Coverage:", round(coverage*100,3), "%")
        print("Exploration Entropy:", round(entropy,3))
        print("Heuristic Efficiency:", round(heuristic_efficiency,3))
        print("Second Pass Optimality Deviation:", round(opt_dev,3))
        print("A* Nodes Expanded:", expanded)
        print("------------------------------------\n")

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running=False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.__init__()
                    elif event.key == pygame.K_ESCAPE:
                        self.running=False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
                    mouse_pos = event.pos
                    if mouse_pos[0] >= self.cfg.map_w:
                        self.handle_button_click(mouse_pos)
                elif event.type == pygame.MOUSEWHEEL:
                    mx,my = pygame.mouse.get_pos()
                    if mx < self.cfg.map_w:
                        if event.y > 0: self.visual.zoom_to_rover(1.15)
                        else: self.visual.zoom_to_rover(1/1.15)

            if not self.paused:
                self.step()

            self.visual.draw()
            self._draw_buttons()

            status_x = self.cfg.map_w + 16
            status_y = 20 + len(self.buttons) * 42
            font = self.visual.font
            ef = self.rover.remaining_energy_fraction()
            txt_energy = font.render(f'Energy: {self.rover.energy:.1f} ({ef*100:.0f}%)', True, (220,220,220))
            self.visual.screen.blit(txt_energy, (status_x, status_y))
            mode = "RETURNING" if self.rover.returning else ("SECOND PASS" if self.rover.second_pass else "EXPLORING")
            txt_mode = font.render(f'Mode: {mode}', True, (200,200,140))
            self.visual.screen.blit(txt_mode, (status_x, status_y + 18))
            txt_samples = font.render(f'Samples: {self.rover.samples}', True, (220,220,220))
            self.visual.screen.blit(txt_samples, (status_x, status_y + 36))

            pygame.display.flip()
            self.visual.clock.tick(self.cfg.fps)

        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
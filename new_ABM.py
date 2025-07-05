import pygame
import random
import numpy as np
import copy

# approximate meters per degree latitude/longitude around Amsterdam
M_PER_DEG = 111320

def compute_distance_m(path, idx):
    total = 0.0
    pts = [pygame.math.Vector2(lon, lat) for lat, lon in path]
    for i in range(idx, len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        mean_lat = np.deg2rad((a.y + b.y) / 2)
        dlat = (b.y - a.y) * M_PER_DEG
        dlon = (b.x - a.x) * M_PER_DEG * np.cos(mean_lat)
        total += np.hypot(dlat, dlon)
    return total


class Aircraft:
    next_id = 1

    def __init__(self, path_id, path, initial_speed, acceleration, min_speed, max_speed):
        self.id = Aircraft.next_id
        Aircraft.next_id += 1
        self.path_id = path_id
        self.path = [pygame.math.Vector2(lon, lat) for lat, lon in path]
        self.position = self.path[0].copy()
        self.speed = initial_speed
        self.target_speed = initial_speed
        self.acceleration = acceleration
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.segment = 0

    def set_target_speed(self, new_speed):
        self.target_speed = max(self.min_speed, min(self.max_speed, new_speed))

    def update(self, dt):
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + self.acceleration * dt)
        else:
            self.speed = max(self.target_speed, self.speed - self.acceleration * dt)
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        if self.segment >= len(self.path) - 1:
            return
        target = self.path[self.segment + 1]
        direction = target - self.position
        dist_deg = direction.length()
        if dist_deg == 0:
            self.segment += 1
            return
        travel_deg = (self.speed * dt) / M_PER_DEG
        if travel_deg >= dist_deg:
            self.position = target.copy()
            self.segment += 1
        else:
            self.position += direction.normalize() * travel_deg

    def distance_to(self, other):
        lat1, lon1 = self.position.y, self.position.x
        lat2, lon2 = other.position.y, other.position.x
        mean_lat = np.deg2rad((lat1 + lat2) / 2)
        dlat = (lat2 - lat1) * M_PER_DEG
        dlon = (lon2 - lon1) * M_PER_DEG * np.cos(mean_lat)
        return np.hypot(dlat, dlon)

    def is_finished(self):
        return self.segment >= len(self.path) - 1


import time

# class GlobalPSO:
#     def __init__(self, aircraft_list, min_speed, max_speed, dt, min_sep,
#                  sep_weight, fuel_weight,
#                  n_particles, w, c1, c2, max_iter,
#                  horizon_steps=10):
#         self.acs = aircraft_list
#         self.m = len(aircraft_list)
#         self.E = np.full(self.m, min_speed)
#         self.L = np.full(self.m, max_speed)
#         self.dt = dt
#         self.min_sep = min_sep
#         self.sep_w = sep_weight
#         self.fuel_w = fuel_weight
#         self.w, self.c1, self.c2 = w, c1, c2
#         self.max_iter = max_iter
#         self.horizon = horizon_steps

#         self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
#         self.V = np.zeros_like(self.X)
#         self.pbest = self.X.copy()
#         self.pbest_cost = np.full(n_particles, np.inf)
#         self.gbest = None
#         self.gbest_cost = np.inf

#     def _simulate_cost(self, speeds):
#         """
#         Forward-simulate the given speed vector over a horizon and compute a combined separation cost
#         speeds: candidate speeds for each aircraft
#         return: weighted sum of penalties
#         """
#         # # --- profile timers ---
#         # t0 = time.perf_counter()

#         # # 1. Deep-copy each aircraft
#         # sims = [copy.deepcopy(ac) for ac in self.acs]
#         # t_copy = time.perf_counter()

#         # sep_pen = fuel_pen = 0.0 
#         # # record initial speeds for velocity‐penalty
#         # init_speeds = [ac.speed for ac in sims]

#         # # 2. Main simulation loop
#         # t_sim = 0.0
#         # for _ in range(self.horizon):
#         #     step_start = time.perf_counter()
#         #     # a) apply speeds & update positions
#         #     for ac, s in zip(sims, speeds):
#         #         ac.speed = np.clip(s, ac.min_speed, ac.max_speed)
#         #         ac.update(self.dt)
#         #     t_after_move = time.perf_counter()

#         #     # b) separation checks
#         #     for i in range(self.m):
#         #         for j in range(i+1, self.m):
#         #             d = sims[i].distance_to(sims[j])
#         #             if d < self.min_sep:
#         #                 sep_pen += (self.min_sep - d)**2
#         #     t_after_sep = time.perf_counter()

#         #     # accumulate simulation loop time
#         #     t_sim += (t_after_move - step_start) + (t_after_sep - t_after_move)

#         # # 3. Velocity‐deviation penalty
#         # fuel_pen = sum((s - init_s)**2 for s, init_s in zip(speeds, init_speeds))
#         # t_vel = time.perf_counter()

#         # cost = self.sep_w * sep_pen + self.fuel_w * fuel_pen
#         # t_end = time.perf_counter()

#         # print({
#         #         'copy_time':      t_copy - t0,
#         #         'move_time':      (t_after_move - t0) - (t_copy - t0) if False else t_after_move - t_copy,
#         #         'sep_time':       t_after_sep - t_after_move,
#         #         'vel_pen_time':   t_vel - t_after_sep,
#         #         'total_time':     t_end - t0,
#         #         'simulate_steps': self.horizon,
#         #         'pair_checks':    self.horizon * (self.m*(self.m-1)//2)
#         #     })

#         # return cost

#         # Record each aircraft’s starting speed
#         init_speeds = [ac.speed for ac in self.acs]

#         sims = [copy.deepcopy(ac) for ac in self.acs]
#         sep_pen = fuel_pen = 0.0
        
#         # Simulate for a fixed number of time steps
#         for t in range(self.horizon):
#             # Apply candidate speeds 
#             for ac, s in zip(sims, speeds):
#                 # Clip to aircraft's allowable speed bounds
#                 ac.speed = np.clip(s, ac.min_speed, ac.max_speed)
#                 # Move along the path segment, respecting acceleration inside ac.update()
#                 ac.update(self.dt)
            
#             # After moving, check every pair for separation violations
            
#             # Conflict check only every 50 steps
#             if t % 50 == 0:
#                 for i in range(self.m):
#                     for j in range(i + 1, self.m):
#                         # Compute horizontal distance between clone i and j
#                         d = sims[i].distance_to(sims[j])
#                         # If too close, penalize squared gap below minimum separation
#                         if d < self.min_sep:
#                             sep_pen += (self.min_sep - d) ** 2
        
#         # After simulating, penalize deviations from initial speeds
#         for s, init_s in zip(speeds, init_speeds):
#             fuel_pen += (s - init_s)**2
        
#         return self.sep_w * sep_pen + self.fuel_w * fuel_pen

#     def optimize(self):
#         nP = self.X.shape[0]
#         for _ in range(self.max_iter):
#             for k in range(nP):
#                 cost = self._simulate_cost(self.X[k])
#                 if cost < self.pbest_cost[k]:
#                     self.pbest_cost[k] = cost
#                     self.pbest[k] = self.X[k].copy()
#                 if cost < self.gbest_cost:
#                     self.gbest_cost = cost
#                     self.gbest = self.X[k].copy()
#             r1, r2 = np.random.rand(nP, self.m), np.random.rand(nP, self.m)
#             self.V = ( self.w * self.V
#                      + self.c1 * r1 * (self.pbest - self.X)
#                      + self.c2 * r2 * (self.gbest - self.X) )
#             self.X = np.clip(self.X + self.V, self.E, self.L)
#         return self.gbest.copy()


import numpy as np
import copy
import bisect

class GlobalPSO:
    def __init__(self, aircraft_list, min_speed, max_speed, dt, min_sep,
                 sep_weight, fuel_weight,
                 n_particles, w, c1, c2, max_iter,
                 horizon_steps=3000,  # total look-ahead steps
                 step_skip=15        # only sample every 50 steps
    ):
        self.acs       = aircraft_list
        self.m         = len(aircraft_list)
        self.E         = np.full(self.m, min_speed)
        self.L         = np.full(self.m, max_speed)
        self.dt        = dt
        self.min_sep   = min_sep
        self.sep_w     = sep_weight
        self.fuel_w    = fuel_weight
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_iter  = max_iter
        self.horizon   = horizon_steps
        self.step_skip = step_skip

        # Precompute each aircraft’s path cumulative‐length (in metres)
        self.path_cumlens = []
        self.path_pts     = []
        for ac in self.acs:
            # ac.path is list of Vector2(lon,lat)
            pts = ac.path
            dist = [0.0]
            for A, B in zip(pts[:-1], pts[1:]):
                mean_lat = np.deg2rad((A.y + B.y)/2)
                dlat = (B.y - A.y)*M_PER_DEG
                dlon = (B.x - A.x)*M_PER_DEG*np.cos(mean_lat)
                dist.append(np.hypot(dlat, dlon))
            cum = np.cumsum(dist)
            self.path_cumlens.append(cum)
            self.path_pts.append(pts)

        # PSO arrays as before…
        self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _simulate_cost(self, speeds):
        # 1) record initial speeds for velocity penalty
        init_speeds = np.array([ac.speed for ac in self.acs])

        sep_pen = 0.0
        # build sparse sample times: t = 0, K, 2K, ..., horizon
        samples = np.arange(0, self.horizon+1, self.step_skip)

        for t in samples:
            τ = t * self.dt
            # compute each AC’s travelled distance
            D = speeds * τ  # vector length m

            # lookup each AC’s position by bisect into cumlen
            positions = []
            for i in range(self.m):
                cum = self.path_cumlens[i]
                pts = self.path_pts[i]
                d = min(D[i], cum[-1])
                idx = bisect.bisect_right(cum, d) - 1
                if idx >= len(pts)-1:
                    positions.append(pts[-1])
                else:
                    A, B = pts[idx], pts[idx+1]
                    seg_len = cum[idx+1] - cum[idx]
                    frac = (d - cum[idx]) / seg_len if seg_len>0 else 0.0
                    positions.append(A + (B - A)*frac)

            # pairwise separation check
            for i in range(self.m):
                for j in range(i+1, self.m):
                    d_ij = positions[i].distance_to(positions[j])
                    if d_ij < self.min_sep:
                        sep_pen += (self.min_sep - d_ij)**2

        # 3) velocity‐deviation penalty
        vel_pen = np.sum((speeds - init_speeds)**2)

        return self.sep_w * sep_pen + self.fuel_w * vel_pen

    def optimize(self):
        nP = self.X.shape[0]
        for _ in range(self.max_iter):
            for k in range(nP):
                cost = self._simulate_cost(self.X[k])
                if cost < self.pbest_cost[k]:
                    self.pbest_cost[k] = cost
                    self.pbest[k] = self.X[k].copy()
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest = self.X[k].copy()
            r1, r2 = np.random.rand(nP, self.m), np.random.rand(nP, self.m)
            self.V = ( self.w * self.V
                     + self.c1 * r1 * (self.pbest - self.X)
                     + self.c2 * r2 * (self.gbest - self.X) )
            self.X = np.clip(self.X + self.V, self.E, self.L)
        return self.gbest.copy()


class ATCAgent:
    def __init__(self,
                 min_speed, max_speed, accel,
                 min_sep, sep_weight, fuel_weight,
                 particles, w, c1, c2, iters, horizon):
        self.min_speed   = min_speed
        self.max_speed   = max_speed
        self.accel       = accel
        self.min_sep     = min_sep
        self.sep_weight  = sep_weight
        self.fuel_weight = fuel_weight

        self.particles = particles
        self.w         = w
        self.c1        = c1
        self.c2        = c2
        self.iters     = iters
        self.horizon   = horizon

    def communicate(self, aircraft_list, dt):
        # only unfinished aircraft
        acs = [ac for ac in aircraft_list if not ac.is_finished()]
        if len(acs) < 2:
            return

        pso = GlobalPSO(
            acs,
            self.min_speed, self.max_speed,
            dt, self.min_sep,
            self.sep_weight, self.fuel_weight,
            self.particles,
            self.w, self.c1, self.c2,
            self.iters,
            horizon_steps=self.horizon
        )
        best_speeds = pso.optimize()

        for ac, s in zip(acs, best_speeds):
            ac.set_target_speed(s)


class Visualization:
    ILS_LAT = 52.33080945036779
    ILS_LON = 5.32394574767848

    PALETTE = [
        (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
        (255, 255,   0), (255,   0, 255), (  0, 255, 255),
        (128,   0, 128), (255, 165,   0), (  0, 128, 128),
        (128, 128,   0),
    ]

    def __init__(
        self,
        paths_dict,
        width=800,
        height=600,
        initial_speed=100.0,
        acceleration=1.0,
        min_speed=50.0,
        max_speed=200.0,
        spawn_interval=2.0,
        pso_interval=1.0,       # <-- add here
        time_scale=1.0,
        fps=60,
        min_sep=5000.0,
        sep_weight=1000.0,
        fuel_weight=1.0,
        pso_particles=10,
        pso_w=0.5,
        pso_c1=1.2,
        pso_c2=1.2,
        pso_iters=20,
        pso_horizon=10,
        ils_radius_m=10000.0
    ):
        # store params…
        self.paths = paths_dict
        self.width, self.height = width, height
        self.initial_speed = initial_speed
        self.acceleration  = acceleration
        self.spawn_interval= spawn_interval
        self.time_scale   = time_scale
        self.fps          = fps
        self.spawn_interval = spawn_interval
        self.pso_interval   = pso_interval    
        self._pso_timer     = 0.0
        self.sim_time = 0.0

        self.min_speed  = min_speed
        self.max_speed  = max_speed
        self.min_sep    = min_sep
        self.sep_weight = sep_weight
        self.fuel_weight= fuel_weight

        # Global PSO ATC
        self.atc = ATCAgent(
            min_speed, max_speed, acceleration,
            min_sep, sep_weight, fuel_weight,
            pso_particles, pso_w, pso_c1, pso_c2,
            pso_iters, pso_horizon
        )

        # compute drawing scales…
        all_lats = [pt[0] for p in paths_dict.values() for pt in p]
        all_lons = [pt[1] for p in paths_dict.values() for pt in p]
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)
        self.px_per_deg = self.width / (self.lon_max - self.lon_min)
        self.sep_radius_px = int((min_sep / M_PER_DEG) * self.px_per_deg)
        self.ils_px = self._geo_to_px(self.ILS_LAT, self.ILS_LON)
        self.ils_radius_px = int((ils_radius_m / M_PER_DEG) * self.px_per_deg)

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.font   = pygame.font.SysFont(None, 20)


    def _geo_to_px(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        y = self.height - (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.height
        return int(x), int(y)

    def draw_paths(self):
        active = {ac.path_id for ac in self.aircraft_list}
        for pid in active:
            pts = [self._geo_to_px(lat, lon) for lat, lon in self.paths[pid]]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (200, 200, 200), False, pts, 2)

    def spawn_aircraft(self):
        pid, path = random.choice(list(self.paths.items()))
        ac = Aircraft(
            pid, path,
            self.initial_speed, self.acceleration,
            self.min_speed, self.max_speed
        )
        self.aircraft_list.append(ac)

    def run(self):
        clock = pygame.time.Clock()
        self.aircraft_list = []
        spawn_timer = 0.0
        self._pso_timer = 0.0
        self.sim_time = 0.0

        running = True
        while running:
            raw_dt = clock.tick(self.fps) / 1000.0
            dt = raw_dt * self.time_scale

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            # Spawn logic
            spawn_timer += dt
            if spawn_timer >= self.spawn_interval:
                self.spawn_aircraft()
                spawn_timer = 0.0

            # PSO invocation every pso_interval seconds
            self._pso_timer += dt
            if self._pso_timer >= self.pso_interval:
                self.atc.communicate(self.aircraft_list, dt)
                self._pso_timer = 0.0

            # Update aircraft
            for ac in self.aircraft_list[:]:
                ac.update(dt)
                if ac.is_finished():
                    self.aircraft_list.remove(ac)
            
            self.sim_time += dt

            # Drawing
            self.screen.fill((255,255,255))
            self.draw_paths()
            # pygame.draw.circle(self.screen, (0,255,0), self.ils_px, self.ils_radius_px, 2)

            for ac in self.aircraft_list:
                x, y = self._geo_to_px(ac.position.y, ac.position.x)
                color = Visualization.PALETTE[ac.id % len(Visualization.PALETTE)]
                pygame.draw.circle(self.screen, color, (x,y), 4)
                lbl = self.font.render(f"{ac.id}|{ac.speed:.0f}", True, (0,0,0))
                self.screen.blit(lbl, (x+6, y-6))
                pygame.draw.circle(self.screen, color, (x,y), self.sep_radius_px, 1)

            # render time in top-left corner
            time_text = f"Time: {self.sim_time:.1f}s"
            img = self.font.render(time_text, True, (0, 0, 0))
            self.screen.blit(img, (10, 10))

            pygame.display.flip()

        pygame.quit()


def runme(
    paths_dict,
    width=1024,
    height=768,
    initial_speed=100.0,
    acceleration=0.5,
    min_speed=80.0,
    max_speed=130.0,
    spawn_interval=400.0,
    pso_interval=200.0,        
    time_scale=100.0,
    fps=30,
    min_sep=3000.0,
    sep_weight=10,
    fuel_weight=1,
    pso_particles=20,
    pso_w=0.5,
    pso_c1=1.2,
    pso_c2=1.2,
    pso_iters=20,
    pso_horizon=800,
    ils_radius_m=400000.0
):
    sim = Visualization(
        paths_dict,
        width, height,
        initial_speed, acceleration,
        min_speed, max_speed,
        spawn_interval, pso_interval,
        time_scale, fps,
        min_sep, sep_weight, fuel_weight,
        pso_particles, pso_w, pso_c1, pso_c2,
        pso_iters, pso_horizon,
        ils_radius_m
    )
    sim.run()


if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

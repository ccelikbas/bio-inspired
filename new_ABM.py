import pygame
import random
import numpy as np
import copy
import bisect

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

def _find_clusters(acs, radius):
    # Build adjacency list
    adj = {ac: [] for ac in acs}
    for i, a in enumerate(acs):
        for b in acs[i+1:]:
            if a.distance_to(b) < radius:
                adj[a].append(b)
                adj[b].append(a)
    # Extract connected components via DFS
    clusters, seen = [], set()
    for a in acs:
        if a in seen: continue
        stack, comp = [a], []
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    stack.append(v)
        if len(comp) > 1:
            clusters.append(comp)
    return clusters


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
                 particles, w, c1, c2, iters,
                 global_horizon, global_step_skip,
                 local_comm_radius, local_horizon, local_sep_weight):
        """
        - global_horizon:    # of discrete steps to look ahead in the GLOBAL PSO
        - global_step_skip:  sample interval (in steps) when simulating cost
                             (sparsifies separation checks)
        - local_comm_radius: radius (m) within which to form LOCAL clusters
        - local_horizon:     # of steps to look ahead in each LOCAL PSO
        - local_sep_weight:  separation‐penalty weight used in LOCAL PSO
        """
        self.min_speed          = min_speed
        self.max_speed          = max_speed
        self.accel              = accel
        self.min_sep            = min_sep
        self.sep_weight         = sep_weight
        self.fuel_weight        = fuel_weight

        self.particles          = particles
        self.w                  = w
        self.c1                 = c1
        self.c2                 = c2
        self.iters              = iters

        self.global_horizon     = global_horizon
        self.global_step_skip   = global_step_skip

        self.local_comm_radius  = local_comm_radius
        self.local_horizon      = local_horizon
        self.local_sep_weight   = local_sep_weight

    def communicate(self, aircraft_list, dt):
        # 1) select only unfinished aircraft
        acs = [ac for ac in aircraft_list if not ac.is_finished()]
        if len(acs) < 2:
            return

        # 2) GLOBAL PSO over all ACs
        global_pso = GlobalPSO(
            acs,
            self.min_speed, self.max_speed,
            dt, self.min_sep,
            self.sep_weight, self.fuel_weight,
            self.particles,
            self.w, self.c1, self.c2,
            self.iters,
            horizon_steps=self.global_horizon,
            step_skip=self.global_step_skip
        )
        global_speeds = global_pso.optimize()
        for ac, s in zip(acs, global_speeds):
            ac.set_target_speed(s)

        # 3) LOCAL PSO on any clusters within local_comm_radius
        clusters = _find_clusters(acs, self.local_comm_radius)
        for cluster in clusters:
            local_pso = GlobalPSO(
                cluster,
                self.min_speed, self.max_speed,
                dt, self.min_sep,
                self.local_sep_weight, self.fuel_weight,
                max(10, len(cluster)*2),
                self.w, self.c1, self.c2,
                self.iters,
                horizon_steps=self.local_horizon,
                step_skip=1
            )
            local_speeds = local_pso.optimize()
            for ac, s in zip(cluster, local_speeds):
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
        initial_speed=100.0,      # m/s at spawn
        acceleration=1.0,         # m/s² accel/decel
        min_speed=50.0,           # m/s lower bound
        max_speed=200.0,          # m/s upper bound
        spawn_interval=2.0,       # s between new AC spawns
        pso_interval=1.0,         # s between ATC replans
        time_scale=1.0,           # sim_time = real_dt * time_scale
        fps=60,                   # display frames per second
        min_sep=5000.0,           # m safety distance
        sep_weight=1000.0,        # global PSO sep‐penalty
        fuel_weight=1.0,          # global PSO vel‐penalty
        pso_particles=10,         # swarm size
        pso_w=0.5,                # PSO inertia
        pso_c1=1.2,               # PSO cognitive
        pso_c2=1.2,               # PSO social
        pso_iters=20,             # global PSO iterations
        horizon_steps=800,        # global PSO look‐ahead steps
        step_skip=15,             # sample every N steps in global sim
        local_comm_radius=20000,  # m: range for local clusters
        local_horizon=40,         # local PSO look‐ahead steps
        local_sep_weight=10000.0, # sep‐penalty in local PSO
        ils_radius_m=10000.0      # m, for ILS circle drawing
    ):
        # store for drawing & spawn logic
        self.paths           = paths_dict
        self.width, self.height = width, height
        self.initial_speed   = initial_speed
        self.acceleration    = acceleration
        self.spawn_interval  = spawn_interval
        self.time_scale      = time_scale
        self.fps             = fps
        self.pso_interval    = pso_interval

        # separation & PSO parameters
        self.min_speed       = min_speed
        self.max_speed       = max_speed
        self.min_sep         = min_sep
        self.sep_weight      = sep_weight
        self.fuel_weight     = fuel_weight

        # build the combined ATC agent
        self.atc = ATCAgent(
            min_speed, max_speed, acceleration,
            min_sep, sep_weight, fuel_weight,
            pso_particles, pso_w, pso_c1, pso_c2, pso_iters,
            global_horizon=horizon_steps,
            global_step_skip=step_skip,
            local_comm_radius=local_comm_radius,
            local_horizon=local_horizon,
            local_sep_weight=local_sep_weight
        )

        # drawing scale
        all_lats = [pt[0] for p in paths_dict.values() for pt in p]
        all_lons = [pt[1] for p in paths_dict.values() for pt in p]
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)
        self.px_per_deg    = width / (self.lon_max - self.lon_min)
        self.sep_radius_px = int((min_sep / M_PER_DEG) * self.px_per_deg)
        self.ils_px        = self._geo_to_px(self.ILS_LAT, self.ILS_LON)
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

            # Find any aircraft pairs closer than min_sep
            conflict_ids = set()
            for i, ac1 in enumerate(self.aircraft_list):
                for ac2 in self.aircraft_list[i+1:]:
                    if ac1.distance_to(ac2) < self.min_sep:
                        conflict_ids.add(ac1.id)
                        conflict_ids.add(ac2.id)

            # Draw each aircraft in red if in conflict, or black otherwise
            for ac in self.aircraft_list:
                x, y = self._geo_to_px(ac.position.y, ac.position.x)
                color = (255, 0, 0) if ac.id in conflict_ids else (0, 0, 0)

                # draw the aircraft dot
                pygame.draw.circle(self.screen, color, (x, y), 4)

                # render label and blit it *below* the dot, centered
                lbl = self.font.render(f"{ac.id} | {ac.speed:.0f}", True, color)
                label_x = x + 10
                label_y = y + 6   # 6 pixels below the center of the dot
                self.screen.blit(lbl, (label_x, label_y))

                # draw the separation circle
                pygame.draw.circle(self.screen, color, (x, y), self.sep_radius_px, 1)


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
    initial_speed=100.0,      # m/s spawn speed
    acceleration=0.5,         # m/s² accel/decel
    min_speed=80.0,           # m/s lower bound
    max_speed=130.0,          # m/s upper bound
    spawn_interval=400.0,     # s between new AC
    pso_interval=200.0,       # s between replans
    time_scale=100.0,         # real-to-sim time
    fps=30,                   # display FPS
    min_sep=3000.0,           # m safety distance
    sep_weight=10,            # global sep penalty weight
    fuel_weight=1,            # global vel penalty weight
    pso_particles=20,         # swarm size
    pso_w=0.5,                # PSO inertia
    pso_c1=1.2,               # PSO cognitive
    pso_c2=1.2,               # PSO social
    pso_iters=20,             # PSO iterations
    horizon_steps=800,        # global look-ahead steps
    step_skip=15,             # sample interval in global sim
    local_comm_radius=20000,  # m: local cluster radius
    local_horizon=40,         # steps in local look-ahead
    local_sep_weight=10000.0, # sep-penalty weight in local PSO
    ils_radius_m=400000.0     # m, ILS circle for drawing
):
    """
    All five new parameters are here and documented above.
    """
    sim = Visualization(
        paths_dict,
        width, height,
        initial_speed, acceleration,
        min_speed, max_speed,
        spawn_interval, pso_interval,
        time_scale, fps,
        min_sep, sep_weight, fuel_weight,
        pso_particles, pso_w, pso_c1, pso_c2,
        pso_iters,
        horizon_steps, step_skip,
        local_comm_radius, local_horizon, local_sep_weight,
        ils_radius_m
    )
    sim.run()


if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

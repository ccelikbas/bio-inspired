import pygame
import random
import numpy as np
import copy

# approximate meters per degree latitude/longitude around Amsterdam
M_PER_DEG = 111320  


def compute_distance_m(path, idx):
    """Remaining distance along path (in meters) from segment idx."""
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

        self.speed = initial_speed      # m/s
        self.target_speed = initial_speed
        self.acceleration = acceleration  # m/s²
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.segment = 0

    def set_target_speed(self, new_speed):
        # clamp to [min_speed, max_speed]
        self.target_speed = max(self.min_speed, min(self.max_speed, new_speed))

    def update(self, dt):
        # accelerate/decelerate toward target
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + self.acceleration * dt)
        else:
            self.speed = max(self.target_speed, self.speed - self.acceleration * dt)
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        # move along the path
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


class LocalPSO:
    def __init__(self, aircraft_list, min_speed, max_speed, dt, min_sep,
                 sep_weight, fuel_weight,
                 n_particles, w, c1, c2, max_iter,
                 horizon_steps=10):
        self.acs = aircraft_list
        self.m = len(aircraft_list)
        self.E = np.full(self.m, min_speed)
        self.L = np.full(self.m, max_speed)
        self.dt = dt
        self.min_sep = min_sep
        self.sep_w = sep_weight
        self.fuel_w = fuel_weight
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_iter = max_iter
        self.horizon = horizon_steps

        # PSO state
        self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _simulate_cost(self, speeds):
        sims = [copy.deepcopy(ac) for ac in self.acs]
        sep_pen = fuel_pen = 0.0

        for _ in range(self.horizon):
            for ac, s in zip(sims, speeds):
                ac.speed = max(ac.min_speed, min(ac.max_speed, s))
                ac.update(self.dt)
                fuel_pen += s**2
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    d = sims[i].distance_to(sims[j])
                    if d < self.min_sep:
                        sep_pen += (self.min_sep - d)**2

        return self.sep_w * sep_pen + self.fuel_w * fuel_pen

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
            self.V = (self.w * self.V
                      + self.c1 * r1 * (self.pbest - self.X)
                      + self.c2 * r2 * (self.gbest - self.X))
            self.X = np.clip(self.X + self.V, self.E, self.L)
        return self.gbest.copy()


class ATCAgent:
    def __init__(self,
                 min_speed, max_speed, accel,
                 min_sep, sep_weight, fuel_weight,
                 local_particles, local_w, local_c1, local_c2, local_iter, local_horizon,
                 cluster_factor):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.accel = accel
        self.min_sep = min_sep
        self.sep_weight = sep_weight
        self.fuel_weight = fuel_weight

        self.local_particles = local_particles
        self.local_w = local_w
        self.local_c1 = local_c1
        self.local_c2 = local_c2
        self.local_iter = local_iter
        self.local_horizon = local_horizon

        self.cluster_factor = cluster_factor
        self.last_clusters = []

    def _find_clusters(self, ac_list):
        clusters, unvisited = [], set(ac_list)
        while unvisited:
            seed = unvisited.pop()
            stack, cluster = [seed], {seed}
            while stack:
                a = stack.pop()
                for b in list(unvisited):
                    if a.distance_to(b) < self.cluster_factor * self.min_sep:
                        unvisited.remove(b)
                        cluster.add(b)
                        stack.append(b)
            clusters.append(list(cluster))
        return clusters

    def communicate(self, aircraft_list, dt):
        clusters = self._find_clusters(aircraft_list)
        self.last_clusters = clusters
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            # always use local PSO for clusters of size >= 2
            local_pso = LocalPSO(
                cluster,
                self.min_speed, self.max_speed,
                dt, self.min_sep,
                self.sep_weight, self.fuel_weight,
                self.local_particles,
                self.local_w, self.local_c1, self.local_c2,
                self.local_iter,
                horizon_steps=self.local_horizon
            )
            best_speeds = local_pso.optimize()
            for ac, s in zip(cluster, best_speeds):
                ac.set_target_speed(s)


class Visualization:
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
        initial_speed=100.0,    # m/s
        acceleration=1.0,       # m/s²
        min_speed=50.0,
        max_speed=200.0,
        spawn_interval=2.0,
        time_scale=1.0,
        fps=60,
        min_sep=5000.0,
        sep_weight=1000.0,
        fuel_weight=1.0,
        local_particles=10,
        local_w=0.5,
        local_c1=1.2,
        local_c2=1.2,
        local_iter=20,
        local_horizon=10,
        cluster_factor=2.0
    ):
        self.paths = paths_dict
        self.width = width
        self.height = height
        self.initial_speed = initial_speed
        self.acceleration = acceleration
        self.spawn_interval = spawn_interval
        self.time_scale = time_scale
        self.fps = fps

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.min_sep = min_sep
        self.sep_weight = sep_weight
        self.fuel_weight = fuel_weight

        self.atc = ATCAgent(
            min_speed, max_speed, acceleration,
            min_sep, sep_weight, fuel_weight,
            local_particles, local_w, local_c1, local_c2, local_iter, local_horizon,
            cluster_factor
        )

        all_lats = [pt[0] for path in paths_dict.values() for pt in path]
        all_lons = [pt[1] for path in paths_dict.values() for pt in path]
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)
        self.px_per_deg = self.width / (self.lon_max - self.lon_min)
        self.sep_radius_px = int((min_sep / M_PER_DEG) * self.px_per_deg)

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Clustered Aircraft Simulation")
        self.font = pygame.font.SysFont(None, 20)

    def latlon_to_screen(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        y = self.height - (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.height
        return x, y

    def draw_paths(self):
        active = {ac.path_id for ac in self.aircraft_list}
        for pid in active:
            pts = [self.latlon_to_screen(lat, lon) for lat, lon in self.paths[pid]]
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

        running = True
        while running:
            raw_dt = clock.tick(self.fps) / 1000.0
            dt = raw_dt * self.time_scale

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            # spawn
            spawn_timer += dt
            if spawn_timer >= self.spawn_interval:
                self.spawn_aircraft()
                spawn_timer = 0.0

            # local PSO per cluster
            self.atc.communicate(self.aircraft_list, dt)

            # update positions
            for ac in self.aircraft_list[:]:
                ac.update(dt)
                if ac.is_finished():
                    self.aircraft_list.remove(ac)

            # render
            self.screen.fill((255, 255, 255))
            self.draw_paths()

            # color by cluster
            cluster_map = {}
            for idx, cluster in enumerate(self.atc.last_clusters):
                if len(cluster) < 2:
                    continue
                for ac in cluster:
                    cluster_map[ac.id] = idx

            for ac in self.aircraft_list:
                x, y = self.latlon_to_screen(ac.position.y, ac.position.x)
                idx = cluster_map.get(ac.id)
                color = (200, 200, 200) if idx is None else Visualization.PALETTE[idx % len(Visualization.PALETTE)]
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 4)
                label = self.font.render(f"{ac.id}|{ac.speed:.0f}", True, (0, 0, 0))
                self.screen.blit(label, (x + 6, y - 6))
                pygame.draw.circle(self.screen, color, (int(x), int(y)), self.sep_radius_px, 1)

            pygame.display.flip()

        pygame.quit()


def runme(
    paths_dict,
    width=1024,
    height=768,
    initial_speed=150.0,
    acceleration=1.0,
    min_speed=50.0,
    max_speed=200.0,
    spawn_interval=150.0,
    time_scale=100.0,
    fps=30,
    min_sep=3000.0,
    sep_weight=500.0,
    fuel_weight=0,
    local_particles=10,
    local_w=0.5,
    local_c1=1.2,
    local_c2=1.2,
    local_iter=20,
    local_horizon=40,
    cluster_factor=4.0
):
    sim = Visualization(
        paths_dict,
        width=width,
        height=height,
        initial_speed=initial_speed,
        acceleration=acceleration,
        min_speed=min_speed,
        max_speed=max_speed,
        spawn_interval=spawn_interval,
        time_scale=time_scale,
        fps=fps,
        min_sep=min_sep,
        sep_weight=sep_weight,
        fuel_weight=fuel_weight,
        local_particles=local_particles,
        local_w=local_w,
        local_c1=local_c1,
        local_c2=local_c2,
        local_iter=local_iter,
        local_horizon=local_horizon,
        cluster_factor=cluster_factor
    )
    sim.run()


if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

import pygame
import random
import numpy as np

# approximate meters per degree latitude/longitude around Amsterdam
M_PER_DEG = 111320  

def compute_distance_m(path, idx):
    """Remaining distance along path (in meters) from segment idx."""
    total = 0.0
    pts = [pygame.math.Vector2(lon, lat) for lat, lon in path]
    for i in range(idx, len(pts)-1):
        a, b = pts[i], pts[i+1]
        mean_lat = np.deg2rad((a.y + b.y) / 2)
        dlat = (b.y - a.y) * M_PER_DEG
        dlon = (b.x - a.x) * M_PER_DEG * np.cos(mean_lat)
        total += np.hypot(dlat, dlon)
    return total

class Aircraft:
    next_id = 1

    def __init__(self, path_id, path, initial_speed, acceleration):
        self.id = Aircraft.next_id
        Aircraft.next_id += 1
        self.path_id = path_id
        # store path as Vector2(lon, lat) in degrees
        self.path = [pygame.math.Vector2(lon, lat) for lat, lon in path]
        self.position = self.path[0].copy()

        self.speed = initial_speed      # m/s
        self.target_speed = initial_speed
        self.acceleration = acceleration  # m/s²
        self.segment = 0

    def set_target_speed(self, new_speed):
        self.target_speed = new_speed

    def update(self, dt):
        # adjust speed toward target speed
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed,
                             self.speed + self.acceleration * dt)
        elif self.speed > self.target_speed:
            self.speed = max(self.target_speed,
                             self.speed - self.acceleration * dt)

        # move along the path
        if self.segment >= len(self.path) - 1:
            return
        target = self.path[self.segment + 1]
        direction = target - self.position
        dist_deg = direction.length()           # degrees
        if dist_deg == 0:
            self.segment += 1
            return
        # travel in degrees = (m/s * dt) / (m/deg)
        travel_deg = (self.speed * dt) / M_PER_DEG
        if travel_deg >= dist_deg:
            self.position = target.copy()
            self.segment += 1
        else:
            self.position += direction.normalize() * travel_deg

    def is_finished(self):
        return self.segment >= len(self.path) - 1

    def draw(self, surface, map_func, font):
        # draw aircraft as a small red dot with its ID and current speed
        lon, lat = self.position.x, self.position.y
        x, y = map_func(lat, lon)
        pygame.draw.circle(surface, (255, 0, 0), (int(x), int(y)), 2)
        label = font.render(f"{self.id} | {self.speed:.0f} m/s", True, (0, 0, 0))
        surface.blit(label, (x + 6, y - 6))


class SpeedPSO:
    def __init__(self, aircraft_list, min_speed, max_speed, dt, min_sep,
                 sep_weight, fuel_weight,
                 n_particles, w, c1, c2, max_iter):
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

        # particles: positions = candidate speed vectors
        self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _simulate_cost(self, speeds):
        # simulate one step ahead, compute cost
        pos_next = []
        for ac, s in zip(self.acs, speeds):
            idx = ac.segment
            if idx >= len(ac.path) - 1:
                pos_next.append(ac.position)
                continue
            a = ac.position
            b = ac.path[idx + 1]
            dir_vec = b - a
            dist_deg = dir_vec.length()
            travel_deg = (s * self.dt) / M_PER_DEG
            travel_deg = min(travel_deg, dist_deg)
            if dist_deg > 0:
                new_pos = a + dir_vec.normalize() * travel_deg
            else:
                new_pos = a
            pos_next.append(new_pos)

        # separation penalty
        sep_pen = 0.0
        for i in range(self.m):
            for j in range(i + 1, self.m):
                d_deg = (pos_next[i] - pos_next[j]).length()
                d_m = d_deg * M_PER_DEG
                if d_m < self.min_sep:
                    sep_pen += (self.min_sep - d_m) ** 2

        # fuel burn ~ sum of squares of speeds
        fuel_pen = np.sum(speeds**2)

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

            r1 = np.random.rand(nP, self.m)
            r2 = np.random.rand(nP, self.m)
            self.V = (self.w * self.V
                      + self.c1 * r1 * (self.pbest - self.X)
                      + self.c2 * r2 * (self.gbest - self.X))
            self.X = np.clip(self.X + self.V, self.E, self.L)

        return self.gbest.copy()


class ATCAgent:
    def __init__(self, min_speed, max_speed, accel, min_sep,
                 sep_weight, fuel_weight,
                 pso_particles, pso_w, pso_c1, pso_c2, pso_max_iter):
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.accel = accel
        self.min_sep = min_sep
        self.sep_weight = sep_weight
        self.fuel_weight = fuel_weight
        self.pso_particles = pso_particles
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_max_iter = pso_max_iter

    def communicate(self, aircraft_list, dt):
        m = len(aircraft_list)
        if m < 2:
            for ac in aircraft_list:
                ac.set_target_speed(self.max_speed)
            return

        pso = SpeedPSO(
            aircraft_list,
            self.min_speed,
            self.max_speed,
            dt,
            self.min_sep,
            self.sep_weight,
            self.fuel_weight,
            self.pso_particles,
            self.pso_w,
            self.pso_c1,
            self.pso_c2,
            self.pso_max_iter
        )
        best_speeds = pso.optimize()

        for ac, s in zip(aircraft_list, best_speeds):
            ac.set_target_speed(s)


class Visualization:
    def __init__(
        self,
        paths_dict,
        width=800,
        height=600,
        initial_speed=100.0,    # m/s
        acceleration=1.0,       # m/s²
        min_speed=50.0,         # PSO lower bound (m/s)
        max_speed=200.0,        # PSO upper bound (m/s)
        spawn_interval=2.0,     # seconds
        time_scale=1.0,         # sim multiplier
        fps=60,
        min_sep=5000.0,         # meters
        sep_weight=1000.0,
        fuel_weight=1.0,
        pso_particles=20,
        pso_w=0.5,
        pso_c1=1.5,
        pso_c2=1.5,
        pso_max_iter=30
    ):
        self.paths = paths_dict
        self.width = width
        self.height = height
        self.initial_speed = initial_speed
        self.acceleration = acceleration
        self.spawn_interval = spawn_interval
        self.time_scale = time_scale
        self.fps = fps
        self.min_sep = min_sep
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.sep_weight = sep_weight
        self.fuel_weight = fuel_weight
        self.pso_particles = pso_particles
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_max_iter = pso_max_iter

        self.atc = ATCAgent(
            min_speed,
            max_speed,
            acceleration,
            min_sep,
            sep_weight,
            fuel_weight,
            pso_particles,
            pso_w,
            pso_c1,
            pso_c2,
            pso_max_iter
        )

        # compute coordinate bounds
        all_lats = [pt[0] for path in paths_dict.values() for pt in path]
        all_lons = [pt[1] for path in paths_dict.values() for pt in path]
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)

        self.px_per_deg = self.width / (self.lon_max - self.lon_min)
        self.sep_radius_px = int((min_sep / M_PER_DEG) * self.px_per_deg)

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Aircraft Arrival Simulation")
        self.font = pygame.font.SysFont(None, 20)
        self.violation_count = 0

    def latlon_to_screen(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        y = self.height - (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.height
        return x, y

    def draw_paths(self, active_ids):
        for pid in active_ids:
            path = self.paths[pid]
            if len(path) < 2:
                continue
            pts = [self.latlon_to_screen(lat, lon) for lat, lon in path]
            pygame.draw.lines(self.screen, (200, 200, 200), False, pts, 2)

    def spawn_aircraft(self):
        pid, path = random.choice(list(self.paths.items()))
        ac = Aircraft(pid, path, self.initial_speed, self.acceleration)
        self.aircraft_list.append(ac)

    def run(self):
        clock = pygame.time.Clock()
        self.aircraft_list = []
        spawn_timer = 0.0
        current_sim_time = 0.0

        running = True
        while running:
            raw_dt = clock.tick(self.fps) / 1000.0
            dt = raw_dt * self.time_scale
            current_sim_time += dt

            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            # spawn new aircraft
            spawn_timer += dt
            if spawn_timer >= self.spawn_interval:
                self.spawn_aircraft()
                spawn_timer = 0.0

            # ATC sets target speeds via PSO
            self.atc.communicate(self.aircraft_list, dt)

            # update and remove finished aircraft
            for ac in self.aircraft_list[:]:
                ac.update(dt)
                if ac.is_finished():
                    self.aircraft_list.remove(ac)

            # draw everything
            self.screen.fill((255, 255, 255))
            active = {ac.path_id for ac in self.aircraft_list}
            self.draw_paths(active)

            for ac in self.aircraft_list:
                ac.draw(self.screen, self.latlon_to_screen, self.font)
                x, y = self.latlon_to_screen(ac.position.y, ac.position.x)
                pygame.draw.circle(self.screen, (0, 0, 255), (int(x), int(y)), self.sep_radius_px, 1)

            pygame.display.flip()

        pygame.quit()

def runme(
    paths_dict,
    width=1024,
    height=768,
    initial_speed=100.0,
    acceleration=1.0,
    min_speed=50.0,
    max_speed=200.0,
    spawn_interval=100.0,
    time_scale=100.0,
    fps=30,
    min_sep=2500.0,
    sep_weight=500.0,
    fuel_weight=0,
    pso_particles=40,
    pso_w=0.6,
    pso_c1=1.2,
    pso_c2=1.8,
    pso_max_iter=50
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
        pso_particles=pso_particles,
        pso_w=pso_w,
        pso_c1=pso_c1,
        pso_c2=pso_c2,
        pso_max_iter=pso_max_iter
    )
    sim.run()

if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

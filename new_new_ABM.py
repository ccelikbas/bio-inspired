import pygame
import random
import numpy as np

# approximate meters per degree latitude/longitude around Amsterdam
M_PER_DEG = 111320  

def compute_distance_m(path, idx):
    """Remaining distance along path (in meters) from segment idx."""
    total = 0.0
    # path: list of (lat, lon) tuples
    for i in range(idx, len(path)-1):
        lat1, lon1 = path[i]
        lat2, lon2 = path[i+1]
        mean_lat = np.deg2rad((lat1 + lat2) / 2)
        dlat = (lat2 - lat1) * M_PER_DEG
        dlon = (lon2 - lon1) * M_PER_DEG * np.cos(mean_lat)
        total += np.hypot(dlat, dlon)
    return total

class Aircraft:
    next_id = 1

    def __init__(self, path_id, path, initial_speed, acceleration):
        self.id = Aircraft.next_id
        Aircraft.next_id += 1
        self.path_id = path_id
        # store raw path as list of (lat, lon)
        self.raw_path = path.copy()
        # convert for movement: Vector2(lon, lat)
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
        lon, lat = self.position.x, self.position.y
        x, y = map_func(lat, lon)
        pygame.draw.circle(surface, (255, 0, 0), (int(x), int(y)), 2)
        label = font.render(f"{self.id} | {self.speed:.0f} m/s", True, (0, 0, 0))
        surface.blit(label, (x + 6, y - 6))

class LandingTimePSO:
    def __init__(self, E, L, T, g, h, SS,
                 n_particles=30, w=0.5, c1=1.5, c2=1.5, max_iter=50):
        self.E, self.L, self.T = E.copy(), L.copy(), T.copy()
        self.g, self.h, self.SS = g.copy(), h.copy(), SS.copy()
        self.n = len(E)
        self.S = n_particles
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_iter = max_iter

        self.X = np.random.uniform(self.E, self.L, (self.S, self.n))
        self.V = np.zeros((self.S, self.n))
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(self.S, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _decode_and_repair(self, Xk):
        x = np.rint(Xk).astype(int)
        x = np.clip(x, self.E.astype(int), self.L.astype(int))
        order = np.argsort(x)
        for i in range(1, self.n):
            prev, curr = order[i-1], order[i]
            if x[curr] < x[prev] + self.SS[prev,curr]:
                x[curr] = x[prev] + self.SS[prev,curr]
        return x

    def _cost(self, x):
        early = np.maximum(0, self.T - x)
        late  = np.maximum(0, x - self.T)
        return np.dot(self.g, early) + np.dot(self.h, late)

    def optimize(self):
        for _ in range(self.max_iter):
            for k in range(self.S):
                xk = self._decode_and_repair(self.X[k])
                cost = self._cost(xk)
                if cost < self.pbest_cost[k]:
                    self.pbest_cost[k] = cost
                    self.pbest[k] = self.X[k].copy()
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest = self.X[k].copy()
            r1 = np.random.rand(self.S, self.n)
            r2 = np.random.rand(self.S, self.n)
            self.V = (
                self.w * self.V
                + self.c1 * r1 * (self.pbest - self.X)
                + self.c2 * r2 * (self.gbest - self.X)
            )
            self.X = np.clip(self.X + self.V, self.E, self.L)
        return self._decode_and_repair(self.gbest)

class ATCAgent:
    def __init__(self,
                 cruise_speed, v_min, v_max,
                 horizon, min_sep,
                 g_weight=1.0, h_weight=1.0,
                 pso_particles=30, w=0.5, c1=1.5, c2=1.5, pso_iter=50):
        self.cruise = cruise_speed
        self.vmin, self.vmax = v_min, v_max
        self.horizon = horizon
        self.min_sep = min_sep
        self.gw, self.hw = g_weight, h_weight
        self.particles = pso_particles
        self.w, self.c1, self.c2 = w, c1, c2
        self.pso_iter = pso_iter

    def communicate(self, aircraft_list, current_time):
        m = len(aircraft_list)
        if m < 2:
            for ac in aircraft_list:
                ac.set_target_speed(self.cruise)
            return

        # compute remaining distances using raw_path
        D = np.array([compute_distance_m(ac.raw_path, ac.segment)
                      for ac in aircraft_list])
        E = np.full(m, int(current_time))
        L = E + int(self.horizon)
        T = E + np.ceil(D / self.cruise).astype(int)
        g = np.full(m, self.gw)
        h = np.full(m, self.hw)
        sep_time = int(np.ceil(self.min_sep / self.vmax))
        SS = np.full((m,m), sep_time, int)
        np.fill_diagonal(SS, 0)

        landing_times = LandingTimePSO(
            E, L, T, g, h, SS,
            n_particles=self.particles,
            w=self.w, c1=self.c1, c2=self.c2,
            max_iter=self.pso_iter
        ).optimize()

        for ac, t_land, dist in zip(aircraft_list, landing_times, D):
            dt_remain = max(1e-3, t_land - current_time)
            v_cmd = dist / dt_remain
            ac.set_target_speed(np.clip(v_cmd, self.vmin, self.vmax))

class Visualization:
    def __init__(
        self,
        paths_dict,
        width=800, height=600,
        initial_speed=100.0, acceleration=1.0,
        cruise_speed=100.0, v_min=50.0, v_max=200.0,
        horizon=3600.0, spawn_interval=2.0,
        time_scale=1.0, fps=60,
        min_sep=5000.0, g_weight=1.0, h_weight=1.0,
        pso_particles=30, pso_w=0.5,
        pso_c1=1.5, pso_c2=1.5, pso_iter=50
    ):
        self.paths = paths_dict
        self.width, self.height = width, height
        self.initial_speed = initial_speed
        self.acceleration = acceleration
        self.spawn_interval = spawn_interval
        self.time_scale = time_scale
        self.fps = fps
        self.min_sep = min_sep
        self.cruise = cruise_speed
        self.vmin, self.vmax = v_min, v_max
        self.horizon = horizon

        self.atc = ATCAgent(
            cruise_speed, v_min, v_max,
            horizon, min_sep,
            g_weight, h_weight,
            pso_particles, pso_w, pso_c1, pso_c2, pso_iter
        )

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
            if len(path) < 2: continue
            pts = [self.latlon_to_screen(lat, lon) for lat, lon in path]
            pygame.draw.lines(self.screen, (200,200,200), False, pts, 2)

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

            spawn_timer += dt
            if spawn_timer >= self.spawn_interval:
                self.spawn_aircraft()
                spawn_timer = 0.0

            self.atc.communicate(self.aircraft_list, current_sim_time)

            for ac in self.aircraft_list[:]:
                ac.update(dt)
                if ac.is_finished():
                    self.aircraft_list.remove(ac)

            self.screen.fill((255,255,255))
            active = {ac.path_id for ac in self.aircraft_list}
            self.draw_paths(active)
            for ac in self.aircraft_list:
                ac.draw(self.screen, self.latlon_to_screen, self.font)
                x,y = self.latlon_to_screen(ac.position.y, ac.position.x)
                pygame.draw.circle(self.screen, (0,0,255), (int(x),int(y)), self.sep_radius_px, 1)

            pygame.display.flip()

        pygame.quit()


def runme(
    paths_dict,
    width=1024, height=768,
    initial_speed=100.0, acceleration=1.0,
    cruise_speed=100.0, v_min=50.0, v_max=200.0,
    horizon=3600.0, spawn_interval=100.0, time_scale=100.0, fps=30,
    min_sep=5000.0, g_weight=1.0, h_weight=1.0,
    pso_particles=30, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, pso_iter=50
):
    sim = Visualization(
        paths_dict,
        width=width, height=height,
        initial_speed=initial_speed, acceleration=acceleration,
        cruise_speed=cruise_speed, v_min=v_min, v_max=v_max,
        horizon=horizon, spawn_interval=spawn_interval,
        time_scale=time_scale, fps=fps,
        min_sep=min_sep, g_weight=g_weight, h_weight=h_weight,
        pso_particles=pso_particles, pso_w=pso_w,
        pso_c1=pso_c1, pso_c2=pso_c2, pso_iter=pso_iter
    )
    sim.run()

if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

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
        a, b = pts[i], pts[i+1]
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
        self.raw_path = list(path)   # list of (lat, lon)
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

class SequencePSO:
    def __init__(self, acs, now, min_speed, max_speed,
                 sep_time,
                 w_delay, w_sep, w_fuel,
                 n_particles, w, c1, c2, max_iter):
        self.acs = acs
        self.now = now
        self.N = len(acs)
        # remaining distance along each path
        self.D = np.array([compute_distance_m(ac.raw_path, ac.segment) for ac in acs])
        self.Tmin = now + self.D / max_speed
        self.Tmax = now + self.D / min_speed
        self.sep_time = sep_time
        self.w_delay, self.w_sep, self.w_fuel = w_delay, w_sep, w_fuel
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_iter = max_iter
        self.X = np.random.uniform(self.Tmin, self.Tmax, (n_particles, self.N))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _cost(self, T):
        delay_pen = np.sum(T - self.Tmin)
        sep_pen = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                dt = abs(T[i] - T[j])
                if dt < self.sep_time:
                    sep_pen += (self.sep_time - dt)**2
        v = self.D / (T - self.now)
        fuel_pen = np.sum((v - np.mean(v))**2)
        return self.w_delay*delay_pen + self.w_sep*sep_pen + self.w_fuel*fuel_pen

    def optimize(self):
        P = self.X.shape[0]
        for _ in range(self.max_iter):
            for k in range(P):
                c = self._cost(self.X[k])
                if c < self.pbest_cost[k]:
                    self.pbest_cost[k] = c
                    self.pbest[k] = self.X[k].copy()
                if c < self.gbest_cost:
                    self.gbest_cost = c
                    self.gbest = self.X[k].copy()
            r1, r2 = np.random.rand(P, self.N), np.random.rand(P, self.N)
            self.V = (self.w*self.V
                      + self.c1*r1*(self.pbest - self.X)
                      + self.c2*r2*(self.gbest - self.X))
            self.X = np.clip(self.X + self.V, self.Tmin, self.Tmax)
        return self.gbest.copy()

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
                for j in range(i+1, self.m):
                    d = sims[i].distance_to(sims[j])
                    if d < self.min_sep:
                        sep_pen += (self.min_sep - d)**2
        return self.sep_w*sep_pen + self.fuel_w*fuel_pen

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
            self.V = (self.w*self.V
                      + self.c1*r1*(self.pbest - self.X)
                      + self.c2*r2*(self.gbest - self.X))
            self.X = np.clip(self.X + self.V, self.E, self.L)
        return self.gbest.copy()

class ATCAgent:
    def __init__(self,
                 min_speed, max_speed, accel,
                 min_sep, sep_weight, fuel_weight,
                 local_particles, local_w, local_c1, local_c2, local_iter, local_horizon,
                 cluster_factor,
                 ils_lat, ils_lon, ils_radius_m,
                 seq_particles, seq_w, seq_c1, seq_c2, seq_iter,
                 w_delay, w_sep, w_fuel,
                 sep_time):
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
        # ILS params
        self.ils_lat = ils_lat
        self.ils_lon = ils_lon
        self.ils_radius_m = ils_radius_m
        self.ils_cluster = []
        # sequence PSO params
        self.seq_particles = seq_particles
        self.seq_w = seq_w
        self.seq_c1 = seq_c1
        self.seq_c2 = seq_c2
        self.seq_iter = seq_iter
        self.w_delay, self.w_sep, self.w_fuel = w_delay, w_sep, w_fuel
        self.sep_time = sep_time

    def _find_clusters(self, ac_list):
        clusters, unvisited = [], set(ac_list)
        while unvisited:
            seed = unvisited.pop()
            stack, cluster = [seed], {seed}
            while stack:
                a = stack.pop()
                for b in list(unvisited):
                    if a.distance_to(b) < self.cluster_factor*self.min_sep:
                        unvisited.remove(b)
                        cluster.add(b)
                        stack.append(b)
            clusters.append(list(cluster))
        return clusters

    def communicate(self, aircraft_list, dt, sim_time):
        # ILS sequence on all in-zone as one cluster
        in_zone = []
        for ac in aircraft_list:
            lat1, lon1 = ac.position.y, ac.position.x
            mean_lat = np.deg2rad((lat1 + self.ils_lat)/2)
            dlat = (self.ils_lat - lat1)*M_PER_DEG
            dlon = (self.ils_lon - lon1)*M_PER_DEG*np.cos(mean_lat)
            if np.hypot(dlat, dlon) <= self.ils_radius_m:
                in_zone.append(ac)
        self.ils_cluster = in_zone
        if len(in_zone) > 1:
            seq = SequencePSO(in_zone, sim_time,
                              self.min_speed, self.max_speed,
                              self.sep_time,
                              self.w_delay, self.w_sep, self.w_fuel,
                              self.seq_particles,
                              self.seq_w, self.seq_c1, self.seq_c2,
                              self.seq_iter)
            T_best = seq.optimize()
            for ac, T_i, D_i in zip(in_zone, T_best, seq.D):
                v_i = D_i / (T_i - sim_time)
                ac.set_target_speed(v_i)
        # local PSO for others
        out_zone = [ac for ac in aircraft_list if ac not in in_zone]
        clusters = self._find_clusters(out_zone)
        self.last_clusters = clusters
        for cluster in clusters:
            if len(cluster) < 2: continue
            local = LocalPSO(cluster,
                             self.min_speed, self.max_speed,
                             dt, self.min_sep,
                             self.sep_weight, self.fuel_weight,
                             self.local_particles,
                             self.local_w, self.local_c1, self.local_c2,
                             self.local_iter,
                             horizon_steps=self.local_horizon)
            best = local.optimize()
            for ac, s in zip(cluster, best): ac.set_target_speed(s)

class Visualization:
    PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
               (0,255,255),(128,0,128),(255,165,0),(0,128,128),(128,128,0)]
    ILS_LAT = 52.33080945036779
    ILS_LON = 5.32394574767848

    def __init__(self, paths_dict,
                 width=800, height=600,
                 initial_speed=100.0, acceleration=1.0,
                 min_speed=50.0, max_speed=200.0,
                 spawn_interval=2.0, time_scale=1.0, fps=60,
                 min_sep=5000.0, sep_weight=1000.0, fuel_weight=1.0,
                 local_particles=10, local_w=0.5, local_c1=1.2, local_c2=1.2,
                 local_iter=20, local_horizon=10, cluster_factor=2.0,
                 ils_radius_m=10000.0,
                 seq_particles=20, seq_w=0.5, seq_c1=1.5, seq_c2=1.5, seq_iter=30,
                 w_delay=1.0, w_sep=100.0, w_fuel=1.0, sep_time=30.0):
        self.paths = paths_dict
        self.width, self.height = width, height
        self.initial_speed, self.acceleration = initial_speed, acceleration
        self.spawn_interval, self.time_scale, self.fps = spawn_interval, time_scale, fps
        self.min_speed, self.max_speed = min_speed, max_speed
        self.min_sep, self.sep_weight, self.fuel_weight = min_sep, sep_weight, fuel_weight
        self.atc = ATCAgent(min_speed, max_speed, acceleration,
                            min_sep, sep_weight, fuel_weight,
                            local_particles, local_w, local_c1, local_c2, local_iter, local_horizon,
                            cluster_factor,
                            self.ILS_LAT, self.ILS_LON, ils_radius_m,
                            seq_particles, seq_w, seq_c1, seq_c2, seq_iter,
                            w_delay, w_sep, w_fuel,
                            sep_time)
        all_lats = [pt[0] for path in paths_dict.values() for pt in path]
        all_lons = [pt[1] for path in paths_dict.values() for pt in path]
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)
        self.px_per_deg = self.width/(self.lon_max-self.lon_min)
        self.sep_radius_px = int((min_sep/M_PER_DEG)*self.px_per_deg)
        self.ils_px = self._geo_to_px(self.ILS_LAT, self.ILS_LON)
        self.ils_radius_px = int((ils_radius_m/M_PER_DEG)*self.px_per_deg)
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("ILS‐Sequence ABM")
        self.font = pygame.font.SysFont(None, 20)

    def _geo_to_px(self, lat, lon):
        x = (lon-self.lon_min)/(self.lon_max-self.lon_min)*self.width
        y = self.height - (lat-self.lat_min)/(self.lat_max-self.lat_min)*self.height
        return int(x), int(y)

    def draw_paths(self):
        active = {ac.path_id for ac in self.aircraft_list}
        for pid in active:
            pts = [self._geo_to_px(lat, lon) for lat, lon in self.paths[pid]]
            if len(pts)>1:
                pygame.draw.lines(self.screen, (200,200,200), False, pts, 2)

    def spawn_aircraft(self):
        pid, path = random.choice(list(self.paths.items()))
        ac = Aircraft(pid, path,
                      self.initial_speed, self.acceleration,
                      self.min_speed, self.max_speed)
        self.aircraft_list.append(ac)

    def run(self):
        clock = pygame.time.Clock()
        self.aircraft_list = []
        spawn_timer = 0.0
        sim_time = 0.0
        running = True
        while running:
            raw_dt = clock.tick(self.fps)/1000.0
            dt = raw_dt*self.time_scale
            sim_time += dt
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
            spawn_timer += dt
            if spawn_timer>=self.spawn_interval:
                self.spawn_aircraft()
                spawn_timer=0.0
            
            self.atc.communicate(self.aircraft_list, dt, sim_time)
            
            for ac in self.aircraft_list[:]:
                ac.update(dt)
                if ac.is_finished():
                    self.aircraft_list.remove(ac)
            self.screen.fill((255,255,255))
            self.draw_paths()
            
            # ILS zone
            pygame.draw.circle(self.screen, (0,255,0), self.ils_px,
                               self.ils_radius_px, 2)
            lbl = self.font.render("Convergence Point", True, (0,128,0))
            self.screen.blit(lbl, (self.ils_px[0]+5, self.ils_px[1]-15))
            
            # local clusters map
            cluster_map = {}
            for idx, c in enumerate(self.atc.last_clusters):
                if len(c)<2: continue
                for ac in c: cluster_map[ac.id] = idx
            for ac in self.aircraft_list:
                x, y = self._geo_to_px(ac.position.y, ac.position.x)
                violated = any(
                    ac.distance_to(o)<self.min_sep
                    for o in self.aircraft_list if o is not ac
                )
                # color priority: violation, ILS cluster, local cluster, gray
                if violated:
                    dot_color = (255,0,0)
                elif ac in self.atc.ils_cluster:
                    dot_color = (0,200,0)
                else:
                    idx = cluster_map.get(ac.id)
                    dot_color = ((200,200,200)
                                 if idx is None
                                 else Visualization.PALETTE[idx%len(Visualization.PALETTE)])
                pygame.draw.circle(self.screen, dot_color, (x,y), 4)
                txt = self.font.render(f"{ac.id}|{ac.speed:.0f}", True, (0,0,0))
                self.screen.blit(txt, (x+6, y-12))
                pygame.draw.circle(self.screen, dot_color, (x,y),
                                   self.sep_radius_px, 1)
            pygame.display.flip()
            
        pygame.quit()

def runme(
    paths_dict,
    width=1024, height=768,
    initial_speed=150.0, acceleration=1.0,
    min_speed=50.0, max_speed=150.0,
    spawn_interval=150.0, time_scale=100.0, fps=30,
    min_sep=2500.0, sep_weight=500.0, fuel_weight=0,
    local_particles=20, local_w=0.5, local_c1=1.2, local_c2=1.2,
    local_iter=20, local_horizon=10, cluster_factor=4.0,
    ils_radius_m=80000.0,
    seq_particles=30, seq_w=0.5, seq_c1=1.5, seq_c2=1.5, seq_iter=100,
    w_delay=1.0, w_sep=100.0, w_fuel=0, sep_time=30.0
):
    sim = Visualization(
        paths_dict,
        width, height,
        initial_speed, acceleration,
        min_speed, max_speed,
        spawn_interval, time_scale, fps,
        min_sep, sep_weight, fuel_weight,
        local_particles, local_w, local_c1, local_c2, local_iter, local_horizon,
        cluster_factor,
        ils_radius_m,
        seq_particles, seq_w, seq_c1, seq_c2, seq_iter,
        w_delay, w_sep, w_fuel,
        sep_time
    )
    sim.run()

if __name__ == '__main__':
    from paths import paths_dict
    runme(paths_dict)

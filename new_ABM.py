import pygame
import random
import numpy as np
import copy
import bisect
from paths import paths_dict
import matplotlib.pyplot as plt


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
    adj = {ac: [] for ac in acs}
    for i, a in enumerate(acs):
        for b in acs[i+1:]:
            if a.distance_to(b) < radius:
                adj[a].append(b)
                adj[b].append(a)
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
        self.id = Aircraft.next_id; Aircraft.next_id += 1
        self.path_id = path_id
        self.path = [pygame.math.Vector2(lon, lat) for lat, lon in path]
        self.position = self.path[0].copy()
        self.speed = initial_speed; self.target_speed = initial_speed
        self.acceleration = acceleration
        self.min_speed = min_speed; self.max_speed = max_speed
        self.segment = 0

    def set_target_speed(self, new_speed):
        self.target_speed = max(self.min_speed, min(self.max_speed, new_speed))

    def update(self, dt):
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + self.acceleration * dt)
        else:
            self.speed = max(self.target_speed, self.speed - self.acceleration * dt)
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        if self.segment >= len(self.path) - 1: return
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
                 horizon_steps=3000, step_skip=15):
        """
        Particle Swarm Optimizer over aircraft speeds.
        """
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

        # Precompute each aircraft’s cumulative path lengths & waypoints
        self.path_cumlens = []
        self.path_pts     = []
        for ac in self.acs:
            pts = ac.path
            dist = [0.0]
            for A, B in zip(pts[:-1], pts[1:]):
                mean_lat = np.deg2rad((A.y + B.y)/2)
                dlat = (B.y - A.y)*M_PER_DEG
                dlon = (B.x - A.x)*M_PER_DEG*np.cos(mean_lat)
                dist.append(np.hypot(dlat, dlon))
            self.path_cumlens.append(np.cumsum(dist))
            self.path_pts.append(pts)

        # Initialize swarm positions & velocities
        self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _simulate_cost(self, speeds):
        """
        Simulate trajectories under `speeds`.  
        Enforce separation only after the first interval, so t=0 is ignored.
        """
        init_speeds = np.array([ac.speed for ac in self.acs])

        # start checking from the first non-zero sample
        samples = np.arange(self.step_skip, self.horizon + 1, self.step_skip)

        for t in samples:
            τ = t * self.dt
            D = speeds * τ
            positions = []
            for i in range(self.m):
                cum = self.path_cumlens[i]
                pts = self.path_pts[i]
                d = min(D[i], cum[-1])
                idx = bisect.bisect_right(cum, d) - 1
                if idx >= len(pts) - 1:
                    positions.append(pts[-1])
                else:
                    A, B = pts[idx], pts[idx + 1]
                    seg_len = cum[idx + 1] - cum[idx]
                    frac = (d - cum[idx]) / seg_len if seg_len > 0 else 0.0
                    positions.append(A + (B - A) * frac)

            # HARD CONSTRAINT: only now enforce min_sep
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    d_ij = positions[i].distance_to(positions[j])
                    if positions[i].distance_to(positions[j]) < self.min_sep:
                        return 1e9 + (self.min_sep - d_ij)**2

        # if no breaches, cost = velocity‐deviation penalty
        vel_pen = np.sum((speeds - init_speeds) ** 2)
        return self.fuel_w * vel_pen
    


    def optimize(self):
        """
        Run PSO and return the best speed profile found.
        Records self.history of gbest_cost per iteration.
        """
        nP = self.X.shape[0]

        # --- Initialization pass to seed pbest & gbest ---
        for k in range(nP):
            cost = self._simulate_cost(self.X[k])
            self.pbest_cost[k] = cost
            self.pbest[k] = self.X[k].copy()
            if self.gbest is None or cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest = self.X[k].copy()

        # Prepare history list
        self.history = [self.gbest_cost]

        # --- Main PSO loop ---
        for _ in range(self.max_iter):
            # Update personal & global bests
            for k in range(nP):
                cost = self._simulate_cost(self.X[k])
                if cost < self.pbest_cost[k]:
                    self.pbest_cost[k] = cost
                    self.pbest[k] = self.X[k].copy()
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest = self.X[k].copy()

            # Record current best cost
            self.history.append(self.gbest_cost)

            # Velocity & position updates
            r1 = np.random.rand(nP, self.m)
            r2 = np.random.rand(nP, self.m)
            self.V = (
                self.w * self.V
                + self.c1 * r1 * (self.pbest - self.X)
                + self.c2 * r2 * (self.gbest - self.X)
            )
            self.X = np.clip(self.X + self.V, self.E, self.L)

        return self.gbest.copy()


class ATCAgent:
    def __init__(self,
                 min_speed, max_speed, accel,
                 min_sep, sep_weight, fuel_weight,
                 particles, w, c1, c2, iters,
                 global_horizon, global_step_skip,
                 local_comm_radius, local_horizon, local_sep_weight):
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
        # select only unfinished aircraft
        acs = [ac for ac in aircraft_list if not ac.is_finished()]
        if len(acs) < 2: return

        # GLOBAL PSO over all ACs
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

        # # Plot PSO costs over itterations to check convergence
        # plt.figure(figsize=(6,4))
        # plt.plot(global_pso.history, marker='o')
        # plt.xlabel("PSO iteration")
        # plt.ylabel("Global best cost")
        # plt.title("PSO convergence (single run)")
        # plt.grid(True)
        # plt.draw()
        # plt.pause(0.001)

        for ac, s in zip(acs, global_speeds):
            ac.set_target_speed(s)
        
        # LOCAL PSO on any clusters within local_comm_radius
        clusters = _find_clusters(acs, self.local_comm_radius)
        # print("Local clusters:", [[ac.id for ac in c] for c in clusters])
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
            
            # Plot PSO costs over itterations to check convergence
            plt.figure(figsize=(6,4))
            plt.plot(local_pso.history, marker='o')
            plt.xlabel("PSO iteration")
            plt.ylabel("Local best cost")
            plt.title("PSO convergence (single run)")
            plt.grid(True)
            plt.draw()
            plt.pause(0.001)
            
            for ac, s in zip(cluster, local_speeds):
                ac.set_target_speed(s)


class Visualization:
    PALETTE=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
             (128,0,128),(255,165,0),(0,128,128),(128,128,0)]
    def __init__(self,paths_dict,width,height,min_sep):
        self.paths=paths_dict; self.width,self.height=width,height; self.min_sep=min_sep
        self.lat_min=min(lat for p in paths_dict.values() for lat,lon in p)
        self.lat_max=max(lat for p in paths_dict.values() for lat,lon in p)
        self.lon_min=min(lon for p in paths_dict.values() for lat,lon in p)
        self.lon_max=max(lon for p in paths_dict.values() for lat,lon in p)
        self.sep_radius_px=int(min_sep/M_PER_DEG*(width/(self.lon_max-self.lon_min)))
        pygame.init(); self.screen=pygame.display.set_mode((width,height))
        self.font=pygame.font.SysFont(None,20); self.aircraft_list=[]

    def _geo_to_px(self,lat,lon):
        x=(lon-self.lon_min)/(self.lon_max-self.lon_min)*self.width
        y=self.height-(lat-self.lat_min)/(self.lat_max-self.lat_min)*self.height
        return int(x),int(y)

    def draw(self,sim_time):
        self.screen.fill((255,255,255))
        active={ac.path_id for ac in self.aircraft_list}
        for pid in active:
            pts=[self._geo_to_px(lat,lon) for lat,lon in self.paths[pid]]
            if len(pts)>1: pygame.draw.lines(self.screen,(200,200,200),False,pts,2)
        clusters=_find_clusters(self.aircraft_list,
                                self.sep_radius_px*M_PER_DEG/self.width*(self.lon_max-self.lon_min))
        cmap={ac.id:i for i,cl in enumerate(clusters) for ac in cl}
        conflicts={ac.id for i,ac in enumerate(self.aircraft_list)
                   for other in self.aircraft_list[i+1:]
                   if ac.distance_to(other)<self.min_sep}
        for ac in self.aircraft_list:
            x,y=self._geo_to_px(ac.position.y,ac.position.x)
            if ac.id in conflicts: color=(255,0,0)
            elif ac.id in cmap: color=Visualization.PALETTE[cmap[ac.id]%len(Visualization.PALETTE)]
            else: color=(0,0,0)
            pygame.draw.circle(self.screen,color,(x,y),4)
            lbl=self.font.render(f"{ac.id}|{ac.speed:.0f}",True,color)
            self.screen.blit(lbl,(x+10,y+6))
            pygame.draw.circle(self.screen,color,(x,y),self.sep_radius_px,1)
        timg=self.font.render(f"Time:{sim_time:.1f}s",True,(0,0,0))
        self.screen.blit(timg,(10,10)); pygame.display.flip()

def runme(paths_dict,
          sim_duration=3600.0,
          visualize=False,
          width=1024,
          height=768,
          initial_speed=100.0,
          acceleration=0.5,
          min_speed=80.0,
          max_speed=130.0,
          spawn_interval=400.0,
          pso_interval=50.0,
          time_scale=100.0,
          fps=30,
          min_sep=3000.0,
          sep_weight=100,
          fuel_weight=1,
          pso_particles=12,
          pso_w=0.5,
          pso_c1=1.2,
          pso_c2=1.2,
          pso_iters=12,
          horizon_steps=3000,
          step_skip=20,
          local_comm_radius=20000,
          local_horizon=40,
          local_sep_weight=10000.0):
    atc = ATCAgent(min_speed, max_speed, acceleration,
                   min_sep, sep_weight, fuel_weight,
                   pso_particles, pso_w, pso_c1, pso_c2, pso_iters,
                   horizon_steps, step_skip,
                   local_comm_radius, local_horizon, local_sep_weight)
    aircraft_list = []
    spawn_t = pso_t = sim_t = 0.0
    collisions = throughput = 0
    if visualize:
        viz = Visualization(paths_dict, width, height, min_sep)
        clock = pygame.time.Clock()
    while sim_t < sim_duration:
        raw = clock.tick(fps)/1000.0 if visualize else 1.0/fps
        dt = raw * time_scale
        if visualize:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sim_t = sim_duration; break
        spawn_t += dt
        if spawn_t >= spawn_interval:
            pid, path = random.choice(list(paths_dict.items()))
            aircraft_list.append(Aircraft(pid, path, initial_speed,
                                         acceleration, min_speed, max_speed))
            spawn_t -= spawn_interval
        pso_t += dt
        if pso_t >= pso_interval:
            atc.communicate(aircraft_list, dt)
            pso_t -= pso_interval
        for ac in aircraft_list[:]:
            ac.update(dt)
            if ac.is_finished():
                aircraft_list.remove(ac)
                # print(f'Aircraft landed')
                throughput += 1
        for i, ac1 in enumerate(aircraft_list):
            for ac2 in aircraft_list[i+1:]:
                if ac1.distance_to(ac2) < min_sep:
                    # print(f'Collisions detected')
                    collisions += 1
        if visualize:
            viz.aircraft_list = aircraft_list
            viz.draw(sim_t)
        sim_t += dt
    if visualize:
        pygame.quit()
    return {'collisions': collisions, 'throughput': throughput}

if __name__ == '__main__':
    import numpy as np

    # parameters for the simulation
    sim_kwargs = dict(
        paths_dict=paths_dict,
        sim_duration=20000.0,
        visualize=True,       # headless for speed
        width=1024,
        height=768,
        initial_speed=100.0,
        acceleration=1,
        min_speed=80.0,
        max_speed=130.0,
        spawn_interval=400.0,
        pso_interval=50.0,
        time_scale=100.0,
        fps=30,
        min_sep=3000.0,
        sep_weight=100,
        fuel_weight=1,
        pso_particles=12,
        pso_w=0.5,
        pso_c1=1.2,
        pso_c2=1.2,
        pso_iters=8,
        horizon_steps=3000,
        step_skip=20,
        local_comm_radius=20000,
        local_horizon=40,
        local_sep_weight=10000.0
    )

    results = []
    for i in range(1):
        res = runme(**sim_kwargs)
        print(f"Run {i+1}: collisions={res['collisions']}, throughput={res['throughput']}")
        results.append(res)

    # extract metrics as numpy arrays
    collisions = np.array([r['collisions'] for r in results])
    throughput = np.array([r['throughput'] for r in results])

    # compute statistics
    coll_mean, coll_std = collisions.mean(), collisions.std(ddof=1)
    thr_mean, thr_std   = throughput.mean(), throughput.std(ddof=1)

    print("\nSummary over 5 runs:")
    print(f"  Collisions: mean = {coll_mean:.2f}, std = {coll_std:.2f}")
    print(f"  Throughput: mean = {thr_mean:.2f}, std = {thr_std:.2f}")


# --- Evolutionary Robotics layer to tune PSO hyperparameters ---

# def random_genome():
#     return {
#         'w': random.uniform(0.1, 1.0),
#         'c1': random.uniform(0.5, 2.5),
#         'c2': random.uniform(0.5, 2.5)
#     }

# def mutate(genome, rate=0.2):
#     for k in genome:
#         if random.random() < rate:
#             genome[k] += random.uniform(-0.1, 0.1)
#             genome[k] = max(0.0, min(3.0, genome[k]))

# def fitness(genome):
#     res = runme(
#         paths_dict,
#         visualize=False,
#         pso_w=genome['w'],
#         pso_c1=genome['c1'],
#         pso_c2=genome['c2']
#     )
#     # print KPIs for this evaluation
#     print(f"    Evaluated {genome} -> collisions={res['collisions']}, throughput={res['throughput']}")
#     return res['throughput'] - 1000 * res['collisions']

# def evolve_population(pop_size=10, generations=5):
#     pop = [random_genome() for _ in range(pop_size)]
#     for gen in range(generations):
#         scored = [(fitness(g), g) for g in pop]
#         scored.sort(key=lambda x: x[0], reverse=True)
#         best_score, best_genome = scored[0]
#         print(f"Gen {gen}: best {best_genome} → score {best_score}")
#         next_pop = [copy.deepcopy(best_genome)]
#         while len(next_pop) < pop_size:
#             parent = copy.deepcopy(random.choice(scored[:pop_size//2])[1])
#             mutate(parent)
#             next_pop.append(parent)
#         pop = next_pop
#     return best_genome


# if __name__ == '__main__':
#     best = evolve_population(pop_size=8, generations=6)
#     print(f"=== Evolved hyperparameters: {best} ===")
#     # run final sim with evolved params
#     result = runme(
#         paths_dict,
#         visualize=True,
#         pso_w=best['w'],
#         pso_c1=best['c1'],
#         pso_c2=best['c2']
#     )
#     print("Final KPIs:", result)



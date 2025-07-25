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

    def update(self, dt, all_aircraft, LOCAL_SEP_RADIUS):
        # 1) Compute my heading vector to next waypoint
        if self.segment < len(self.path) - 1:
            vec     = self.path[self.segment + 1] - self.position
            heading = vec.normalize() if vec.length() > 0 else pygame.math.Vector2(0,0)
            my_next = self.path[self.segment + 1]
        else:
            heading = pygame.math.Vector2(0,0)
            my_next = None

        # --- 2) Look for any “leader” ahead on the same converging lane ---
        COS_THRESH = 0.99   # dot(heading, other_heading) ≥ this → same direction
        for other in all_aircraft:
            if other is self:
                continue

            # 2a) quick distance test
            if self.distance_to(other) > LOCAL_SEP_RADIUS:
                continue

            # 2b) compute their heading too
            if other.segment < len(other.path) - 1:
                ov = other.path[other.segment + 1] - other.position
                oheading = ov.normalize() if ov.length() > 0 else pygame.math.Vector2(0,0)
            else:
                continue

            # 2c) only if nearly parallel
            if heading.dot(oheading) < COS_THRESH:
                continue

            # 2d) projection: along > 0 means other is in front
            rel = other.position - self.position
            along = rel.dot(heading)
            if along > 0:
                # clamp so I never overtake
                self.target_speed = min(self.target_speed, other.speed - 5)

                other.target_speed = min(other.max_speed, other.target_speed + 5)

        # Intersection‑yield rule
        # only if both have a “next” waypoint
        if my_next is not None:
            for other in all_aircraft:
                if other is self or other.segment >= len(other.path) - 1:
                    continue

                their_next = other.path[other.segment + 1]
                # 3a) same next waypoint?
                if (their_next - my_next).length() > 1e-4:
                    continue

                # 3b) different heading? (not parallel)
                ov       = their_next - other.position
                oheading = ov.normalize() if ov.length() > 0 else pygame.math.Vector2(0,0)
                if heading.dot(oheading) > COS_THRESH:
                    continue

                # 3c) proximity check (3× LOCAL_SEP_RADIUS)
                if self.distance_to(other) > 5 * LOCAL_SEP_RADIUS:
                    continue

                # check who is farther from next node
                d_self  = (my_next - self.position).length() * M_PER_DEG
                d_other = (their_next - other.position).length() * M_PER_DEG

                if d_self > d_other:
                    # print(f'airfraft {self.id} is slowing down due to intersection colision with aircraft {other.id}')
                    self.target_speed = min(self.target_speed, other.speed - 15)
                
                if d_self > d_other:
                    # I’m follower → slow to other.speed
                    self.target_speed = min(self.target_speed, other.speed - 5)
                    # other’s lead → boost up
                    other.target_speed = min(other.max_speed,
                                             other.target_speed + 5)
                else:
                    # I’m lead → boost me
                    self.target_speed = min(self.max_speed,
                                            self.target_speed + 5)
                    # other slows
                    other.target_speed = min(other.target_speed, self.speed - 5)

                break

        # --- 3) accelerate/decelerate toward this (possibly clamped) target_speed ---
        if self.speed < self.target_speed:
            self.speed = min(self.target_speed, self.speed + self.acceleration * dt)
        else:
            self.speed = max(self.target_speed, self.speed - self.acceleration * dt)
        self.speed = max(self.min_speed, min(self.max_speed, self.speed))

        # --- 4) finally, move along the path as before --------------------
        if self.segment >= len(self.path) - 1:
            return
        nxt = self.path[self.segment + 1]
        vec = nxt - self.position
        dist_deg = vec.length()
        if dist_deg == 0:
            self.segment += 1
            return
        travel_deg = (self.speed * dt) / M_PER_DEG
        if travel_deg >= dist_deg:
            self.position = nxt.copy()
            self.segment += 1
        else:
            self.position += vec.normalize() * travel_deg


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
                 horizon_steps=3000, step_skip=15,
                 enforce_overtake=False):
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
        self.enforce_overtake = enforce_overtake

        # Precompute each aircraft’s cumulative path lengths & waypoints
        self.path_cumlens = []
        self.path_pts     = []
        for ac in self.acs:
            pts = ac.path
            # print(pts)
            dist = [0.0]
            for A, B in zip(pts[:-1], pts[1:]):
                mean_lat = np.deg2rad((A.y + B.y)/2)
                dlat = (B.y - A.y)*M_PER_DEG
                dlon = (B.x - A.x)*M_PER_DEG*np.cos(mean_lat)
                dist.append(np.hypot(dlat, dlon))
            
            # cum[k] = distance from waypoint[0] up through waypoint[k]
            self.path_cumlens.append(np.cumsum(dist))
            self.path_pts.append(pts)

        # # print total path length for each aircraft
        # print("Total path lengths to runway for each aircraft from start point:")
        # for ac, cum in zip(self.acs, self.path_cumlens):
        #     print(f"  AC {ac.id} (path {ac.path_id}): {cum[-1]:.1f} m")

        # Initialize swarm positions & velocities
        self.X = np.random.uniform(self.E, self.L, (n_particles, self.m))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_cost = np.full(n_particles, np.inf)
        self.gbest = None
        self.gbest_cost = np.inf

    def _simulate_cost(self, speeds):
        """
        Simulate trajectories under `speeds`, starting from each AC's current
        position. 
        """
        init_speeds = np.array([ac.speed for ac in self.acs])
        samples = np.arange(self.step_skip, self.horizon + 1, self.step_skip)

        spread_pen = 0.0

        for t in samples:
            τ = t * self.dt
            positions = []
            for i in range(self.m):
                ac  = self.acs[i]
                cum = self.path_cumlens[i]
                pts = self.path_pts[i]

                # 1. Compute how far this AC already is along its path:
                #    -> cum[ac.segment] is distance up to last passed waypoint
                #    -> add the partial distance from that waypoint to current pos
                seg_idx = ac.segment
                already = cum[seg_idx]
                A = pts[seg_idx] # last past waypoint 
                P = ac.position # current pos aicraft 
                # convert dif. lat/lon between A and P into meters
                mean_lat = np.deg2rad((A.y + P.y)/2)
                dlat = (P.y - A.y) * M_PER_DEG
                dlon = (P.x - A.x) * M_PER_DEG * np.cos(mean_lat)
                already += np.hypot(dlat, dlon)

                # 2. Simulate forward-moving distance at constant speed
                forward = speeds[i] * τ

                # Total distance along path = already + forward
                d_total = min(already + forward, cum[-1])

                # Find which segment that lands us in, and interpolate
                idx = bisect.bisect_right(cum, d_total) - 1
                if idx >= len(pts) - 1:
                    positions.append(pts[-1])
                else:
                    A, B = pts[idx], pts[idx + 1]
                    seg_len = cum[idx + 1] - cum[idx]
                    frac = (d_total - cum[idx]) / seg_len if seg_len > 0 else 0.0
                    positions.append(A + (B - A) * frac)

            # smoother optimisation attempt
            max_allowed_penalty = 1e10
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    d_ij = positions[i].distance_to(positions[j])
                    if d_ij < self.min_sep:
                        spread_pen += 1000 * (1 / (d_ij + 1e-5))**4
                        # spread_pen += 10000 * (self.min_sep - d_ij)**2
                    else:
                        spread_pen += (1 / d_ij)**2

                    # if spread_pen > max_allowed_penalty:
                    #     return spread_pen  # soft early abort


            # # Pairwise separation check
            # for i in range(self.m):
            #     for j in range(i+1, self.m):
            #         d_ij = positions[i].distance_to(positions[j])
            #         if d_ij < self.min_sep:
            #             # Hard-violation: abort with huge penalty
            #             # return 1e9 + (self.min_sep - d_ij)**2
            #             return 5000 + (self.min_sep - d_ij)**2
            #         else:
            #             spread_pen += (self.min_sep / d_ij)**2
                    
                    # if d_ij < self.min_sep:
                    #     # add a steep but continuous penalty
                    #     spread_pen += 1e5 * (self.min_sep - d_ij)**2
                    # else:
                    #     spread_pen += (self.min_sep / d_ij)**2

        # fuel‐change penalty, etc.
        vel_pen = np.sum((speeds - init_speeds)**2)

        return self.sep_w * spread_pen + self.fuel_w * vel_pen
    

    def optimize(self):
        """
        Run PSO and return the best speed profile found.
        Before starting, compute & print each AC’s remaining distance to runway.
        """
        # Compute remaining distance for each aircraft
        for i, ac in enumerate(self.acs):
            cum = self.path_cumlens[i]
            pts = self.path_pts[i]
            seg = ac.segment

            # a) distance up to last passed waypoint
            already = cum[seg]

            # b) plus the partial distance from that waypoint to current position
            A = pts[seg]
            P = ac.position
            mean_lat = np.deg2rad((A.y + P.y)/2)
            dlat = (P.y - A.y) * M_PER_DEG
            dlon = (P.x - A.x) * M_PER_DEG * np.cos(mean_lat)
            already += np.hypot(dlat, dlon)

            # c) remaining = total path length minus what’s already flown
            remaining = cum[-1] - already
            # print(f"  AC {ac.id} (on path '{ac.path_id}'): {remaining:.1f} m")

        # Seed pbest & gbest 
        nP = self.X.shape[0]
        for k in range(nP):
            cost = self._simulate_cost(self.X[k])
            self.pbest_cost[k] = cost
            self.pbest[k] = self.X[k].copy()
            if self.gbest is None or cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest = self.X[k].copy()

        self.history = [self.gbest_cost]

        # Main PSO loop 
        for _ in range(self.max_iter):
            for k in range(nP):
                cost = self._simulate_cost(self.X[k])
                if cost < self.pbest_cost[k]:
                    self.pbest_cost[k] = cost
                    self.pbest[k] = self.X[k].copy()
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest = self.X[k].copy()
            self.history.append(self.gbest_cost)
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
                 local_comm_radius, local_horizon, local_sep_weight, add_con_plot):
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

        self.global_histories = []
        self.local_histories  = []


        if add_con_plot:
            plt.ion()

            # ---- create a persistent figure/axis for GLOBAL PSO ----
            self._g_fig, self._g_ax = plt.subplots(figsize=(5,3))
            self._g_ax.set_title("Global PSO Convergence")
            self._g_ax.set_xlabel("Iteration")
            self._g_ax.set_ylabel("Best Cost")
            self._g_ax.grid(True)

            # ---- create a second persistent figure/axis for LOCAL PSO ----
            self._l_fig, self._l_ax = plt.subplots(figsize=(5,3))
            self._l_ax.set_title("Local PSO Convergence")
            self._l_ax.set_xlabel("Iteration")
            self._l_ax.set_ylabel("Best Cost")
            self._l_ax.grid(True)

    def communicate(self, aircraft_list, dt, add_con_plot):
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

        self.global_histories.append(global_pso.history)

        if add_con_plot:
            # update the global plot
            self._g_ax.clear()
            self._g_ax.plot(global_pso.history, marker='o')
            self._g_ax.set_title("Global PSO Convergence")
            self._g_ax.set_xlabel("Iteration")
            self._g_ax.set_ylabel("Best Cost")
            self._g_ax.grid(True)
            self._g_fig.canvas.draw()
            self._g_fig.canvas.flush_events()

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
                step_skip=3,
                enforce_overtake=False    # ← turn on no‑overtake here
            )
            local_speeds = local_pso.optimize()

            self.local_histories.append(local_pso.history)
            
            if add_con_plot:
                # update the local plot
                self._l_ax.clear()
                self._l_ax.plot(local_pso.history, marker='o')
                self._l_ax.set_title("Local PSO Convergence")
                self._l_ax.set_xlabel("Iteration")
                self._l_ax.set_ylabel("Best Cost")
                self._l_ax.grid(True)
                self._l_fig.canvas.draw()
                self._l_fig.canvas.flush_events()
            
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
          visualize=True,
          add_con_plot = True,
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
          pso_particles=24,
          pso_w=0.5,
          pso_c1=1.2,
          pso_c2=1.2,
          pso_iters=24,
          horizon_steps=3000,
          step_skip=20,
          local_comm_radius=20000,
          local_horizon=40,
          local_sep_weight=10000.0, 
          LOCAL_SEP_RADIUS = 5000):
    
    atc = ATCAgent(min_speed, max_speed, acceleration,
                   min_sep, sep_weight, fuel_weight,
                   pso_particles, pso_w, pso_c1, pso_c2, pso_iters,
                   horizon_steps, step_skip,
                   local_comm_radius, local_horizon, local_sep_weight, add_con_plot)
    
    aircraft_list = []
    spawn_t = pso_t = sim_t = 0.0
    collisions = throughput = 0
    arrival_times = []

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
            atc.communicate(aircraft_list, dt, add_con_plot)
            pso_t -= pso_interval

        for ac in aircraft_list[:]:
            ac.update(dt, aircraft_list, LOCAL_SEP_RADIUS)
            if ac.is_finished():
                aircraft_list.remove(ac)
                arrival_times.append(sim_t)
                throughput += 1

        for i, ac1 in enumerate(aircraft_list):
            for ac2 in aircraft_list[i+1:]:
                if ac1.distance_to(ac2) < min_sep:
                    collisions += 1

        if visualize:
            viz.aircraft_list = aircraft_list
            viz.draw(sim_t)

        sim_t += dt

    if visualize:
        pygame.quit()

    # KPI: Compute arrival spread
    arrival_times.sort()
    if len(arrival_times) >= 2:
        inter_arrivals = np.diff(arrival_times)
        mean_arrival_gap = float(np.mean(inter_arrivals))
        std_arrival_gap  = float(np.std(inter_arrivals))
        min_arrival_gap  = float(np.min(inter_arrivals))
    else:
        mean_arrival_gap = std_arrival_gap = min_arrival_gap = None

    return {
        'collisions': collisions,
        'throughput': throughput,
        'mean_arrival_gap': mean_arrival_gap,
        'std_arrival_gap': std_arrival_gap,
        'min_arrival_gap': min_arrival_gap,

        # for plotting convergence 
        'global_histories': atc.global_histories,
        'local_histories':  atc.local_histories,
    }



# import numpy as np
# import matplotlib.pyplot as plt
# from paths import paths_dict

# def main():
#     # Single full simulation
#     out = runme(
#         paths_dict=paths_dict,
#         visualize=False,
#         add_con_plot=False,
#         sim_duration=2000.0,
#         spawn_interval=300.0,
#         min_sep=3000.0,
#         sep_weight=1,
#         fuel_weight=0,
#         pso_particles=20,
#         pso_iters=10,
#         horizon_steps=3000,
#         step_skip=20
#     )

#     global_hist = out['global_histories']  # list of lists
#     local_hist  = out['local_histories']   # list of lists

#     # # Plot every global PSO history
#     # plt.figure(figsize=(8,5))
#     # for h in global_hist:
#     #     plt.plot(range(1, len(h)+1), h, color='grey', alpha=0.7, linewidth=1)
#     # # plot avarge of globval history 
#     # plt.plot(range(1, len(h)+1), global_hist.mean(axis=0), color='black', linewidth=2, label='Mean')
#     # plt.title("Global PSO Convergence")
#     # plt.xlabel("Iteration")
#     # plt.ylabel("Best Cost")
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig("global_all_histories.png", dpi=300)
#     # plt.show()
#     # ── Global PSO histories ──
#     min_len_g = min(len(h) for h in global_hist)
#     G = np.array([h[:min_len_g] for h in global_hist])
#     xg = np.arange(1, min_len_g + 1)

#     plt.figure(figsize=(8,5))
#     for h in G:
#         plt.plot(xg, h, color='blue', alpha=0.4, linewidth=1)
#     # plot average
#     plt.plot(xg, G.mean(axis=0), color='black', linewidth=2, label='Mean')
#     # plt.ylim(1000000, 1006000)
#     plt.title("Global PSO Convergence")
#     plt.xlabel("Iteration")
#     plt.ylabel("Best Cost")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("global_all_histories.png", dpi=300)
#     plt.show()

#     # # Plot every local PSO history
#     # plt.figure(figsize=(8,5))
#     # for h in local_hist:
#     #     plt.plot(range(1, len(h)+1), h, color='grey', alpha=0.7, linewidth=1)
#     # # plot avarge of local history 
#     # plt.plot(range(1, len(h)+1), local_hist.mean(axis=0), color='black', linewidth=2, label='Mean')
#     # plt.title("Local PSO Convergence")
#     # plt.xlabel("Iteration")
#     # plt.ylabel("Best Cost")
#     # plt.grid(True)
#     # plt.tight_layout()
#     # plt.savefig("local_all_histories.png", dpi=300)
#     # plt.show()

#     if local_hist:
#         min_len_l = min(len(h) for h in local_hist)
#         L = np.array([h[:min_len_l] for h in local_hist])
#         xl = np.arange(1, min_len_l + 1)

#         plt.figure(figsize=(8,5))
#         for h in L:
#             plt.plot(xl, h, color='blue', alpha=0.4, linewidth=1)
#         # plot average
#         plt.plot(xl, L.mean(axis=0), color='black', linewidth=2, label='Mean')
#         plt.title("Local PSO Convergence")
#         plt.xlabel("Iteration")
#         plt.ylabel("Best Cost")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("local_all_histories.png", dpi=300)
#         plt.show()

    

# if __name__ == '__main__':
#     main()



if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import os

    from paths import paths_dict

    # where to save intermediate & final results
    SAVE_DIR = os.path.expanduser("~/Documents/SimulationResults")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # === PSO hyperparameter grid ===
    particles  = [25, 50, 100]
    iterations = [5, 10, 20]
    n_runs     = 5

    pso_violations = np.zeros((len(iterations), len(particles)))
    pso_times      = np.zeros_like(pso_violations)

    for i, iters in enumerate(tqdm(iterations, desc="Iters")):
        for j, npart in enumerate(particles):
            coll_list, ct_list = [], []
            print(f'Running: {iters} itterations and {npart} particles')
            for run in range(n_runs):
                sim_kwargs = dict(
                    paths_dict=paths_dict,
                    sim_duration=20000.0,
                    visualize=False,
                    add_con_plot=False,      # no per-call plotting
                    initial_speed=100.0,
                    acceleration=1.0,
                    min_speed=80.0,
                    max_speed=130.0,
                    spawn_interval=300.0,
                    fps=30,
                    min_sep=3000.0,
                    sep_weight=1,
                    fuel_weight=0,
                    pso_particles=npart,
                    pso_w=0.64,
                    pso_c1=1.5,
                    pso_c2=1.74,
                    pso_iters=iters,
                    horizon_steps=3000,
                    step_skip=20,
                    local_comm_radius=20000,
                    local_horizon=40,
                    local_sep_weight=10000.0,
                    LOCAL_SEP_RADIUS=6000
                )
                t0 = time.perf_counter()
                out = runme(**sim_kwargs)
                coll_list.append(out['collisions'])
                ct_list.append(time.perf_counter() - t0)

            pso_violations[i, j] = np.mean(coll_list)
            pso_times[i, j]      = np.mean(ct_list)

            # save as you go
            np.save(os.path.join(SAVE_DIR, "pso_violations.npy"), pso_violations)
            np.save(os.path.join(SAVE_DIR, "pso_times.npy"), pso_times)

    # === Plotting ===
    P, I = np.meshgrid(particles, iterations)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    cs0 = axs[0].contourf(P, I, pso_violations, cmap='viridis')
    axs[0].set_title("Mean Collisions vs PSO Params")
    axs[0].set_xlabel("Particles")
    axs[0].set_ylabel("Iterations")
    fig.colorbar(cs0, ax=axs[0], label="Avg Collisions")

    cs1 = axs[1].contourf(P, I, pso_times, cmap='viridis')
    axs[1].set_title("Mean Compute Time vs PSO Params")
    axs[1].set_xlabel("Particles")
    axs[1].set_ylabel("Iterations")
    fig.colorbar(cs1, ax=axs[1], label="Avg Time (s)")

    plt.tight_layout()
    fig_path = os.path.join(SAVE_DIR, "pso_sensitivity_contours.png")
    plt.savefig(fig_path, dpi=300)
    plt.show()



# if __name__ == '__main__':
#     import time
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Simulation parameters
#     sim_kwargs = dict(
#         paths_dict=paths_dict,
#         sim_duration=20000.0,
#         visualize=True,
#         add_con_plot=True,
#         width=1024,
#         height=768,
#         initial_speed=100.0,
#         acceleration=1,
#         min_speed=80.0,
#         max_speed=130.0,
#         spawn_interval=300.0,
#         pso_interval=50.0,
#         time_scale=100.0,
#         fps=30,
#         min_sep=3000.0,
#         sep_weight=1,
#         fuel_weight=0,
#         pso_particles=100,
#         pso_w=0.643852371671638,
#         pso_c1=1.5066396693442399,
#         pso_c2=1.7414431113477675,
#         pso_iters=20,
#         horizon_steps=3000,
#         step_skip=20,
#         local_comm_radius=20000,
#         local_horizon=40,
#         local_sep_weight=10000.0,
#         LOCAL_SEP_RADIUS=6000
#     )

#     # Run the simulation multiple times, measuring compute time
#     results = []
#     for i in range(30):
#         t0 = time.perf_counter()
#         res = runme(**sim_kwargs)
#         res['comp_time'] = time.perf_counter() - t0

#         # Per‐run printout
#         if res['mean_arrival_gap'] is not None:
#             print(f"Run {i+1}: "
#                   f"collisions={res['collisions']}, "
#                   f"throughput={res['throughput']}, "
#                   f"mean_gap={res['mean_arrival_gap']:.1f}s, "
#                   f"std_gap={res['std_arrival_gap']:.1f}s, "
#                   f"min_gap={res['min_arrival_gap']:.1f}s, "
#                   f"comp_time={res['comp_time']:.2f}s")
#         else:
#             print(f"Run {i+1}: insufficient arrivals for gap metrics, "
#                   f"comp_time={res['comp_time']:.2f}s")

#         results.append(res)

#     # Extract KPI arrays
#     collisions = np.array([r['collisions']        for r in results])
#     throughput = np.array([r['throughput']        for r in results])
#     mean_gaps  = np.array([r['mean_arrival_gap']  for r in results
#                            if r['mean_arrival_gap'] is not None])
#     std_gaps   = np.array([r['std_arrival_gap']   for r in results
#                            if r['std_arrival_gap'] is not None])
#     min_gaps   = np.array([r['min_arrival_gap']   for r in results
#                            if r['min_arrival_gap'] is not None])
#     comp_times = np.array([r['comp_time']         for r in results])

#     # Print overall summary
#     print("\nSummary over 30 runs:")
#     print(f"  Collisions:    mean = {collisions.mean():.2f}, std = {collisions.std(ddof=1):.2f}")
#     print(f"  Throughput:    mean = {throughput.mean():.2f}, std = {throughput.std(ddof=1):.2f}")
#     print(f"  Comp. Time:    mean = {comp_times.mean():.2f}s, std = {comp_times.std(ddof=1):.2f}s")

#     if len(mean_gaps) > 0:
#         print(f"  Mean Arrival Gap: mean = {mean_gaps.mean():.1f}s, std = {mean_gaps.std(ddof=1):.1f}s")
#         print(f"  Std Arrival Gap:  mean = {std_gaps.mean():.1f}s, std = {std_gaps.std(ddof=1):.1f}s")
#         print(f"  Min Arrival Gap:  mean = {min_gaps.mean():.1f}s, std = {min_gaps.std(ddof=1):.1f}s")
#     else:
#         print("  Not enough arrivals to compute gap statistics.")

#     # Plot histograms for each KPI
#     kpis = {
#         'Collisions': collisions,
#         'Throughput': throughput,
#         'Mean Gap (s)': mean_gaps,
#         'Std Gap (s)': std_gaps,
#         'Min Gap (s)': min_gaps,
#         'Comp Time (s)': comp_times
#     }

#     for name, data in kpis.items():
#         plt.figure()
#         plt.hist(data, bins=10)
#         plt.title(f"Histogram of {name}")
#         plt.xlabel(name)
#         plt.ylabel("Frequency")
#         plt.show()


# import random
# import copy
# sim_kwargs = dict(
#     paths_dict=paths_dict,
#     sim_duration=20000.0,
#     visualize=False,
#     width=1024,
#     height=768,
#     initial_speed=100.0,
#     acceleration=0.5,
#     min_speed=80.0,
#     max_speed=130.0,
#     spawn_interval=400.0,
#     pso_interval=50.0,
#     time_scale=100.0,
#     fps=30,
#     min_sep=3000.0,
#     sep_weight=100,
#     fuel_weight=1,
#     pso_particles=50,
#     pso_w=0.643852371671638,
#     pso_c1=1.5066396693442399,
#     pso_c2=1.7414431113477675,
#     pso_iters=30,
#     horizon_steps=3000,
#     step_skip=20,
#     local_comm_radius=20000,
#     local_horizon=40,
#     local_sep_weight=10000.0
# )


# def random_genome():
#     return {
#         'w':            random.uniform(0.1, 1.0),
#         'c1':           random.uniform(0.5, 2.5),
#         'c2':           random.uniform(0.5, 2.5),
#         'n_particles':  random.randint(5, 30),
#         'max_iter':     random.randint(5, 50),
#     }

# def mutate(genome, rate=0.2):
#     # mutate floats
#     for key in ('w','c1','c2'):
#         if random.random() < rate:
#             genome[key] += random.uniform(-0.1, 0.1)
#             genome[key] = max(0.0, min(3.0, genome[key]))
#     # mutate ints
#     for key, lo, hi in (('n_particles',5,30), ('max_iter',5,50)):
#         if random.random() < rate:
#             genome[key] += random.choice([-1,1])
#             genome[key] = max(lo, min(hi, genome[key]))

# def fitness(genome):
#     # override sim_kwargs with this genome’s PSO settings
#     params = {
#         **sim_kwargs,
#         'visualize': False,
#         'pso_particles': genome['n_particles'],
#         'pso_w':         genome['w'],
#         'pso_c1':        genome['c1'],
#         'pso_c2':        genome['c2'],
#         'pso_iters':     genome['max_iter']
#     }
#     res = runme(**params)
#     score = res['throughput'] - 1000 * res['collisions']
#     print(f"  {genome} → coll={res['collisions']}, thr={res['throughput']}, score={score}")
#     return score

# def evolve_population(pop_size=10, generations=5, retain=0.5, mutate_rate=0.2):
#     population = [random_genome() for _ in range(pop_size)]
#     for gen in range(generations):
#         scored = sorted([(fitness(g), g) for g in population],
#                         key=lambda x: x[0], reverse=True)
#         best_score, best_genome = scored[0]
#         print(f"Gen {gen}: best={best_genome} (score={best_score})")
#         survivors = [g for _, g in scored[:int(pop_size * retain)]]
#         # refill
#         population = []
#         while len(population) < pop_size:
#             parent = copy.deepcopy(random.choice(survivors))
#             mutate(parent, rate=mutate_rate)
#             population.append(parent)
#     return best_genome

# def main():
#     best = evolve_population(pop_size=8, generations=6, retain=0.5, mutate_rate=0.3)
#     print(f"\nEvolved hyperparameters: {best}\n")
#     # final visual run
#     final_params = {
#         **sim_kwargs,
#         'visualize': True,
#         'pso_particles': best['n_particles'],
#         'pso_w':         best['w'],
#         'pso_c1':        best['c1'],
#         'pso_c2':        best['c2'],
#         'pso_iters':     best['max_iter']
#     }
#     results = runme(**final_params)
#     print("Final KPIs:", results)

# if __name__ == "__main__":
#     main()


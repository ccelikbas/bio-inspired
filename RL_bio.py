#!/usr/bin/env python3
# RL_abm.py

import random
import numpy as np
import pygame
import gym
from gym import spaces
from stable_baselines3 import PPO
from paths import paths_dict

# approximate meters per degree latitude/longitude around Amsterdam
M_PER_DEG = 111_320

class MultiAircraftEnv(gym.Env):
    """
    Gym environment for multi-aircraft separation on approach,
    with optional pygame rendering. Automatically computes
    a horizon long enough for aircraft to complete the longest path.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 max_ac=3,
                 min_sep=3000.0,
                 dt=10.0,
                 horizon=None,
                 render_mode=False,
                 width=800,
                 height=600):
        super().__init__()
        # core parameters
        self.max_ac       = max_ac
        self.min_sep      = min_sep
        self.dt           = dt
        self.current_step = 0
        self.render_mode  = render_mode
        self.width        = width
        self.height       = height

        # geographic bounds for rendering
        lats = [lat for p in paths_dict.values() for lat, lon in p]
        lons = [lon for p in paths_dict.values() for lat, lon in p]
        self.lat_min, self.lat_max = min(lats), max(lats)
        self.lon_min, self.lon_max = min(lons), max(lons)

        # precompute each path's cumulative distances & waypoint vectors
        self.paths = list(paths_dict.values())
        self.path_cumlens = []
        self.path_pts     = []
        for path in self.paths:
            pts = [pygame.math.Vector2(lon, lat) for lat, lon in path]
            dist = [0.0]
            for A, B in zip(pts[:-1], pts[1:]):
                mean_lat = np.deg2rad((A.y + B.y) / 2)
                dlat = (B.y - A.y) * M_PER_DEG
                dlon = (B.x - A.x) * M_PER_DEG * np.cos(mean_lat)
                dist.append(np.hypot(dlat, dlon))
            cum = np.cumsum(dist)
            self.path_cumlens.append(cum)
            self.path_pts.append(pts)

        # determine horizon if not provided
        if horizon is None:
            # estimate time to traverse longest path at 100 m/s
            max_dist = max(c[-1] for c in self.path_cumlens)
            est_steps = max_dist / (100.0 * self.dt)
            self.horizon = int(np.ceil(est_steps)) + 20  # cushion
        else:
            self.horizon = horizon

        # action: speed for each aircraft
        self.action_space = spaces.Box(
            low  = np.full((self.max_ac,),  80.0, dtype=np.float32),
            high = np.full((self.max_ac,), 130.0, dtype=np.float32),
            dtype=np.float32
        )
        # observation: remaining distance to runway for each aircraft
        max_len = max(c[-1] for c in self.path_cumlens)
        self.observation_space = spaces.Box(
            low  = np.zeros((self.max_ac,), dtype=np.float32),
            high = np.full((self.max_ac,), max_len, dtype=np.float32),
            dtype=np.float32
        )

        # initialize rendering
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Multi‑Aircraft Approach")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # spawn distinct aircraft on random paths
        chosen = random.sample(range(len(self.paths)), k=self.max_ac)
        self.ac_states = []
        for idx in chosen:
            self.ac_states.append({
                'path_idx': idx,
                'dist':     0.0,    # meters along path
                'speed':    100.0   # initial speed
            })
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        # 1) apply speeds
        for i, sp in enumerate(action):
            self.ac_states[i]['speed'] = float(sp)

        # 2) advance each aircraft and compute positions
        positions = []
        for state in self.ac_states:
            # integrate forward
            state['dist'] = min(
                state['dist'] + state['speed'] * self.dt,
                self.path_cumlens[state['path_idx']][-1]
            )
            cum = self.path_cumlens[state['path_idx']]
            pts = self.path_pts[state['path_idx']]
            idx = np.searchsorted(cum, state['dist'], side='right') - 1
            if idx >= len(pts) - 1:
                positions.append(pts[-1])
            else:
                A, B = pts[idx], pts[idx+1]
                seg_len = cum[idx+1] - cum[idx]
                frac = ((state['dist'] - cum[idx]) / seg_len) if seg_len>0 else 0.0
                positions.append(A + (B - A) * frac)

        # 3) compute reward
        reward = 0.0
        # a) penalty for collisions
        for i in range(self.max_ac):
            for j in range(i+1, self.max_ac):
                d = positions[i].distance_to(positions[j])
                if d < self.min_sep:
                    reward -= 1000.0 * (self.min_sep - d)
        # b) bonus for each aircraft that finished
        for state in self.ac_states:
            if state['dist'] >= self.path_cumlens[state['path_idx']][-1]:
                reward += 10.0

        self.current_step += 1
        # end when all landed or horizon reached
        all_done = all(
            state['dist'] >= self.path_cumlens[state['path_idx']][-1]
            for state in self.ac_states
        )
        done = all_done or (self.current_step >= self.horizon)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        rems = []
        for state in self.ac_states:
            total = self.path_cumlens[state['path_idx']][-1]
            rems.append(total - state['dist'])
        rems += [0.0] * (self.max_ac - len(rems))
        return np.array(rems, dtype=np.float32)

    def render(self, mode='human'):
        if not self.render_mode:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
        self.screen.fill((255,255,255))
        # draw paths
        for pts in self.path_pts:
            px = [self._geo_to_px(p.y, p.x) for p in pts]
            if len(px)>1:
                pygame.draw.lines(self.screen, (200,200,200), False, px, 2)
        # draw aircraft
        colors = [(255,0,0),(0,128,0),(0,0,255),(255,165,0),(128,0,128)]
        for i, state in enumerate(self.ac_states):
            cum = self.path_cumlens[state['path_idx']]
            pts = self.path_pts[state['path_idx']]
            d   = min(state['dist'], cum[-1])
            idx = np.searchsorted(cum, d, side='right') - 1
            if idx >= len(pts)-1:
                pos = pts[-1]
            else:
                A, B = pts[idx], pts[idx+1]
                seg_len = cum[idx+1] - cum[idx]
                frac = ((d - cum[idx]) / seg_len) if seg_len>0 else 0.0
                pos = A + (B - A)*frac
            x, y = self._geo_to_px(pos.y, pos.x)
            pygame.draw.circle(self.screen, colors[i%len(colors)], (x,y), 6)
        pygame.display.flip()
        self.clock.tick(30)

    def _geo_to_px(self, lat, lon):
        x = (lon - self.lon_min) / (self.lon_max - self.lon_min) * self.width
        y = self.height - (lat - self.lat_min) / (self.lat_max - self.lat_min) * self.height
        return int(x), int(y)

    def close(self):
        if self.render_mode:
            pygame.quit()


def train(save_path="ppo_aircraft_approach.zip",
          max_ac=3,
          min_sep=3000.0,
          dt=10.0,
          horizon=None,
          total_timesteps=1_000_000):
    env = MultiAircraftEnv(
        max_ac      = max_ac,
        min_sep     = min_sep,
        dt          = dt,
        horizon     = horizon,
        render_mode = False
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_aircraft_tb/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    env.close()
    print(f"Model trained (dt={dt}s, horizon={env.horizon} steps) "
          f"for {total_timesteps} timesteps → saved to {save_path}")
    return save_path


def test(model_path,
         max_ac=3,
         dt=10.0,
         horizon=None):
    env = MultiAircraftEnv(
        max_ac      = max_ac,
        min_sep     = 3000.0,
        dt          = dt,
        horizon     = horizon,
        render_mode = True
    )
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        env.render()
        total_reward += reward
    print(f"Test run (dt={dt}s, horizon={env.horizon} steps) completed, "
          f"total reward = {total_reward:.1f}")
    env.close()


if __name__ == "__main__":
    model_file = "ppo_aircraft_approach.zip"
    train(
        save_path=model_file,
        max_ac=3,
        min_sep=3000.0,
        dt=10.0,
        horizon=None,
        total_timesteps=1_000_000
    )
    test(
        model_path=model_file,
        max_ac=3,
        dt=10.0,
        horizon=None
    )

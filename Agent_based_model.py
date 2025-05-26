#!/usr/bin/env python3
import numpy as np
import pygame
import math
import argparse

from functions import get_xy, circle_lat, circle_lon
from paths import paths_dict  # your precomputed {radial: [(lat,lon), ...]}

# --------------------- Aircraft & Simulator ---------------------
class Aircraft:
    def __init__(self, id, path, t_spawn, v_nom):
        self.id = id
        self.path = path
        self.L = len(path)
        self.s = 0
        self.t_spawn = t_spawn
        self.v_nom = v_nom

    def advance(self, dt):
        ds = self.v_nom * dt
        self.s = min(self.L-1, int(self.s + ds))

    @property
    def done(self):
        return self.s >= self.L-1

class Simulator:
    def __init__(self, R, ils_point, runway_point, params):
        pygame.init(); pygame.font.init()
        size = 800
        self.screen = pygame.display.set_mode((size, size))
        self.scale = size/(2*R)
        self.offset = np.array([size/2, size/2])
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, params['font_size'])
        self.R = R
        self.ils = ils_point
        self.runway = runway_point
        self.params = params
        self.aircraft = []
        self.time = 0.0

    def world2screen(self, pt):
        return tuple((np.array(pt)*self.scale + self.offset).astype(int))

    def spawn_aircraft(self):
        radial = np.random.choice(list(paths_dict.keys()))
        latlons = paths_dict[radial]
        xy = [get_xy(lat, lon) for lat, lon in latlons]

        # straight final approach this can be removed this is included in the paths

        full_path = xy 
        self.aircraft.append(Aircraft(len(self.aircraft),
                                      full_path,
                                      self.time,
                                      self.params['v_nom']))

    def run(self):
        spawn_rate = self.params['lambda']
        running = True
        while running:
            dt = self.params['dt'] * self.params['sim_speed']
            self.time += dt
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            if np.random.rand() < spawn_rate*dt:
                self.spawn_aircraft()

            self.screen.fill((255,255,255))
            # boundary
            for i in range(self.params['boundary_dashes']):
                a1 = 2*math.pi*i/self.params['boundary_dashes']
                a2 = 2*math.pi*(i+0.5)/self.params['boundary_dashes']
                p1 = (self.runway[0]+self.R*math.cos(a1),
                      self.runway[1]+self.R*math.sin(a1))
                p2 = (self.runway[0]+self.R*math.cos(a2),
                      self.runway[1]+self.R*math.sin(a2))
                pygame.draw.line(self.screen, (0,0,0),
                                 self.world2screen(p1),
                                 self.world2screen(p2), 1)
            # runway & ILS
            pygame.draw.line(self.screen, (200,0,0),
                             self.world2screen(self.ils),
                             self.world2screen(self.runway), 4)
            pygame.draw.circle(self.screen, (0,0,200),
                               self.world2screen(self.ils), 5)
            # aircraft
            for ac in list(self.aircraft):
                ac.advance(dt)
                pos = self.world2screen(ac.path[ac.s])
                pygame.draw.circle(self.screen, (0,150,0),
                                   pos, self.params['dot_size'])
                txt = self.font.render(str(ac.id), True, (0,0,0))
                self.screen.blit(txt, (pos[0]+self.params['dot_size']+2,
                                       pos[1]-self.params['dot_size']))
                for j in range(0, len(ac.path)-1, self.params['dash_step']):
                    p1 = self.world2screen(ac.path[j])
                    p2 = self.world2screen(ac.path[j+1])
                    pygame.draw.line(self.screen, (150,150,150), p1, p2, 1)
                if ac.done:
                    self.aircraft.remove(ac)

            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out", help="unused")
    args = parser.parse_args()

    params = {
        'ds':         2.0,
        'v_nom':      100.0,
        'lambda':     0.1,
        'dt':         1.0,
        'sim_speed':  0.1,
        'dot_size':   10,
        'font_size':  18,
        'dash_step':  5,
        'boundary_dashes': 60
    }

    ils_xy = get_xy(circle_lat, circle_lon)
    runway_xy = (0.0, 0.0)

    sim = Simulator(
        R=300.0,                  
        ils_point=ils_xy,         
        runway_point=runway_xy,   
        params=params
    )
    sim.run()

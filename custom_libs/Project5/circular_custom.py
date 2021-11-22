import numpy as np
import time
from tkinter import *
import pygame

from custom_libs import ColorizedLogger
from custom_libs.CARLO.world import World
from custom_libs.CARLO.agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from custom_libs.CARLO.geometry import Point
from custom_libs.CARLO.steering_control import PIDSteering

logger = ColorizedLogger('Circular Scenario', 'cyan')


class CircularScenario:
    def __init__(self, width: int, height: int, dt: float, build_radius: float,
                 num_lanes: int, lane_marker_width: float, num_lane_markers: int, lane_width: float):

        self.w = World(dt=dt, width=width, height=height, ppm=6)
        self.width = width
        self.height = height
        self.build_radius = build_radius
        self.num_lane_markers = num_lane_markers
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.lane_marker_width = lane_marker_width
        self.dt = dt
        self.cb, self.rb, self.c1 = None, None, None

    def build_scene(self):

        cb = CircleBuilding(Point(self.width / 2, self.height / 2), self.build_radius, 'gray80')
        self.w.add(cb)

        rb = RingBuilding(Point(self.width / 2, self.height / 2),
                          self.build_radius + self.num_lanes * self.lane_width + (
                                  self.num_lanes - 1) * self.lane_marker_width,
                          1 + np.sqrt((self.width / 2) ** 2 + (self.height / 2) ** 2), 'gray80')
        self.w.add(rb)

        # Add lane markers
        for lane_no in range(self.num_lanes - 1):
            lane_markers_radius = self.build_radius + (lane_no + 1) * self.lane_width \
                                  + (lane_no + 0.5) * self.lane_marker_width
            lane_marker_height = np.sqrt(2 * (lane_markers_radius ** 2) * (1 - np.cos((2 * np.pi) / (
                    2 * self.num_lane_markers))))
            for theta in np.arange(0, 2 * np.pi, 2 * np.pi / self.num_lane_markers):
                dx = lane_markers_radius * np.cos(theta)
                dy = lane_markers_radius * np.sin(theta)
                self.w.add(Painting(Point(self.width / 2 + dx, self.height / 2 + dy),
                                    Point(self.lane_marker_width, lane_marker_height), 'white',
                                    heading=theta))

        # Create Car
        c1 = Car(Point(91.75, 60), np.pi / 2)
        c1.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
        c1.velocity = Point(0, 3.0)
        self.w.add(c1)
        return cb, rb, c1

    def run(self, human_control: bool = False):
        self.cb, self.rb, self.c1 = self.build_scene()
        self.w.render()
        try:
            if human_control:
                self._run_with_human_control()
            else:
                self._run_automatic()
        except Exception as e:
            logger.info(e)

        self.w.close()
        pygame.display.quit()
        pygame.quit()

    def _run_automatic(self):
        # Let's implement some simple policy for the car c1
        desired_lane = 1
        return_val = 0
        # pid_controller1 = PIDSteering(2.0, 0.001, 0.01, dt)
        for k in range(600):
            lp = 0.
            if self.c1.distanceTo(self.cb) < desired_lane * (
                    self.lane_width + self.lane_marker_width) + 0.2:
                lp += 0.
            elif self.c1.distanceTo(self.rb) < (self.num_lanes - desired_lane - 1) * (
                    self.lane_width + self.lane_marker_width) + 0.3:
                lp += 1.

            v = self.c1.center - self.cb.center
            v = np.mod(np.arctan2(v.y, v.x) + np.pi / 2, 2 * np.pi)
            if self.c1.heading < v:
                lp += 0.7
            else:
                lp += 0.

            if np.random.rand() < lp:
                self.c1.set_control(0.2, 0.1)
            else:
                self.c1.set_control(-0.1, 0.1)

            self.w.tick()  # This ticks the world for one time step (dt second)
            self.w.render()
            time.sleep(self.dt / 4)  # Let's watch it 4x

            if self.w.collision_exists():  # We can check if there is any collision at all.
                logger.info('Collision exists somewhere...')
                return_val = 1
                break

        return return_val

    def _run_with_human_control(self):
        from custom_libs.CARLO.interactive_controllers import KeyboardController
        return_val = 0
        self.c1.set_control(0., 0.)  # Initially, the car will have 0 steering and 0 throttle.
        controller = KeyboardController(self.w)
        for k in range(600):
            self.c1.set_control(controller.steering, controller.throttle)
            self.w.tick()  # This ticks the world for one time step (dt second)
            self.w.render()
            time.sleep(self.dt / 4)  # Let's watch it 4x
            if self.w.collision_exists():
                # import sys
                # sys.exit(0)
                return_val = 1
                break

        return return_val

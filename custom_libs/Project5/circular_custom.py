import numpy as np
import copy
import time
import pickle
import pygame
from matplotlib import pyplot as plt

from custom_libs import ColorizedLogger

import sys

sys.path.append("custom_libs/CARLO")
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting
from geometry import Point
from interactive_controllers import KeyboardController

logger = ColorizedLogger('Circular Scenario', 'cyan')


class CircularScenario:
    def __init__(self, width: int, height: int, dt: float, build_radius: float,
                 num_lanes: int, lane_marker_width: float, num_lane_markers: int,
                 lane_width: float, rl_path: str = 'data/car1_rl.pkl'):

        self.rl_path = rl_path
        self.w = World(dt=dt, width=width, height=height, ppm=6)
        self.width = width
        self.height = height
        self.build_radius = build_radius
        self.num_lane_markers = num_lane_markers
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.lane_marker_width = lane_marker_width
        self.dt = dt
        self.c1, self.cb, self.rb, self.c1 = None, None, None, None

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
        return c1, cb, rb, c1

    def run(self, opt: str = 'default', max_iters: int = 600, speed: int = 4, plot: bool = False):
        self.c1, self.cb, self.rb, self.c1 = self.build_scene()
        self.w.add(self.c1)
        self.w.render()
        try:
            if opt == 'default':
                err_code, error_vals = self._run_default(max_iters=max_iters, speed=speed)
            elif opt == 'pid':
                err_code, error_vals = self._run_pid(max_iters=max_iters, speed=speed)
            elif opt == 'qlearning':
                err_code, error_vals = self._run_qlearning(max_iters=max_iters, speed=speed)
            elif opt == 'human_control':
                err_code, error_vals = self._run_with_human_control(max_iters=max_iters, speed=speed)
            else:
                raise NotImplementedError('Option not yet implemented!')
            if plot:
                plt.plot(np.arange(len(error_vals)), error_vals)
        except Exception as e:
            self.w.close()
            pygame.display.quit()
            pygame.quit()
            logger.warn(str(e))
            raise e

        self.w.close()
        pygame.display.quit()
        pygame.quit()

        return error_vals

    def _run_default(self, max_iters: int = 600, speed: int = 4):
        # Let's implement some simple policy for the car c1
        desired_lane = 1
        return_val = 0
        errors = []
        for k in range(max_iters):
            lp = 0.
            if self.c1.distanceTo(self.cb) < desired_lane * (
                    self.lane_width + self.lane_marker_width) + 0.2:
                lp += 0.
            elif self.c1.distanceTo(self.rb) < (self.num_lanes - desired_lane - 1) * (
                    self.lane_width + self.lane_marker_width) + 0.3:
                lp += 1.

            v = self.c1.center - self.cb.center
            v = np.mod(np.arctan2(v.y, v.x) + np.pi / 2, 2 * np.pi)
            err = v - self.c1.heading
            errors.append(err)
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
            time.sleep(self.dt / speed)

            if self.w.collision_exists():  # We can check if there is any collision at all.
                logger.info('Collision exists somewhere...')
                return_val = 1
                break

        return return_val, errors

    def _run_pid(self, max_iters: int = 600, speed: int = 4):

        pid_controller1 = PIDSteering(2.0, 0.001, 0.01, self.dt)
        errors = []
        return_val = 0
        for k in range(max_iters):
            v1 = self.c1.center - self.cb.center
            v1 = np.mod(np.arctan2(v1.y, v1.x) + np.pi / 2, 2 * np.pi)
            err1 = v1 - self.c1.heading
            errors.append(err1)
            heading_new1 = pid_controller1.correct(err1)
            self.c1.set_control(heading_new1, 0.06)

            self.w.tick()  # This ticks the world for one time step (dt second)
            self.w.render()
            time.sleep(self.dt / speed)  # Let's watch it 4x

            if self.w.collision_exists():  # We can check if there is any collision at all.
                logger.info('Collision exists somewhere...')
                return_val = 1
                break

        return return_val, errors

    def _run_with_human_control(self, max_iters: int = 600, speed: int = 4):
        return_val = 0
        self.c1.set_control(0., 0.)  # Initially, the car will have 0 steering and 0 throttle.
        controller = KeyboardController(self.w)
        errors = []
        for k in range(max_iters):
            self.c1.set_control(controller.steering, controller.throttle)

            v = self.c1.center - self.cb.center
            v = np.mod(np.arctan2(v.y, v.x) + np.pi / 2, 2 * np.pi)
            err1 = v - self.c1.heading
            errors.append(err1)

            self.w.tick()  # This ticks the world for one time step (dt second)
            self.w.render()
            time.sleep(self.dt / speed)
            if self.w.collision_exists():
                logger.info("Collision exists..")
                return_val = 1
                break

        return return_val, errors

    def _run_qlearning(self, max_iters: int = 600, speed: int = 4):
        c1_rl = RLSteering(self.c1, self.w)
        saved_rl_obj = self.read_rl_policy(self.rl_path)
        c1_rl.overwrite(saved_rl_obj)

        self.w.render()  # This visualizes the world we just constructed.
        errors = []
        return_val = 0
        for k in range(max_iters):
            rl_action = c1_rl.get_rl_action(self.c1.center.x, self.c1.center.y)
            heading_new = self.c1.heading + rl_action
            self.c1.set_control(heading_new, 0.06)  # no acceleration

            v = self.c1.center - self.cb.center
            v = np.mod(np.arctan2(v.y, v.x) + np.pi / 2, 2 * np.pi)
            err1 = v - self.c1.heading
            errors.append(err1)

            self.w.tick()  # This ticks the world for one time step (dt second)
            self.w.render()
            time.sleep(self.dt / speed)

            if self.w.collision_exists():  # We can check if there is any collision at all.
                logger.info('Collision exists somewhere...')
                return_val = 1
                break

        return return_val, errors

    def train(self, epochs: int = 15000):
        self.c1 = self.cb, self.rb, self.c1 = self.build_scene()
        self.w.add(self.c1)
        self.w.render()

        c1_rl = RLSteering(self.c1, self.w, episode=epochs)
        c1_rl.q_learning()
        c1_rl.delete_vars()
        self.save_rl_policy(c1_rl, 'data/car1_rl.pkl')

        # close world
        self.w.close()
        pygame.display.quit()
        pygame.quit()

    @staticmethod
    def read_rl_policy(name: str):
        with open(name, 'rb') as f:
            var_rl = pickle.load(f)
        f.close()
        return var_rl

    @staticmethod
    def save_rl_policy(var_rl, name: str):
        with open(name, 'wb') as f:
            pickle.dump(var_rl, f)
        f.close()


class PIDSteering:
    def __init__(self, P=0.0, I=0.0, D=0.0, dt=0.1) -> None:
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.dt = dt
        self.err_prev = 0.0
        self.integral = 0.0

    def correct(self, err):
        proportional = err
        self.integral = self.integral + err * self.dt
        derivative = (err - self.err_prev) / self.dt
        output = self.Kp * proportional + self.Ki * self.integral + self.Kd * derivative
        self.err_prev = err
        return output


class RLSteering:
    def __init__(self, car, world, friction=0.06, alpha=0.1, gamma=0.9, epsilon=0.1, episode=1000,
                 resolution=0.2):
        self.car = copy.copy(car)
        self.world = copy.copy(world)
        self.car_x_init = self.car.center.x
        self.car_y_init = self.car.center.y
        self.car_heading_init = self.car.heading
        self.radius = np.sqrt(
            (self.car.center.x - self.world.width / 2) ** 2 + (
                    self.car.center.y - self.world.height / 2) ** 2)
        self.friction = friction
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = episode
        self.resolution = resolution
        self.action_theta = np.array(
            (-np.pi / 2, -np.pi / 3, -np.pi / 4, -np.pi / 6, 0, np.pi / 6, np.pi / 4, np.pi / 3,
             np.pi / 2))

        # size of policy and Q
        m = int(self.world.width / self.resolution) + 1
        n = int(self.world.height / self.resolution) + 1
        self.policy = np.full((m, n, len(self.action_theta)), 0.2)  # store probability of each action
        self.Q = np.zeros((m, n, len(self.action_theta)))

    def car_pos_reset(self):
        car = self.world.agents[-1]
        car.center.x = self.car_x_init
        car.center.y = self.car_y_init
        car.heading = self.car_heading_init
        return car

    def pos_to_index(self, pos):
        ind = round(pos / self.resolution)
        return ind

    def q_learning(self):
        for n_episode in range(self.episode):
            logger.info(f"Training epoch: {n_episode}")
            car = self.car_pos_reset()  # reset

            while True:
                heading = car.heading
                x = self.pos_to_index(car.center.x)
                y = self.pos_to_index(car.center.y)

                # choose an action; epsilon-greedy
                n_action = len(self.policy[x, y, :])
                epsilon_ind = np.argmax(self.policy[x, y, :])
                rnd = np.random.random()
                if rnd <= self.epsilon:
                    while True:
                        rnd_int = np.random.randint(n_action - 1)
                        if rnd_int >= epsilon_ind:
                            rnd_int += 1
                        action = rnd_int
                        if self.policy[x, y, action] != -1:
                            break
                else:
                    action = epsilon_ind
                theta = self.action_theta[action]
                heading_new = heading + theta  # new heading
                car.set_control(heading_new, self.friction)  # move
                self.world.tick()
                # self.world.render()

                # calculate reward
                x_new = self.pos_to_index(car.center.x)
                y_new = self.pos_to_index(car.center.y)
                collision = False
                if self.world.collision_exists():
                    reward = -1000
                    collision = True
                else:
                    reward = 10
                new_radius = np.sqrt(
                    (car.center.x - self.world.width / 2) ** 2 + (
                            car.center.y - self.world.height / 2) ** 2)
                reward -= 10 * abs(new_radius - self.radius)

                # update Q value
                max_q = max(self.Q[x_new, y_new, :])
                self.Q[x, y, action] += self.alpha * (
                        reward + self.gamma * max_q - self.Q[x, y, action])

                # update policy
                if collision is True:  # terminate
                    self.policy[x, y, action] = -1
                    break
                else:
                    max_q = max(self.Q[x, y, :])
                    n_max_q = (self.Q[x, y, :] == max_q).sum()
                    n_block = (self.policy[x, y, :] == -1).sum()
                    for i in range(n_action):
                        if self.Q[x, y, i] == max_q:
                            self.policy[x, y, i] = (1 - self.epsilon) / n_max_q
                        else:
                            self.policy[x, y, i] = self.epsilon / (n_action - n_max_q - n_block)
        return self.policy

    def get_rl_action(self, pos_x, pos_y):
        x = round(pos_x / self.resolution)
        y = round(pos_y / self.resolution)
        ind = np.argmax(self.policy[x, y, :])
        theta = self.action_theta[ind]
        return theta

    def delete_vars(self):
        del self.car, self.world

    def overwrite(self, obj):
        self.friction = obj.friction
        self.alpha = obj.alpha
        self.gamma = obj.gamma
        self.epsilon = obj.epsilon
        self.episode = obj.episode
        self.resolution = obj.resolution
        self.policy = obj.policy
        self.Q = obj.Q

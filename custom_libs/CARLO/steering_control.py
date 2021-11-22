from agents import Car, Pedestrian, RectangleBuilding
from entities import Entity, RectangleEntity, CircleEntity, RingEntity
from world import World
import numpy as np
import copy


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
	def __init__(self, car, world, friction=0.06, alpha=0.1, gamma=0.9, epsilon=0.1, episode=1000, resolution=0.2):
		self.car = copy.copy(car)
		self.world = copy.copy(world)
		self.car_x_init = self.car.center.x
		self.car_y_init = self.car.center.y
		self.car_heading_init = self.car.heading
		self.radius = np.sqrt(
			(self.car.center.x - self.world.width / 2) ** 2 + (self.car.center.y - self.world.height / 2) ** 2)
		self.friction = friction
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.episode = episode
		self.resolution = resolution
		self.action_theta = np.array(
			(-np.pi / 2, -np.pi / 3, -np.pi / 4, -np.pi / 6, 0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2))

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
			print("Training epoch:", n_episode)
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
					(car.center.x - self.world.width / 2) ** 2 + (car.center.y - self.world.height / 2) ** 2)
				reward -= 10 * abs(new_radius - self.radius)

				# update Q value
				max_q = max(self.Q[x_new, y_new, :])
				self.Q[x, y, action] += self.alpha * (reward + self.gamma * max_q - self.Q[x, y, action])

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

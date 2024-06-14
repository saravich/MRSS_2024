import numpy as np
import scipy.stats
from numpy.random import uniform
import cv2
from random import random


class ParticleFilter:

    def __init__(self, window_width=800, window_height=600, name_of_window='Particle Filter'):

        self.window_width = window_width
        self.window_height = window_height
        self.name_of_window = name_of_window
        self.center = np.array([[0, 0]])

        self.mouseX = 0
        self.mouseY = 0
        # self.prev_x = -1
        # self.prev_y = -1

        self.x_range = np.array([0, window_width])
        self.y_range = np.array([0, window_height])

        self.particles_num = 400
        self.particles = np.empty((self.particles_num, 3))  # [x, y, theta]
        self.create_uniform_particles()

        self.landmarks = []
        self.weights = np.ones(self.particles_num) / self.particles_num
        self.seen_landmarks = np.array([])
        self.landmarks_dict = {'id': [1, 2, 3, 4, 5, 6],
                               'pose': np.array([[0, 117], [90, 0], [261, 0], [351, 117], [261, 235], [90, 235]]),
                               'seen': [True, False, False, False, False, False]}

    # def integrate_movement(self, u, dt=1):

    def main(self):

        img = np.zeros((self.window_height, self.window_width, 3), np.uint8)

        cv2.namedWindow(self.name_of_window)
        DELAY_MSEC = 50

        while True:

            pf.particle_filter_update(heading_vel=0.3, angular_vel=0.0, measured_distances=[100], measured_ids=[1])

            img = np.zeros((self.window_height, self.window_width, 3), np.uint8)

            for landmark in self.landmarks:
                # cv2.line(img, (int(self.mouseX), int(self.mouseY)), (int(landmark[0]), int(landmark[1])), (255, 255, 0), 2)
                cv2.circle(img, (int(landmark[0]), int(landmark[1])), 20, (0, 0, 255), -1)

            # drawing particles
            for particle in self.particles:
                cv2.circle(img, (int(particle[0]), int(particle[1])), 1, (255, 255, 255), -1)

            if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
                break

            best_particles_idx = self.weights.argsort()[:int(0.4 * len(self.particles))]
            best_particle = self.particle_distance(self.particles[best_particles_idx])
            self.mouseX, self.mouseY = best_particle[0], best_particle[1]

            # Draw a line from mouse position to each landmark

            # Draw a circle at each landmark
            # for i, landmark in enumerate(self.landmarks):
            #     color = (0, 255, 255) if self.landmarks_dict['seen'][i] else (255, 0, 0)
            #     cv2.circle(img, (int(landmark[0]), int(landmark[1])), 20, color, -1)

            # draw grid lines ###############
            x = 0
            y = 0
            while x < img.shape[1]:
                cv2.line(img, (x, 0), (x, img.shape[0]), color=(155, 100, 255), thickness=1)
                x += 50

            while y < img.shape[0]:
                cv2.line(img, (0, y), (img.shape[1], y), color=(155, 100, 255), thickness=1)
                y += 50
            ###############################
            cv2.putText(img, "Mouse Position: ({}, {})".format(self.mouseX, self.mouseY), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)

            cv2.imshow(self.name_of_window, img)

            print("Estimated Position: {}, {}".format(self.mouseX, self.mouseY))

        cv2.destroyAllWindows()

    def particle_filter_update(self, heading_vel, angular_vel, measured_distances, measured_ids, dt=0.1):

        # Get new mouse position
        # self.mouseX, self.mouseY = x, y

        # mouse position noise is held in "center"
        # we add noise to the mouse position which represents the motion model of the agent
        # self.center = np.array(
        #     [[int(np.random.uniform(low=0.9, high=1.1) * x + 0.2), int(np.random.uniform(low=0.9, high=1.1) * y)]])
        #
        # self.center =
        # direction = np.arctan2(np.array([y - self.prev_y]), np.array([self.prev_x - x]))
        direction = angular_vel * dt
        direction = -direction + np.pi if direction > 0 else -np.pi - direction

        # distance = np.linalg.norm(np.array([[self.prev_x, self.prev_y]]) - np.array([[x, y]]), axis=1)
        distance = heading_vel * dt
        # mouse movement noise is implemented in "distance" which is to be included in "u"
        # velocity of the mouse which is considered to be a vector composed of "distance" and "direction"

        u = np.array([direction, distance])

        # predicting particles position with respect to the last movement of the mouse
        self.predict(u)

        self.landmark_detected(measured_ids)
        self.update(z=measured_distances, R=50)

        indexes = self.systematic_resample()
        self.resample_from_index(indexes)

    def particle_distance(self, particles):
        dist = np.zeros(len(particles))
        for i, particle in enumerate(particles):
            dist[i] = sum(np.sqrt((particle[0] - particles[:, 0]) ** 2 + (particle[1] - particles[:, 1]) ** 2))
        return particles[np.argmin(dist)]

    def create_uniform_particles(self):
        # self.particles[:, 0] = uniform(self.x_range[0], self.x_range[1], size=len(self.particles))
        # self.particles[:, 1] = uniform(self.y_range[0], self.y_range[1], size=len(self.particles))
        # print(len(self.particles))
        self.particles[:, 0] = uniform(self.x_range[0], self.x_range[1], size=len(self.particles))
        self.particles[:, 1] = uniform(self.y_range[0], self.y_range[1], size=len(self.particles))
        self.particles[:, 2] = uniform(-np.pi, np.pi, size=self.particles_num)

    # def predict(self, d, dtheta, dt=1):
    #     self.particles[:, 0] += np.cos(u[0]) * ((u[1] * dt + np.random.normal(0.1, 5, self.particles_num)))
    #     self.particles[:, 1] += np.sin(u[0]) * ((u[1] * dt + np.random.normal(0.1, 5, self.particles_num)))
    #
    #     self.particles[:, 0] += (d * np.cos(yaw_setpoint) + np.random.normal(self.translation_model[0], self.translation_model[1],
    #                                                   self.num_of_particles))
    #     self.particles[:, 1] += (d * np.sin(yaw_setpoint)+ np.random.normal(self.translation_model[0], self.translation_model[1],
    #                                                   self.num_of_particles))

    def predict(self, u, dt=1):
        direction, distance = u
        std_dev = 0.2
        self.particles[:, 0] += distance * np.cos(direction) + np.random.normal(0, std_dev, self.particles_num)
        self.particles[:, 1] += distance * np.sin(direction) + np.random.normal(0, std_dev, self.particles_num)
        self.particles[:, 2] += direction + np.random.normal(0, std_dev, self.particles_num)

        # Ensure particles stay within bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.window_width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.window_height)
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update(self, z, R):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            if self.landmarks_dict['seen'][i]:
                distance = np.power(
                    (self.particles[:, 0] - landmark[0]) ** 2 + (self.particles[:, 1] - landmark[1]) ** 2, 0.5)
                R = distance * 0.5
                self.weights *= scipy.stats.norm(distance, R).pdf(z[i])

        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)

    def systematic_resample(self):
        N = len(self.weights)
        positions = (np.arange(N) + np.random.random()) / N

        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < N and j < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        return indexes

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    def landmark_detected(self, landmark_ids):
        self.landmarks = []
        for i in range(len(landmark_ids)):
            # self.landmarks.append(self.landmarks_dict['pose'][landmark_ids[i]])
            index = self.landmarks_dict['id'].index(landmark_ids[i])
            # Retrieve the pose using the index
            pose = self.landmarks_dict['pose'][index]
            self.landmarks.append(pose)


landmarks = [[0, 117], [90, 0], [261, 0], [351, 117], [261, 235], [90, 235]]

pf = ParticleFilter(window_width=351, window_height=234, name_of_window="Particle Filter")
pf.main()
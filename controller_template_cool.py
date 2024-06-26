"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time

import numpy as np
import cv2
import pyrealsense2 as rs
from time import sleep

from matplotlib import pyplot as plt
from numpy.random import uniform
import scipy

CONNECT_SERVER = False  # False for local tests, True for deployment

# ----------- DO NOT CHANGE THIS PART -----------

# The deploy.py script runs on the Jetson Nano at IP 192.168.123.14
# and listens on port 9292
# whereas this script runs on one of the two other Go1's Jetson Nano

SERVER_IP = "192.168.123.14"
SERVER_PORT = 9292

# Maximum duration of the task (seconds):
TIMEOUT = 180

# Minimum control loop duration:
MIN_LOOP_DURATION = 0.1


# Use this function to send commands to the robot:
def send(sock, x, y, r):
    """
    Send a command to the robot.

    :param sock: TCP socket
    :param x: forward velocity (between -1 and 1)
    :param y: side velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    data = struct.pack('<hfff', code, x, y, r)
    if sock is not None:
        sock.sendall(data)


# ---------- HELPING FUNCTIONS

MIN_OBS_DISTANCE = 0.5  # meters


def spin(s, speed):
    """
    Spin the robot in place.

    :param s: TCP socket
    :param speed: angular velocity (between -1 and 1)
    """
    send(s, 0., 0., speed)


def go_back(s, a=-1):
    """
    Move the robot backwards.

    :param s: TCP socket
    :param a: backward velocity (between -1 and 1)
    """
    # clip the value to the range [-1, 1]
    a = np.clip(a, -1, 0)
    send(s, a, 0., 0.)


def move(s, x, r):
    """
    Move the robot forward.

    :param x: forward velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    send(s, x, 0., r)


# ------ FRONTEND CLASS

class Fronted():
    """
    This class hanlde all the RGB and depth data
    """
    map_landmarks = {
        "1": [-0.58, 0.],
        "2": [0.32, 1.175],
        "3": [2.03, 1.175],
        "4": [2.93, 0.],
        "5": [2.03, -1.175],
        "6": [0.32, -1.175]
    }  # in meters

    def __init__(self):
        # Fisheye camera (distortion_model: narrow_stereo):
        self.map_augmenter()

        self.image_width = 640
        self.image_height = 480

        # --------- DO NOT CHANGE THIS PART ---------

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

        self.arucoParams = cv2.aruco.DetectorParameters_create()
        self.arucoParams.markerBorderBits = 1

        # --------- CHANGE THIS PART (optional) ---------

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        device = self.pipeline_profile.get_device()

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("Could not find a depth camera with color sensor")
            exit(0)

        # Depht available FPS: up to 90Hz
        self.config.enable_stream(rs.stream.depth, self.image_width, self.image_height, rs.format.z16, 30)
        # RGB available FPS: 30Hz
        self.config.enable_stream(rs.stream.color, self.image_width, self.image_height, rs.format.bgr8, 30)
        # # Accelerometer available FPS: {63, 250}Hz
        # config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        # # Gyroscope available FPS: {200,400}Hz
        # config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 2.5  # 3 meter
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        self.depth_frame = None
        self.color_frame = None

        self.normalized_depth_directions = []

    def map_augmenter(self):
        # augment the apriltag landmarks with the pseudo landmarks
        apriltag_size_half = 0.075
        # augment the map with pseudo landmarks that are 10 cm away from the real landmarks
        self.map_landmarks["1_1"] = [self.map_landmarks["1"][0], self.map_landmarks["1"][0] + apriltag_size_half]
        self.map_landmarks["1_2"] = [self.map_landmarks["1"][0], self.map_landmarks["1"][0] - apriltag_size_half]

        self.map_landmarks["2_1"] = [self.map_landmarks["2"][0] + apriltag_size_half, self.map_landmarks["2"][0]]
        self.map_landmarks["2_2"] = [self.map_landmarks["2"][0] - apriltag_size_half, self.map_landmarks["2"][0]]

        self.map_landmarks["3_1"] = [self.map_landmarks["3"][0] + apriltag_size_half, self.map_landmarks["3"][0]]
        self.map_landmarks["3_2"] = [self.map_landmarks["3"][0] - apriltag_size_half, self.map_landmarks["3"][0]]

        self.map_landmarks["4_1"] = [self.map_landmarks["4"][0], self.map_landmarks["4"][0] + apriltag_size_half]
        self.map_landmarks["4_2"] = [self.map_landmarks["4"][0], self.map_landmarks["4"][0] - apriltag_size_half]

        self.map_landmarks["5_1"] = [self.map_landmarks["5"][0] + apriltag_size_half, self.map_landmarks["5"][0]]
        self.map_landmarks["5_2"] = [self.map_landmarks["5"][0] - apriltag_size_half, self.map_landmarks["5"][0]]

        self.map_landmarks["6_1"] = [self.map_landmarks["6"][0] + apriltag_size_half, self.map_landmarks["6"][0]]
        self.map_landmarks["6_2"] = [self.map_landmarks["6"][0] - apriltag_size_half, self.map_landmarks["6"][0]]

    def smoother(self, data, window_size):
        """ Mean smoothing """
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def meters2scale(self, meters) -> float:
        return meters / self.depth_scale

    def read(self) -> bool:
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # read the frames
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not self.depth_frame or not self.color_frame:
            return False
        return True

    def run(self, history=None):
        # read data
        if self.read() is False:
            return

        # extract safe direction
        error_direction, average_depth, depth_strip = self.extract_direction_and_depth()

        # extract april tags
        detected_corners, detected_ids, _ = self.get_april_tags()

        measurements = self.landmarks2map(detected_ids, detected_corners, depth_strip)

        return {
            "error_direction": error_direction,
            "average_depth": average_depth,
            "measurements": measurements
        }

    def extract_direction_and_depth(self) -> (float, float):
        # Convert images to numpy arrays
        depth_image = np.asanyarray(self.depth_frame.get_data())

        # compute the average depth
        average_depth = np.mean(depth_image[280:, :].copy())

        cv2.imshow('depth', depth_image.copy())
        cv2.waitKey(1)
        # remove the measurement farther than clipping_distance
        depth_image = np.clip(depth_image, 0, self.clipping_distance)

        # crop an horizontal strip of the depth image image_height = 480
        depth_image_bottom_crop = depth_image[280:, :]



        # compute the median value of the cropped depth image
        depth_bottom_crop_median = np.median(depth_image_bottom_crop)

        # squeeze the depth image to a 1D array averaging the values of the horizontal strip
        # in this way we do not care much about artifacts in the depth image
        depth_strip = np.mean(depth_image_bottom_crop, axis=0)
        depth_strip_clone = depth_strip.copy()

        # threshold the depth image according to the meadian value
        depth_strip[depth_strip < depth_bottom_crop_median] = 0.
        depth_strip[depth_strip >= depth_bottom_crop_median] = 1.

        # find the segments of contiguous ones
        segments = self.find_segments(depth_strip)

        # find the segment with the maximum length
        if segments:
            segment = max(segments, key=self.segment_length)
            depth_center = self.midpoint(segment)
            # center the depth_center in the depth image
            depth_center -= self.image_width / 2.
            # normalize the depth_center
            depth_center /= ((self.image_width / 2.) / 2.)

            self.normalized_depth_directions.append(depth_center)

            if len(self.normalized_depth_directions) > 5:
                depth_center = self.smoother(self.normalized_depth_directions, 5)[0]
                self.normalized_depth_directions.pop(0)

                return depth_center, average_depth, depth_strip_clone
            return None, average_depth, depth_strip_clone
        else:
            if len(self.normalized_depth_directions) > 5:
                return 0., average_depth
            return None, average_depth, depth_strip_clone

        if np.isnan(depth_center):
            return 0., average_depth, depth_strip_clone

    def get_april_tags(self) -> (np.array, np.array, np.array):
        # Convert images to numpy arrays
        color_image = np.asanyarray(self.color_frame.get_data())

        # Markersangular_output_cmd detection:
        grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(grey_frame, self.aruco_dict,
                                                                             parameters=self.arucoParams)
        cv2.imshow('frame', cv2.aruco.drawDetectedMarkers(color_image, detected_corners, detected_ids))
        cv2.waitKey(1)
        return detected_corners, detected_ids, rejected

    def find_segments(self, arr):
        """Function to find segments of contiguous ones"""
        segments = []
        start = None
        for i, val in enumerate(arr):
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:
            segments.append((start, len(arr) - 1))
        return segments

    def segment_length(self, segment):
        """Function to calculate the length of a segment"""
        return segment[1] - segment[0] + 1

    def midpoint(self, segment):
        """Function to calculate the midpoint of a segment"""
        return (segment[0] + segment[1]) / 2

    def landmarks2map(self, detected_ids, detected_corners, depth_strip) -> dict:
        angular_output_cmd = 0.0
        linear_vel = 0
        measurements = {
            "1": None,
            "2": None,
            "3": None,
            "4": None,
            "5": None,
            "6": None,
            "1_1": None,
            "2_1": None,
            "3_1": None,
            "4_1": None,
            "5_1": None,
            "6_1": None,
            "1_2": None,
            "2_2": None,
            "3_2": None,
            "4_2": None,
            "5_2": None,
            "6_2": None
        }

        if detected_ids is None:
            return measurements
        for i, id_ in enumerate(detected_ids):
            id_ = str(id_[0])
            if not id_ in self.map_landmarks:
                continue
            # on the depth strip get the x coordinates of the corners and the center of the landmark
            landmark_center = detected_corners[i][0][0] + (detected_corners[i][0][2] - detected_corners[i][0][0]) / 2.
            landmark_center = landmark_center.astype(int)
            x = landmark_center[0]
            measurements[id_] = depth_strip[x]

            # ugmented
            x_1 = detected_corners[i][0][2][0].astype(int)
            measurements[id_ + "_1"] = depth_strip[x_1]

            x_2 = detected_corners[i][0][0][0].astype(int)
            measurements[id_ + "_2"] = depth_strip[x_2]
        return measurements

    def stop(self):
        self.pipeline.stop()

# ------ PID CLASS
# https://github.com/m-lundberg/simple-pid/

def _clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif (upper is not None) and (value > upper):
        return upper
    elif (lower is not None) and (value < lower):
        return lower
    return value

class PID(object):
    """A simple PID controller."""

    def __init__(
            self,
            Kp=1.0,
            Ki=0.0,
            Kd=0.0,
            setpoint=0,
            sample_time=0.01,
            output_limits=(None, None),
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=True,
            error_map=None,
            time_fn=None,
            starting_output=0.0,
    ):
        """
        Initialize a new PID controller.

        :param Kp: The value for the proportional gain Kp
        :param Ki: The value for the integral gain Ki
        :param Kd: The value for the derivative gain Kd
        :param setpoint: The initial setpoint that the PID will try to achieve
        :param sample_time: The time in seconds which the controller should wait before generating
            a new output value. The PID works best when it is constantly called (eg. during a
            loop), but with a sample time set so that the time difference between each update is
            (close to) constant. If set to None, the PID will compute a new output value every time
            it is called.
        :param output_limits: The initial output limits to use, given as an iterable with 2
            elements, for example: (lower, upper). The output will never go below the lower limit
            or above the upper limit. Either of the limits can also be set to None to have no limit
            in that direction. Setting output limits also avoids integral windup, since the
            integral term will never be allowed to grow outside of the limits.
        :param auto_mode: Whether the controller should be enabled (auto mode) or not (manual mode)
        :param proportional_on_measurement: Whether the proportional term should be calculated on
            the input directly rather than on the error (which is the traditional way). Using
            proportional-on-measurement avoids overshoot for some types of systems.
        :param differential_on_measurement: Whether the differential term should be calculated on
            the input directly rather than on the error (which is the traditional way).
        :param error_map: Function to transform the error value in another constrained value.
        :param time_fn: The function to use for getting the current time, or None to use the
            default. This should be a function taking no arguments and returning a number
            representing the current time. The default is to use time.monotonic() if available,
            otherwise time.time().
        :param starting_output: The starting point for the PID's output. If you start controlling
            a system that is already at the setpoint, you can set this to your best guess at what
            output the PID should give when first calling it to avoid the PID outputting zero and
            moving the system away from the setpoint.
        """
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            # Use the user supplied time function
            self.time_fn = time_fn
        else:
            import time

            try:
                # Get monotonic time to ensure that time deltas are always positive
                self.time_fn = time.monotonic
            except AttributeError:
                # time.monotonic() not available (using python < 3.3), fallback to time.time()
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

        # Set initial state of the controller
        self._integral = _clamp(starting_output, output_limits)

    def __call__(self, input_, dt=None):
        """
        Update the PID controller.

        Call the PID controller with *input_* and calculate and return a control output if
        sample_time seconds has passed since the last update. If no new output is calculated,
        return the previous output instead (or None if no value has been calculated yet).

        :param dt: If set, uses this value for timestep instead of real time. This can be used in
            simulations when simulation time is different from real time.
        """
        if not self.auto_mode:
            return self._last_output

        now = self.time_fn()
        if dt is None:
            dt = now - self._last_time if (now - self._last_time) else 1e-16
        elif dt <= 0:
            raise ValueError('dt has negative value {}, must be positive'.format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # Only update every sample_time seconds
            return self._last_output

        # Compute error terms
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
        d_error = error - (self._last_error if (self._last_error is not None) else error)

        # Check if must map the error
        if self.error_map is not None:
            error = self.error_map(error)

        # Compute the proportional term
        if not self.proportional_on_measurement:
            # Regular proportional-on-error, simply set the proportional term
            self._proportional = self.Kp * error
        else:
            # Add the proportional error on measurement to error_sum
            self._proportional -= self.Kp * d_input

        # Compute integral and derivative terms
        self._integral += self.Ki * error * dt
        self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup

        if self.differential_on_measurement:
            self._derivative = -self.Kd * d_input / dt
        else:
            self._derivative = self.Kd * d_error / dt

        # Compute final output
        output = self._proportional + self._integral + self._derivative
        output = _clamp(output, self.output_limits)

        # Keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_error = error
        self._last_time = now

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'Kp={self.Kp!r}, Ki={self.Ki!r}, Kd={self.Kd!r}, '
            'setpoint={self.setpoint!r}, sample_time={self.sample_time!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
        """
        The P-, I- and D-terms from the last computation as separate components as a tuple. Useful
        for visualizing what the controller is doing or when tuning hard-to-tune systems.
        """
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Ki, Kd)."""
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """Set the PID tunings."""
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        """
        Enable or disable the PID controller, optionally setting the last output value.

        This is useful if some system has been manually controlled and if the PID should take over.
        In that case, disable the PID by setting auto mode to False and later when the PID should
        be turned back on, pass the last output variable (the control variable) and it will be set
        as the starting I-term when the PID is set to auto mode.

        :param enabled: Whether auto mode should be enabled, True or False
        :param last_output: The last output, or the control variable, that the PID should start
            from when going from manual mode to auto mode. Has no effect if the PID is already in
            auto mode.
        """
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

            self._integral = last_output if (last_output is not None) else 0
            self._integral = _clamp(self._integral, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if (None not in limits) and (max_output < min_output):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self._integral = _clamp(self._integral, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        self._last_input = None

# ------ CONTROLLER CLASS

class Controller:

    def __init__(self):
        # define the PID controller
        self.controller_theta = PID(
            Kp=1.0, Ki=0.05, Kd=0.05,
            setpoint=0,  # image_width / 2,
            sample_time=MIN_LOOP_DURATION,
            output_limits=[-1, 1],
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=False,
            error_map=None,
            time_fn=None,
            starting_output=0
        )

        self.controller_vel = PID(
            Kp=1.0, Ki=0.01, Kd=0.08,
            setpoint=-0.5,
            sample_time=MIN_LOOP_DURATION,
            output_limits=[-1, 1],
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=False,
            error_map=None,
            time_fn=None,
            starting_output=0
        )

    def backward(self, s, error_x, error_theta, dt):
        self.controller_vel.setpoint = 0.7
        u_v = self.controller_vel(error_x, dt)
        u_theta = self.controller_theta(error_theta, dt)
        send(s, u_v, 0., u_theta)

    def forward(self, s, error_x, error_theta, dt):
        self.controller_vel.setpoint = -0.5
        u_v = self.controller_vel(-error_x, dt)
        u_theta = self.controller_theta(error_theta, dt)
        send(s, u_v, 0., u_theta)

    def setpoints(self, set_x=None, set_theta=None):
        if set_x is not None:
            self.controller_vel.setpoint = set_x
        if set_theta is not None:
            self.controller_theta.setpoint = set_theta

    def happy_moves(self, s):
        send(s, 0., 0., -0.7)
        sleep(2.)
        send(s, 0., 0., 0.7)
        sleep(2.)
        send(s, 0., 0., -0.7)
        sleep(2.)
        send(s, 0., 0., 0.7)
        sleep(2.)
        send(s, 0., 0., -0.7)
        sleep(2.)
        send(s, 0., 0., 0.)

    def run(self, error_x, error_theta, dt):
        # self.controller_theta.setpoint = error_theta
        u_theta = self.controller_theta(error_theta, dt)
        u_v = self.controller_vel(error_x, dt)
        return u_v, u_theta

# ------- STATE ESTIMATOR
class ParticleFilter:

    def __init__(self, x_range, y_range, map_landmarks):

        self.mouseX = 0
        self.mouseY = 0
        # self.prev_x = -1
        # self.prev_y = -1

        self.x_range = np.array(x_range)
        self.y_range = np.array(y_range)

        self.particles_num = 5000
        self.particles = np.empty((self.particles_num, 3))  # [x, y, theta]
        self.create_uniform_particles()

        self.landmarks = []
        self.weights = np.ones(self.particles_num) / self.particles_num
        self.seen_landmarks = np.array([])
        self.landmarks_dict = {'id': [1, 11, 12, 2, 21, 22, 3, 31, 32, 4, 41, 42, 5, 51, 52, 6, 61, 62],
                               'pose': np.array([[-58, 0], map_landmarks["1_1"], map_landmarks["1_2"],
                                                 [32, 117.5], map_landmarks["2_1"], map_landmarks["2_2"],
                                                 [203, 117.5], map_landmarks["3_1"], map_landmarks["3_2"],
                                                 [293, 0], map_landmarks["4_1"], map_landmarks["4_2"],
                                                 [203, -117.5], map_landmarks["5_1"], map_landmarks["5_2"],
                                                 [32, -117.5], map_landmarks["6_1"], map_landmarks["6_2"]]),
                               }

        tr_model_mean = 0.000
        tr_model_var = 0.01
        tr_model = [tr_model_mean, tr_model_var]

        rot_model_mean = 0.000
        rot_model_var = 0.01
        rot_model = [rot_model_mean, rot_model_var]
        self.rotation_model = rot_model
        self.translation_model = tr_model

    def particle_filter_update(self, heading_vel, angular_vel, measurements, dt=0.1):

        measured_distances, measured_ids = list(measurements.values()), list(measurements.keys())

        # measured_distances where not none
        mask = np.array([d is not None for d in measured_distances])
        measured_distances = np.array(measured_distances)[mask]
        measured_ids = np.array(measured_ids)[mask]

        # print(measured_distances, measured_ids)
        direction = -angular_vel * dt
        direction = -direction + np.pi if direction > 0 else -np.pi - direction

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

        best_particles_idx = self.weights.argsort()[:int(0.4 * len(self.particles))]
        best_particle = self.particle_distance(self.particles[best_particles_idx])
        self.X, self.Y, self.Theta = best_particle[0], best_particle[1], best_particle[2]
        # from mm to m
        self.X /= 100
        self.Y /= 100

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

    def predict(self, u, dt=1, yaw_setpoint=np.pi/2):

        direction, distance = u
        std_dev = 0.2
        self.particles[:, 0] += distance * np.cos(direction) + np.random.normal(0, std_dev, self.particles_num)
        self.particles[:, 1] += distance * np.sin(direction) + np.random.normal(0, std_dev, self.particles_num)
        self.particles[:, 2] += direction + np.random.normal(0, std_dev, self.particles_num)


        # # self.particles[:, 0] += distance * np.cos(direction) + np.random.normal(0, std_dev, self.particles_num)
        # # self.particles[:, 1] += distance * np.sin(direction) + np.random.normal(0, std_dev, self.particles_num)
        # # self.particles[:, 2] += direction + np.random.normal(0, std_dev, self.particles_num)
        #
        # self.particles[:, 0] += (distance * np.cos(yaw_setpoint) + np.random.normal(self.translation_model[0],
        #                                                                             self.translation_model[1],
        #                                                                             self.particles_num))
        # self.particles[:, 1] += (distance * np.sin(yaw_setpoint) + np.random.normal(self.translation_model[0],
        #                                                                             self.translation_model[1],
        #                                                                             self.particles_num))
        # self.particles[:, 2] += direction + np.random.normal(self.rotation_model[0], self.rotation_model[1],
        #                                                      self.particles_num)

        # Ensure particles stay within bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], self.x_range[0], self.x_range[1])
        self.particles[:, 1] = np.clip(self.particles[:, 1], self.y_range[0], self.y_range[1])
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi

    def update(self, z, R):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            distance = np.power(
                (self.particles[:, 0] - landmark[0]) ** 2 + (self.particles[:, 1] - landmark[1]) ** 2, 0.5)
            # R = distance * 0.5
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

    ids_dict = {
        '1': 1,
        '1_1': 11,
        '1_2': 12,
        '2': 2,
        '2_1': 21,
        '2_2': 22,
        '3': 3,
        '3_1': 31,
        '3_2': 32,
        '4': 4,
        '4_1': 41,
        '4_2': 42,
        '5': 5,
        '5_1': 51,
        '5_2': 52,
        '6': 6,
        '6_1': 61,
        '6_2': 62
    }

    def id2str(self, id_str):
        return self.ids_dict[id_str]

    def landmark_detected(self, landmark_ids):
        self.landmarks = []
        for i in range(len(landmark_ids)):
            # self.landmarks.append(self.landmarks_dict['pose'][landmark_ids[i]])
            index = self.landmarks_dict['id'].index(self.id2str(landmark_ids[i]))
            # Retrieve the pose using the index
            pose = self.landmarks_dict['pose'][index]
            self.landmarks.append(pose)

# ------ PLANNER CLASS

class Planner:

    def __init__(self,
                 reso=0.02,
                 robot_radius=0.4,
                 max_p=10.,
                 KP=5.0,
                 ETA=100.,
                 AREA_WIDTH=3.51,
                 AREA_HEIGHT=2.34
                 ):
        """
        initialize the planner

        params:
            - reso: float, the resolution of the grid for the potential field.  Default = 0.01
            - robot_radius: float, radius of the robot in meters, control size boundaries around obstacles.  Default=0.4
            - max_p: float, scaling factor that converts potential to speed.  Default=10
            - KP: float, attractive potential gain.  Default=5.0
            - ETA: float, repulsive potential gain.  Dfault=100.
            - AREA_WIDTH:, float, width of map. Default=3.51
            - AREA_HEIGHT: float, height of map.  Default=2.34
        """

        self.KP = KP  # attractive potential gain
        self.ETA = ETA  # repulsive potential gain
        self.AREA_WIDTH = AREA_WIDTH  # potential area width [m] 350 w 234 [h]
        self.AREA_HEIGHT = AREA_HEIGHT
        # the number of previous positions used to check oscillations
        self.OSCILLATIONS_DETECTION_LENGTH = 3

        self.reso = reso
        self.rr = robot_radius
        self.max_p = max_p

        self.previous_ids = []

        self.ox = []
        self.oy = []

        self.gx = None
        self.gy = None

        self.set_goal(0., 117.5/100)

        # generate the potential field
        self.potential_field_planning()

    def run(self, x, y):  # -> (float, float):

        if self.oscillations_detection(s, y):
            return None, None

        p, x, y, theta = self.get_potential(x, y)

        # was getting
        p = 1 if p > 1 else p
        p = -1 if p < -1 else p
        return p, theta

    def new_obstacle(self, ox, oy):

        self.set_obstacle(ox, oy)

        return self.potential_field_planning()

    def set_goal(self, gx, gy):
        "Setter for the goal position"
        self.gx = gx
        self.gy = gy

    def set_obstacle(self, ox, oy):
        self.ox.append(ox)
        self.oy.append(oy)

    def calc_potential_field(self, gx, gy, ox, oy, rr):

        minx = 0  # min(min(ox), sx, gx) - self.AREA_WIDTH / 2.0
        miny = 0  # min(min(oy), sy, gy) - self.AREA_WIDTH / 2.0
        maxx = self.AREA_WIDTH  # max(max(ox), sx, gx) + self.AREA_WIDTH / 2.0
        maxy = self.AREA_HEIGHT  # max(max(oy), sy, gy) + self.AREA_WIDTH / 2.0
        xw = int(round((maxx - minx) / self.reso))
        yw = int(round((maxy - miny) / self.reso))

        # calc each potential
        pmap = [[0.0 for i in range(yw)] for i in range(xw)]

        for ix in range(xw):
            x = ix * self.reso + minx

            for iy in range(yw):
                y = iy * self.reso + miny
                ug = self.calc_attractive_potential(x, y, gx, gy)

                uo = self.calc_repulsive_potential(x, y, ox, oy, rr) if ox != [] else 0
                uf = ug + uo
                pmap[ix][iy] = uf

        return pmap, minx, miny

    def calc_attractive_potential(self, x, y, gx, gy):
        return 0.5 * self.KP * np.hypot(x - gx, y - gy)

    def calc_repulsive_potential(self, x, y, ox, oy, rr):
        # search nearest obstacledq
        minid = -1
        dmin = float("inf")
        for i, _ in enumerate(ox):
            d = np.hypot(x - ox[i], y - oy[i])
            if dmin >= d:
                dmin = d
                minid = i

        # calc repulsive potential
        # print(ox, oy)
        dq = np.hypot(x - ox[minid], y - oy[minid])

        if dq <= rr:
            if dq <= 0.1:
                dq = 0.1

            return 0.5 * self.ETA * (1.0 / dq - 1.0 / rr) ** 2
        else:
            return 0.0

    def get_motion_model(self):
        # dx, dy
        motion = [[1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1],
                  [-1, -1],
                  [-1, 1],
                  [1, -1],
                  [1, 1]]

        self.theta_options = [0., np.pi / 2., np.pi, -np.pi / 2., -3 * np.pi / 4., 3 * np.pi / 4., -1 * np.pi / 4.,
                              np.pi / 4]

        return motion

    def oscillations_detection(self, ix, iy):
        self.previous_ids.append((ix, iy))

        # print(previous_ids)

        if (len(self.previous_ids) > self.OSCILLATIONS_DETECTION_LENGTH):
            self.previous_ids.pop(0)

        # check if contains any duplicates by copying into a set
        previous_ids_set = set()
        for index in self.previous_ids:
            if index in previous_ids_set:
                return True
            else:
                previous_ids_set.add(index)
        return False

    def potential_field_planning(self):

        # calc potential field
        self.pmap, self.minx, self.miny = self.calc_potential_field(self.gx, self.gy, self.ox, self.oy,
                                                                    self.rr)

        return self.pmap, self.minx, self.miny

    def get_potential(self, x, y):

        ix = int(round(x / self.reso))
        iy = int(round(y / self.reso))

        minp = float("inf")
        minix, miniy, mini = -1, -1, 0

        motion = self.get_motion_model()

        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(self.pmap) or iny >= len(self.pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                # print("outside potential!")
            else:
                p = self.pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
                mini = i

        x = minix * self.reso
        y = miniy * self.reso

        theta = self.theta_options[mini]
        return p / self.max_p, x, y, theta

    def get_potential_index(self, ix, iy):

        minp = float("inf")
        minix, miniy = -1, -1

        motion = self.get_motion_model()

        for i, _ in enumerate(motion):
            inx = int(ix + motion[i][0])
            iny = int(iy + motion[i][1])
            if inx >= len(self.pmap) or iny >= len(self.pmap[0]) or inx < 0 or iny < 0:
                p = float("inf")  # outside area
                # print("outside potential!")
            else:
                p = self.pmap[inx][iny]
            if minp > p:
                minp = p
                minix = inx
                miniy = iny
        return p, minix, miniy

# ------ STATE MACHINE

class StateMachine:
    states = {
        'STOP': -1,
        'EXPLORE': 0,
        'LOCALIZE': 1,
        'PLAN': 2,
        'MOVE': 3,
    }
    states2str = {
        -1: 'STOP',
        0: 'EXPLORE',
        1: 'LOCALIZE',
        2: 'PLAN',
        3: 'MOVE',
    }

    def __init__(self):
        self.state = 0

    def state_transition(self):
        if self.state < 3:
            self.state += 1
        else:
            self.state = -1
        self.print_state()

    def print_state(self):
        print(f"Current state: {self.states2str[self.state]}")

    def reset(self):
        self.state = -1

# ----------- DO NOT CHANGE THIS PART -----------

RECORD = False
history = []

# ----------------- CONTROLLER -----------------

frontend = Fronted()
# state_estimator = StateEstimator()
state_estimator = ParticleFilter([-58, 293], [-117.5, 117.5], map_landmarks=frontend.map_landmarks)

planner = Planner()
controller = Controller()

state_machine = StateMachine()

try:
    # We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

    print("Client connecting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        if CONNECT_SERVER:
            s.connect((SERVER_IP, SERVER_PORT))
            print("Connected.")
        else:
            s = None

        code = 1  # 1 for velocity commands

        task_complete = False
        start_time = time.time()
        previous_time_stamp = start_time

        min_obs_distance = frontend.meters2scale(MIN_OBS_DISTANCE)
        vel, theta = 0, 0
        # main control loop:
        while not task_complete and not time.time() - start_time > TIMEOUT:
            error_theta = 0.
            error_x = 0.
            # avoid busy loops:
            now = time.time()
            dt = now - previous_time_stamp
            if dt < MIN_LOOP_DURATION:
                time.sleep(MIN_LOOP_DURATION - dt)

            res_frontend = frontend.run()

            # print(res_frontend)

            # stop and go back if too close with an obstacle
            if res_frontend['average_depth'] < min_obs_distance:
                controller.backward(s, res_frontend['average_depth'], 0, dt)
                continue

            if res_frontend['error_direction'] is not None:
                error_theta = res_frontend['error_direction']

            if state_machine.state == StateMachine.states['EXPLORE']:
                if not any(res_frontend['measurements'].values()):
                    state_machine.state_transition()
                # we cannot localize properly, then explore
                controller.forward(s, res_frontend['average_depth'], error_theta, dt)
                continue

            if state_machine.state == StateMachine.states['LOCALIZE']:
                ######### PASS res_frontend['measurements'] to state estimation
                # localize the robot
                if len(list(res_frontend['measurements'].values()))>=0 or all([value is not None for value in res_frontend['measurements'].values()])==False:
                    state_estimator.particle_filter_update(heading_vel=vel,
                                                           angular_vel=theta,
                                                           measurements=res_frontend['measurements'],
                                                           dt=dt)
                print("Estimated Pose: {}, {}, {}".format(state_estimator.X, state_estimator.Y, state_estimator.Theta))
                state_machine.state_transition()

            if state_machine.state == StateMachine.states['PLAN']:
                ######### PASS the results of state estimation to planer (in global frame)
                # plan the path
                vel, theta = planner.run(state_estimator.X+58, state_estimator.Y+117.5)
                # vel = max([0.5, vel])
                if len(list(res_frontend['measurements'].values()))>=0 or all([value is not None for value in res_frontend['measurements'].values()])==False:
                    state_estimator.particle_filter_update(heading_vel=vel,
                                                           angular_vel=theta,
                                                           measurements=res_frontend['measurements'],
                                                           dt=dt)
                # plot the x and y
                # plot a fixed size window
                # plt.scatter(state_estimator.X, state_estimator.Y)
                # plt.xlim(-0.5, 4)
                # plt.ylim(-0.5, 4)
                # plt.show()
                print("Estimated Pose: {}, {}, {}".format(state_estimator.X, state_estimator.Y, state_estimator.Theta))
                print(f"Vel: {vel}, Theta: {theta}")

            if state_machine.state == StateMachine.states['MOVE']:
                # move the robot
                pass
                # from here go back to state == -1 with state transition
                # and controller.happy_moves()

        print(f"End of main loop.")

finally:
    # Stop streaming
    frontend.stop()

"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time

import numpy as np
import cv2
import pyrealsense2 as rs


CONNECT_SERVER = True  # False for local tests, True for deployment


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

COUNTING_BACK=0
# Use this function to send commands to the robot:
def send(sock, x, y, r):
    """
    Send a command to the robot.

    :param sock: TCP socket
    :param x: forward velocity (between -1 and 1)
    :param y: side velocity (between -1 and 1)
    :param r: yaw rate (between -1 and 1)
    """
    code = 1  # 1 for velocity commands
    data = struct.pack('<hfff', code, x, y, r)
    if sock is not None:
        sock.sendall(data)

# ---------- PID CONTROLLER ----------
# Source: https://github.com/m-lundberg/simple-pid/

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

# ---------- END PID CONTROLLER ----------

# Motion priors:

def spin(s, speed):
    """
    Rotate the robot in place.

    :param direction: 1 for clockwise, -1 for counterclockwise
    :param speed: speed of the rotation (between 0 and 1)
    """
    send(s, 0., 0., speed)

def go_backward(s, a):
    """
    Move the robot backward.
    """
    send(s, -a, 0., 0.)

def go_forward(s, a):
    """
    Move the robot forward.

    :param x: forward velocity (between -1 and 1)
    """
    send(s, a, 0., 0.)

def move(s, a, r):
    """
    Move the robot forward or backward and rotate it.

    :param a: forward velocity (between -1 and 1)
    :param r: angular velocity (between -1 and 1)
    """
    send(s, a, 0., r)

# ----------

# ---------- HELPERS FUNCTIONS ----------

def find_segments(arr):
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

def segment_length(segment):
    """Function to calculate the length of a segment"""
    return segment[1] - segment[0] + 1

def midpoint(segment):
    """Function to calculate the midpoint of a segment"""
    return (segment[0] + segment[1]) / 2

# Fisheye camera (distortion_model: narrow_stereo):

image_width = 640
image_height = 480

# --------- CHANGE THIS PART (optional) ---------

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("Could not find a depth camera with color sensor")
    exit(0)

# Depht available FPS: up to 90Hz
config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)
# RGB available FPS: 30Hz
config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
# # Accelerometer available FPS: {63, 250}Hz
# config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
# # Gyroscope available FPS: {200,400}Hz
# config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

# Start streaming
pipeline.start(config)

# ----------- DO NOT CHANGE THIS PART -----------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.markerBorderBits = 1

RECORD = False
history = []

# ----------------- CONTROLLER -----------------

align_to = rs.stream.color
align = rs.align(align_to)
GOAL_REACHED = False

try:
    # We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

    print("Client connecting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

        if CONNECT_SERVER:
            s.connect((SERVER_IP, SERVER_PORT))
            print("Connected.")
        else:
            s = None

        task_complete = False
        start_time = time.time()
        previous_time_stamp = start_time

        controller_theta = PID(
            Kp=1.0, Ki=0.05, Kd=0.05,
            setpoint=0, #  image_width / 2,
            sample_time=MIN_LOOP_DURATION,
            output_limits = [-1,1],
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=False,
            error_map=None,
            time_fn=None,
            starting_output=0
        )

        controller_vel = PID(
            Kp=1.5, Ki=0.01, Kd=0.08,
            setpoint=-0.5,
            sample_time=MIN_LOOP_DURATION,
            output_limits = [-1,1],
            auto_mode=True,
            proportional_on_measurement=False,
            differential_on_measurement=False,
            error_map=None,
            time_fn=None,
            starting_output=0
        )

        # main control loop:
        while not task_complete and not time.time() - start_time > TIMEOUT:

            # avoid busy loops:
            now = time.time()
            dt = now - previous_time_stamp
            if dt < MIN_LOOP_DURATION:
                time.sleep(MIN_LOOP_DURATION - dt)

            if GOAL_REACHED:
                send(s, 0., 0., 0.)
                break

            # ---------- CHANGE THIS PART (optional) ----------

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if RECORD:
                history.append((depth_frame, color_frame))

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # remove the measurement farther than 2.5m
            depth_image[depth_image > 2500] = 2500.

            # crop a square in the center of the depth image (480x640)
            depth_image_center = depth_image[200:280, 280:360]

            # compute the mean value of the center of the depth image
            depth_center_mean = np.mean(depth_image_center)

            # THE PID with setpoint -0.5 should already be able to keep the robot at a distance of 50 cm from the wall
            # # if the robot is too close to the wall (<50 cm), go back
            # if depth_center_mean < 500.:
            #     depth_center = 0
            #     go_backward(s, 1.)
            #     continue

            # crop an horizontal strip of the bottom part of depth image
            # given the setup of the arena, we want to avoid the robot 
            # to see outside its boundaries
            depth_image_crop = depth_image[280:, :]

            # compute the mean value ove the y-axis of the cropped depth image
            # this will give us an idea of the depth in front of the robot
            depth_crop_strip = np.mean(depth_image_crop, axis=0)

            # compute the median value of the perceived depth
            median_depth = np.median(depth_crop_mean)

            # threshold the depth image according to the median value
            depth_crop_strip[depth_crop_strip < median_depth] = 0
            depth_crop_strip[depth_crop_strip >= median_depth] = 1

            # compute the part of the image where the robot can move
            # i.e. the part where the depth is greater than the median value
            segments = find_segments(depth_crop_strip)

            # compute the center of the longest segment
            if segments:
                segment = max(segments, key=segment_length)
                depth_center = midpoint(segment)
                # center the depth_center in the depth image
                depth_center -= image_width / 2.
                # normalize the depth_center according to the image width
                theta_value = depth_center/ ((image_width / 2.) / 2.)
            else:
                theta_value = 0
            
            # --- Detect markers ---

            # Markers angular_output_cmd detection:
            grey_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(grey_frame, aruco_dict, parameters=arucoParams)

            angular_output_cmd = 0.0
            linear_vel=0
            if detected_ids is not None:
                for i, id_ in enumerate(detected_ids):
                    # we are interested in markers 1, 2 and 6, as they are close to the goal position
                    if id_!=6 or id_!=1 or id_!=2:
                        continue

                    # estimate the distance from the marker
                    int_corners = detected_corners[i][0].astype(int)
                    depth_estimate_mm = np.mean(depth_image[int_corners[:, 1], int_corners[:, 0]])
                    depth_estimate = depth_estimate_mm / 1000.

                    # compute the center of the marker in the image
                    landmark_center = detected_corners[i][0][0] + (detected_corners[i][0][2] - detected_corners[i][0][0]) / 2.
                    landmark_center = landmark_center.astype(int)
                    x = landmark_center[0]

                    # keep into account the obstacles before seeking the marker
                    if np.sign(theta_value) == np.sign(x):
                        theta_value = max(abs(theta_value), x) * np.sign(theta_value)
                    
                    print(f"Following marker {id_} at x={x}")
                    if depth_estimate < 0.6:
                        print("GOAL REACHED")
                        GOAL_REACHED = True
                        send(s, 0., 0., 0.)
                        break


            # compute the angular output command
            angular_output_cmd = controller_theta(theta_value, dt)

            # compute the linear velocity command
            # the higher the depth, the higher the velocity
            linear_vel=controller_vel(-depth_center_mean, dt)

            move(s, linear_vel, angular_output_cmd)
            #--- Compute control ---

        print(f"End of main loop.")

        if RECORD:
            import pickle as pkl
            with open("frames.pkl", 'wb') as f:
                pkl.dump(frames, f)
finally:
    # Stop streaming
    pipeline.stop()
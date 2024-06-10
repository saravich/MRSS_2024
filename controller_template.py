"""
This is the template of a python controller script to use with a server-enabled Agent.
"""

import struct
import socket
import time
import cv2
import numpy as np


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
    sock.sendall(data)


# Fisheye camera (distortion_model: narrow_stereo):

image_width = 928
image_height = 800

cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

# --------- CHANGE THIS PART (optional) ---------

# These un-distortion parameters are very rough (found by hand).
# You can get better parameters by doing a proper calibration via OpenCV.
# You need a chessboard picture for this purpose...
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

c_x = 928 / 2
c_y = 800 / 2
k_1 = -1.515
k_2 = 1.215
p_1 = 0
p_2 = 0
k_3 = 0.0

f = 2.42
f_x = c_x * f
f_y = c_y * f

camera_matrix = np.array([f_x, 0.0, c_x,
                          0.0, f_y, c_y,
                          0.0, 0.0, 1.0]).reshape(3, 3)

distortion_coefficients = np.array([k_1,
                                    k_2,
                                    p_1,
                                    p_2,
                                    k_3]).reshape(5, 1)

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                       distortion_coefficients,
                                                       (image_width, image_height),
                                                       1,
                                                       (image_width, image_height))

# ----------- DO NOT CHANGE THIS PART -----------

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.markerBorderBits = 1

# ----------------- CONTROLLER -----------------

# We create a TCP socket to talk to the Jetson at IP 192.168.123.14, which runs our walking policy:

print("Client connecting...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((SERVER_IP, SERVER_PORT))
    print("Connected.")

    code = 1  # 1 for velocity commands

    task_complete = False
    start_time = time.time()
    previous_time_stamp = start_time

    # main control loop:
    while not task_complete and not time.time() - start_time > TIMEOUT:

        # avoid busy loops:
        now = time.time()
        if now - previous_time_stamp < MIN_LOOP_DURATION:
            time.sleep(MIN_LOOP_DURATION - (now - previous_time_stamp))

        # capture camera frame:
        ret, frame = cam.read()

        # --------------- CHANGE THIS PART ---------------

        # --- Detect markers ---

        # Un-distort fisheye image:
        dst = cv2.undistort(dst, camera_matrix, distortion_coefficients, None, new_camera_matrix)

        # Markers detection:
        (detected_corners, detected_ids, rejected) = cv2.aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)

        # --- Compute control ---

        x_velocity = 0.0
        y_velocity = 0.0
        r_velocity = 0.0

        # --- Send control to the walking policy ---

        send(s, x_velocity, y_velocity, r_velocity)

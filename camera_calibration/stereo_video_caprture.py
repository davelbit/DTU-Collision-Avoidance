#!/usr/bin/env python3
######################################################################
# Authors:      <s210245> Varun Ghatrazu
#                     <s202385> David Parham
#
# Course:        Deep learning approaches for damage limitation in car-human collisions
# Semester:    Spring 2022
# Institution:  Technical University of Denmark (DTU)
#
# Module: Create the stereo vision images for both cameras
######################################################################

import cv2

import EasyPySpin
import sys
import time


t = time.localtime()
timestamp = time.strftime("%b-%d-%Y_%H%M", t)


capL = EasyPySpin.VideoCapture(0)
capR = EasyPySpin.VideoCapture(1)


def get_camera_settings(cam):
    settings = {
        "WIDTH": cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        "HEIGHT": cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "FPS": cam.get(cv2.CAP_PROP_FPS),
        "BIGHTNESS": cam.get(cv2.CAP_PROP_BRIGHTNESS),
        "CONTRAST": cam.get(cv2.CAP_PROP_CONTRAST),
        "SATURATION": cam.get(cv2.CAP_PROP_SATURATION),
        "HUE": cam.get(cv2.CAP_PROP_HUE),
        "EXPOSRUE": cam.get(cv2.CAP_PROP_EXPOSURE),
        "AUTO_WB": cam.get(cv2.CAP_PROP_AUTO_WB),
        "TRIGGER": cam.get(cv2.CAP_PROP_TRIGGER),
        "TRIGGER_DELAY": cam.get(cv2.CAP_PROP_TRIGGER_DELAY),
        "TEMPERATURE": cam.get(cv2.CAP_PROP_TEMPERATURE),
        "GAIN": cam.get(cv2.CAP_PROP_GAIN),
        "BACKLIGHT": cam.get(cv2.CAP_PROP_BACKLIGHT),
        "GAMMA": cam.get(cv2.CAP_PROP_GAMMA),
    }
    for key, value in settings.items():
        print(f"{key}: {value}")


def setup_camera(cam):
    cam.set_pyspin_value("PixelFormat", "BGR8")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2200)
    cam.set(cv2.CAP_PROP_FPS, 20.0)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.0)
    cam.set(cv2.CAP_PROP_CONTRAST, False)
    cam.set(cv2.CAP_PROP_SATURATION, False)
    cam.set(cv2.CAP_PROP_HUE, False)
    cam.set(cv2.CAP_PROP_EXPOSURE, 18000.0)
    cam.set(cv2.CAP_PROP_AUTO_WB, True)
    cam.set(cv2.CAP_PROP_TRIGGER, False)
    cam.set(cv2.CAP_PROP_TRIGGER_DELAY, 36.0)
    cam.set(cv2.CAP_PROP_TEMPERATURE, 47.25)
    cam.set(cv2.CAP_PROP_GAIN, 25)
    cam.set(cv2.CAP_PROP_BACKLIGHT, True)
    cam.set(cv2.CAP_PROP_GAMMA, 0.800048828125)


setup_camera(capL)
setup_camera(capR)

get_camera_settings(capL)
print("")
get_camera_settings(capR)

width = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"camera resolution: ({width}x{height})")

size = (width, height)
fps = int(capL.get(cv2.CAP_PROP_FPS))

videoL = cv2.VideoWriter(
    f"videos/left/videoL-{timestamp}.mp4", cv2.VideoWriter_fourcc(*"X264"), fps, size
)
videoR = cv2.VideoWriter(
    f"videos/right/videoR-{timestamp}.mp4", cv2.VideoWriter_fourcc(*"X264"), fps, size
)

down_scaling = 0.25

while capL.isOpened() and capR.isOpened():

    success1, imgL = capL.read()
    success2, imgR = capR.read()

    if not all((success1, success2)):
        break

    videoL.write(imgL)
    videoR.write(imgR)

    resized_L = cv2.resize(
        imgL, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR
    )
    resized_R = cv2.resize(
        imgR, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR
    )
    cv2.imshow("Left image", resized_L)
    cv2.imshow("Right image", resized_R)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release and destroy all windows before termination
capL.release()
capR.release()

videoL.release()
videoR.release()

cv2.destroyAllWindows()
print("The video was successfully saved")

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
timestamp = time.strftime('%b-%d-%Y_%H%M', t)


capL = EasyPySpin.VideoCapture(0)
capR = EasyPySpin.VideoCapture(1)

width = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"camera resolution: ({width}x{height})")

size = (width, height)
fps = int(capL.get(cv2.CAP_PROP_FPS))

videoL = cv2.VideoWriter(f"videos/left/videoL-{timestamp}.mp4", cv2.VideoWriter_fourcc(*'X264'), fps, size)
videoR = cv2.VideoWriter(f"videos/right/videoR-{timestamp}.mp4", cv2.VideoWriter_fourcc(*'X264'), fps, size)

down_scaling = 0.25

while capL.isOpened() and capR.isOpened():

    success1, imgL = capL.read()
    success2, imgR = capR.read()

    if all((success1, success2)):

        videoL.write(imgL)
        videoR.write(imgR)

        resized_L = cv2.resize(imgL, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR)
        resized_R = cv2.resize(imgR, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Left image", resized_L)
        cv2.imshow("Right image", resized_R)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# Release and destroy all windows before termination
capL.release()
capR.release()

videoL.release()
videoR.release()

cv2.destroyAllWindows()
print("The video was successfully saved")

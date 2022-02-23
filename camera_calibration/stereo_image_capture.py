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


capL = EasyPySpin.VideoCapture(0)
capR = EasyPySpin.VideoCapture(1)
# sys.exit(-1)


num = 0

down_scaling = 0.25

while capL.isOpened() and capR.isOpened():

    success1, imgL = capL.read()
    success2, imgR = capR.read()

    k = cv2.waitKey(5)

    if k == 27 or k == ord("q"):
        break

    elif k == ord("s"):  # wait for 's' key to save and exit
        cv2.imwrite(f"images/stereo_left/imageL{num}.png", imgL)
        cv2.imwrite(f"images/stereo_right/imageR{num}.png", imgR)
        print(f"{num}: images saved!")
        num += 1

    resized_L = cv2.resize(imgL, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR)
    resized_R = cv2.resize(imgR, None, fx=down_scaling, fy=down_scaling, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Left image", resized_L)
    cv2.imshow("Right image", resized_R)

# Release and destroy all windows before termination
capL.release()
capR.release()

cv2.destroyAllWindows()

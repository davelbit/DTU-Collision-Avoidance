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

import cv2 as cv
import sys
import EasyPySpin


# Camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open("stereoMap.xml", cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()

# Open both cameras
# cap_left = EasyPySpin.VideoCapture(0)
# cap_right = EasyPySpin.VideoCapture(1)

cap_left = cv.VideoCapture("videos/left/videoL-Feb-19-2022_1605.mp4")
cap_right = cv.VideoCapture("videos/right/videoR-Feb-19-2022_1605.mp4")

# Check if camera opened successfully
if cap_left.isOpened() == False:
    print("Error opening left video file")
    sys.exit(-1)

if cap_right.isOpened() == False:
    print("Error opening right video file")
    sys.exit(-1)

frameSize = (640, 480)

while cap_right.isOpened() and cap_left.isOpened():

    success_left, frame_left = cap_left.read()
    success_right, frame_right = cap_right.read()

    # Undistort and rectify images
    frame_left = cv.remap(
        frame_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0
    )
    frame_right = cv.remap(
        frame_right,
        stereoMapR_x,
        stereoMapR_y,
        cv.INTER_LANCZOS4,
        cv.BORDER_CONSTANT,
        0,
    )

    frame_left = cv.resize(frame_left, frameSize)
    frame_right = cv.resize(frame_right, frameSize)

    # Show the frames
    cv.imshow("frame left", frame_left)
    cv.imshow("frame right", frame_right)

    # Hit "q" to close the window
    if cv.waitKey(25) & 0xFF == ord("q"):
        break


# Release and destroy all windows before termination
cap_left.release()
cap_right.release()


cv.destroyAllWindows()

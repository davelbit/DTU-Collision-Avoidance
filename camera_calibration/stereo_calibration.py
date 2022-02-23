#!/usr/bin/env python3
######################################################################
# Authors:      <s210245> Varun Ghatrazu
#                     <s202385> David Parham
#
# Course:        Deep learning approaches for damage limitation in car-human collisions
# Semester:    Spring 2022
# Institution:  Technical University of Denmark (DTU)
#
# Module: Used for stereo camera calibration
######################################################################

import glob
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_object_points_and_image_points(chessboardSize=(9, 6)):
    """Find chessboard corners and object and image points"""

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    frameSize = (3208, 2200)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(
        -1, 2
    )

    size_of_chessboard_squares_mm = 39
    objp = objp * size_of_chessboard_squares_mm

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpointsL = []  # 2d points in image plane.
    imgpointsR = []  # 2d points in image plane.

    imagesLeft = glob.glob("images/stereo_left/*.png")
    imagesRight = glob.glob("images/stereo_right/*.png")

    # Images should be perfect pairs. Otherwise all the calibration will be false.
    # Be sure that first cam and second cam images are correctly prefixed and numbers are ordered as pairs.
    # Sort will fix the globs order to ensure correct ordering.
    imagesLeft.sort()
    imagesRight.sort()

    # Pairs should be same size. Otherwise we have sync problem.
    if len(imagesLeft) != len(imagesRight):
        print("Numbers of left and right images are not equal. They should be pairs.")
        print("Left images count: ", len(imagesLeft))
        print("Right images count: ", len(imagesRight))
        sys.exit(-1)

    for imgLeft, imgRight in tqdm(zip(imagesLeft, imagesRight), total=len(imagesLeft)):

        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)

        imgL = cv.resize(imgL, frameSize)
        imgR = cv.resize(imgR, frameSize)

        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if retL and retR == True:

            objpoints.append(objp)

            cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsL.append(cornersL)

            cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)

            # Draw and display the corners
            cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
            cv.imshow("img left", imgL)
            cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
            cv.imshow("img right", imgR)
            cv.waitKey(1000)

        else:
            print("Chessboard couldn't detected. Image pair: ", imgL, " and ", imgR)
            continue

    cv.destroyAllWindows()

    cameraL = {"imgpointsL": imgpointsL, "imgL": imgL, "grayL": grayL}
    cameraR = {"imgpointsR": imgpointsR, "imgR": imgR, "grayR": grayR}

    return (objpoints, cameraL, cameraR)


def calibration(objpoints, cameraL, cameraR):
    """Create the calibration data for the respective cameras"""
    frameSize = (3208, 2200)

    retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
        objpoints, cameraL.get("imgpointsL"), frameSize, None, None
    )
    heightL, widthL, channelsL = cameraL.get("imgL").shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
        cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL)
    )

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
        objpoints, cameraR.get("imgpointsR"), frameSize, None, None
    )
    heightR, widthR, channelsR = cameraR.get("imgR").shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
        cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR)
    )

    cameraL.update({"newCameraMatrixL": newCameraMatrixL, "distL": distL})
    cameraR.update({"newCameraMatrixR": newCameraMatrixR, "distR": distR})

    return (cameraL, cameraR)


def calc_essential_and_fundamental_matrix(objpoints, cameraL, cameraR):
    # Stereo Vision Calibration

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
    # Hence intrinsic parameters are the same

    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
    (
        retStereo,
        newCameraMatrixL,
        distL,
        newCameraMatrixR,
        distR,
        rot,
        trans,
        essentialMatrix,
        fundamentalMatrix,
    ) = cv.stereoCalibrate(
        objpoints,
        cameraL.get("imgpointsL"),
        cameraR.get("imgpointsR"),
        cameraL.get("newCameraMatrixL"),
        cameraL.get("distL"),
        cameraR.get("newCameraMatrixR"),
        cameraR.get("distR"),
        cameraL.get("grayL").shape[::-1],
        criteria_stereo,
        flags,
    )

    cameraL.update({"newCameraMatrixL": newCameraMatrixL, "distL": distL})
    cameraR.update({"newCameraMatrixR": newCameraMatrixR, "distR": distR})

    return (cameraL, cameraR, rot, trans)


def save_rectified_calibraton_data(cameraL, cameraR, rot, trans):
    """Rectify stereo cameras and save the calibration data"""

    rectifyScale = 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
        cameraL.get("newCameraMatrixL"),
        cameraL.get("distL"),
        cameraR.get("newCameraMatrixR"),
        cameraR.get("distR"),
        cameraL.get("grayL").shape[::-1],
        rot,
        trans,
        rectifyScale,
        (0, 0),
    )

    stereoMapL = cv.initUndistortRectifyMap(
        cameraL.get("newCameraMatrixL"),
        cameraL.get("distL"),
        rectL,
        projMatrixL,
        cameraL.get("grayL").shape[::-1],
        cv.CV_16SC2,
    )
    stereoMapR = cv.initUndistortRectifyMap(
        cameraR.get("newCameraMatrixR"),
        cameraR.get("distR"),
        rectR,
        projMatrixR,
        cameraR.get("grayR").shape[::-1],
        cv.CV_16SC2,
    )

    print("Saving parameters!")
    cv_file = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_WRITE)

    cv_file.write("stereoMapL_x", stereoMapL[0])
    cv_file.write("stereoMapL_y", stereoMapL[1])
    cv_file.write("stereoMapR_x", stereoMapR[0])
    cv_file.write("stereoMapR_y", stereoMapR[1])

    cv_file.release()


if __name__ == "__main__":
    objpoints, cameraL, cameraR = get_object_points_and_image_points()
    cameraL, cameraR = calibration(objpoints, cameraL, cameraR)
    cameraL, cameraR, rot, trans = calc_essential_and_fundamental_matrix(
        objpoints, cameraL, cameraR
    )
    save_rectified_calibraton_data(cameraL, cameraR, rot, trans)

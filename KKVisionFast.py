#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2019 license: use it as much as you want. Crediting is recommended because it lets me know that I am being useful.
# Credit to Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen

#BASED ON CHICKENVISION
#https://github.com/team3997/ChickenVision
#----------------------------------------------------------------------------

import json
import time
import sys


from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
import math

###################### PROCESSING OPENCV ################################

#Angles in radians

#image size ratioed to 16:9
image_width = 240
image_height = 135

#Lifecam 3000 from datasheet
#Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf
diagonalView = math.radians(68.5)

#16:9 aspect ratio
horizontalAspect = 16
verticalAspect = 9

#Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
#Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView/2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView/2) * (verticalAspect / diagonalAspect)) * 2

#Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = image_width / (2*math.tan((horizontalView/2)))
V_FOCAL_LENGTH = image_height / (2*math.tan((verticalView/2)))

#Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip( frame, -1 )

# Masks the video based on a range of hsv colors
# Takes in a frame, returns a masked frame
def threshold_video(frame):
    img = frame.copy()
    #blur = cv2.blur(img, 7)
    #blur = cv2.medianBlur(img, 3)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of green light in HSV
    #lower_color = np.array([0,220,25])
    #upper_color = np.array([101, 255, 255])
    lower_color = np.array([60, 80, 215])
    upper_color = np.array([110, 255, 255])
    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)


    # Returns the masked imageBlurs video to smooth out image


    return mask



# Finds the contours from the masked image and displays them on original stream
def findContours(frame, mask):
    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage) #TODO finding largest
    if len(contours) != 0:
        #the following line will draw red lines around targets
        #image = cv2.drawContours(frame, contours, -1, (0,0,255), 3)
        image = findTargets(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video

    img= frame.copy()
    

    #Drawing convexHull/
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray ,50, 255, cv2.THRESH_BINARY)
    _, contour2, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #create am empty black image/


    #draw contours and hull points/

    #Creating hull array for convex hull points/
    
    #hull = []
    

    #calculate points for each contour
    for i in range(len(contour2)):

        #creating convex hull object for each contour
        #hull.append(cv2.convexHull(contour2[i], False))

        color_contours = (0, 225, 0)#this makes color for contours green/
        color = (225, 0, 0) #makes color for convex hull blue/
        #draw it with contour/
    #    cv2.drawContours(image, contour2, i, color_contours, 1, 8, hierarchy)
        #draw ith cpnvex hull object
     #   cv2.drawContours(image, hull, i, color, 1, 8)
    
    return image
    


# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTargets(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape;
    #Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        #Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        biggestCnts = []
        posXs = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            #hull = cv2.convexHull(cnt)
            # Calculate Contour area
            #cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            #hullArea = cv2.contourArea(hull)
            # Filters contours based off of size
            #if (checkContours(cntArea, hullArea)):
            if (True):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if(len(biggestCnts) < 13):
                    #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE ##########
                    #rotation = getEllipseRotation(image, cnt)

                    # Calculates yaw of contour (horizontal position in degrees)
                    #yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    #pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Get the rotation of the rectangle
                    #rect is the (top,left), (height,width), angle
                    rotation = rect[2]
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle
                    #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)
                    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)


                    # Calculates yaw of contour (horizontal position in degrees)
                    #yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    #pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)


                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    #cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    #cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    #(x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    #center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    #radius = int(radius)
                    # Makes bounding rectangle of contour
                    #rx, ry, rw, rh = cv2.boundingRect(cnt)
                    #boundingRect = cv2.boundingRect(cnt)
                    # Draws countour of bounding rectangle and enclosing circle in green
                    #cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    #cv2.circle(image, center, radius, (23, 184, 80), 1)

                    # Appends important info to array
                    #don't append if already in the array - determined by X position
                    #if [cx, cy, rotation, cnt] not in biggestCnts:
                    if cs not in posXs:
                        posXs.append(cx)
                        biggestCnts.append([cx, cy, rotation, cnt])


        # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # Target Checking
        for i in range(len(biggestCnts) - 1):
            #Rotation of two adjacent contours
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            #openCV function minAreaRect returns a value between -90 and 0 (excluding 0) 
            #testing shows that reflective tape for 2019 should be about -75 and -15 degrees
            #so adding 45 degrees should result in two values on opposite sides of 0 for the test below
            tilt1 = tilt1 + 45
            tilt2 = tilt2 + 45 

            #x coords of contours
            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]
            # If contour angles are opposite

            #print("Angles: " + str(tilt1) + " : " + str(tilt2))

            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                #ellipse negative tilt means rotated to right
                #Note: if using rotated rect (min area rectangle)
                #      negative tilt means rotated to left
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        print("Skipping contour - left is pointing left")
                        continue
                # If right contour rotation is tilted to the right then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        print("Skipping contour - right is pointing right")
                        continue
                #Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                
                #Push to NetworkTable
                #table.putNumber("yawToTarget", yawToTarget)
                
                #Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget])
    #Check if there are targets seen
    if (len(targets) > 0):
        #Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        #Draws yaw of target + line where center of target is
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

        currentAngleError = finalTarget[1]
        
        table.putNumber("currentAngleError", currentAngleError)
        
    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image


# Checks if contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    return cntSize > 200


#Forgot how exactly it works, but it works!
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfTargetFromCamera = heightOfTarget - heightOfCamera

    # Uses trig and pitch to find distance to target
    '''
    d = distance
    h = height between camera and target
    a = angle = pitch

    tan a = h/d (opposite over adjacent)

    d = h / tan a

                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
    '''
    distance = math.fabs(heightOfCameraFromTarget / math.tan(math.radians(pitch)))

    return distance


# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)

def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        # Gets smaller side
        if widthE > heightE:
            smaller_side = heightE
        else:
            smaller_side = widthE

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation

#################### FRC VISION PI Image Specific #############
configFile = "/boot/frc.json"

class CameraConfig: pass

team = None
server = False
cameraConfigs = []

"""Report parse error."""
def parseError(str):
    print("config error in '" + configFile + "': " + str, file=sys.stderr)

"""Read single camera configuration."""
def readCameraConfig(config):
    cam = CameraConfig()

    # name
    try:
        cam.name = config["name"]
    except KeyError:
        parseError("could not read camera name")
        return False

    # path
    try:
        cam.path = config["path"]
    except KeyError:
        parseError("camera '{}': could not read path".format(cam.name))
        return False

    cam.config = config

    cameraConfigs.append(cam)
    return True

"""Read configuration file."""
def readConfig():
    global team
    global server

    # parse file
    try:
        with open(configFile, "rt") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False

    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))

    # cameras
    try:
        cameras = j["cameras"]
    except KeyError:
        parseError("could not read cameras")
        return False
    for camera in cameras:
        if not readCameraConfig(camera):
            return False

    return True

"""Start running the camera."""
def startCamera(config):
    print("Starting camera '{}' on {}".format(config.name, config.path))
    cs = CameraServer.getInstance()
    camera = cs.startAutomaticCapture(name=config.name, path=config.path)

    camera.setConfigJson(json.dumps(config.config))

    return cs, camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables and create table instance
    ntinst = NetworkTablesInstance.getDefault()
    table = NetworkTables.getTable("PiData")
    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)

    # start cameras
    cameras = []
    streams = []
    for cameraConfig in cameraConfigs:
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    
    #Get the first camera
    cameraServer0 = streams[0]
    
    # Get a CvSink. This will capture images from the camera
    cvSink0 = cameraServer0.getVideo()

    cameraServer1 = streams[1]
    cvSink1 = cameraServer1.getVideo()

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    outputStream = cameraServer0.putVideo("stream", image_width, image_height)
    outputStream2 = cameraServer1.putVideo("stream1", image_width, image_height)
    # Allocating new images is very expensive, always try to preallocate
    img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

    table.putNumber('camIndex', 0)
    
    # loop forever
    while True:

        camIndex = table.getNumber('camIndex', 0)

        print("CamIndex " + str(camIndex))
        if int(camIndex) == 0:
            print("FRONT CAM")
            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            timestamp, img = cvSink0.grabFrame(img)
            frame = img
            #frame = flipImage(img)
            if timestamp == 0:
                # Send the output the error.
                outputStream.notifyError(cvSink0.getError())
                # skip the rest of the current iteration
                continue
            threshold = threshold_video(frame)
            #outputStream.putFrame(threshold)
            processed = findContours(frame, threshold)
            # (optional) send some image back to the dashboard
            outputStream.putFrame(processed)


        if int(camIndex) == 1:
            print("BACK CAM")
            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            timestamp, img = cvSink1.grabFrame(img)
            frame = img
            #frame = flipImage(img)
            if timestamp == 0:
                # Send the output the error.
                outputStream2.notifyError(cvSink1.getError())
                # skip the rest of the current iteration
                continue
            threshold = threshold_video(frame)
            #outputStream.putFrame(threshold)
            processed = findContours(frame, threshold)
            # (optional) send some image back to the dashboard
            outputStream2.putFrame(processed)



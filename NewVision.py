#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2019 license: use it as much as you want. Crediting is recommended because it lets me know that I am being useful.
# Credit to Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
#----------------------------------------------------------------------------
#import board
#import neopixel
import json
import time
import sys
from threading import Thread


from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance
import cv2
import numpy as np
from networktables import NetworkTables
import math
import datetime
import time

########### SET RESOLUTION TO 256x144 !!!! ############

class driverLights:
    def __init__(self, sboard): 
        self.stopped = False

    def start(self):
        Thread(target=self.run, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def run(self):
        print("Starting LED")
        pixels = neopixel.NeoPixel(board.D10, 59, brightness=0.5, auto_write=False)
        print("Have Pixels")
        blue = (0, 0, 255)
        red = (255, 0, 0)
        yellow = (255, 255, 0)
        black = (0, 0, 0)
        purple = (255, 0, 255)

        while not self.stopped:            
            #status = sboard.getString("LedStatus")
            #centerValue = sboard.getNumber("xPercent")
            status = 'vision'
            centerValue = .5
            pixels.fill(black) #clear last status
            if status == 'climber':
                for x in range(0,20):
                    pixels[x] = blue
                for x in range(39,59):
                    pixels[x] = blue
            elif status == 'rocket1':
                pixels.fill(red)
            elif status == 'rocket2':
                pixels.fill(yellow)
            elif status == 'vision':
                index = math.floor(17*centerValue)
                pixels[index + 21] = purple 
            pixels.show()

#Class to examine Frames per second of camera stream. Currently not used.
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()
#class that runs separate thread for showing video,
class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, imgWidth, imgHeight, cameraServer, frame=None):
        self.outputStream = cameraServer.putVideo("stream", imgWidth, imgHeight)
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            self.outputStream.putFrame(self.frame)

    def stop(self):
        self.stopped = True
    def notifyError(self, error):
        self.outputStream.notifyError(error)

# Class that runs a separate thread for reading  camera server also controlling exposure.
class WebcamVideoStream:
    def __init__(self, camera, cameraServer, frameWidth, frameHeight, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream

        #Automatically sets exposure to 0 to track tape
        self.webcam = camera
        #self.webcam.setExposureManual(manual_exposure_level)
        #self.webcam.setBrightness(manual_brightness_level)
        #Some booleans so that we don't keep setting exposure over and over to the same value
        self.autoExpose = False
        self.prevValue = self.autoExpose
        #Make a blank image to write on
        self.img = np.zeros(shape=(frameWidth, frameHeight, 3), dtype=np.uint8)
        #Gets the video
        self.stream = cameraServer.getVideo()
        (self.timestamp, self.img) = self.stream.grabFrame(self.img)

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def setStream(self, cameraName):
        self.stream = cameraServer.getVideo(name = cameraName)

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            #Boolean logic we don't keep setting exposure over and over to the same value
            if self.autoExpose:
                if(self.autoExpose != self.prevValue):
                    self.prevValue = self.autoExpose
                    self.webcam.setExposureAuto()
            else:
                if (self.autoExpose != self.prevValue):
                    self.prevValue = self.autoExpose
                    self.webcam.setExposureManual(manual_exposure_level)
                    self.webcam.setBrightness(manual_brightness_level)
            #gets the image and timestamp from cameraserver
            (self.timestamp, self.img) = self.stream.grabFrame(self.img)

    def read(self):
        # return the frame most recently read
        return self.timestamp, self.img

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
    def getError(self):
        return self.stream.getError()

###################### PROCESSING OPENCV ################################

manual_exposure_level = -2
manual_brightness_level = 20

#Angles in radians

#image size ratioed to 16:9
image_width = 256
image_height = 144

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
#blurs have to be odd
front_green_blur = 1
back_green_blur = 7
orange_blur = 27

# define range of green of retroreflective tape in HSV
back_lower_green = np.array([42,0,215])
back_upper_green = np.array([103,121, 255])
front_lower_green = np.array([62,0,16])
front_upper_green = np.array([180, 255, 255])

#Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip( frame, -1 )

#Blurs frame
def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img,(blur_radius,blur_radius))
    return blur

# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):


    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # hold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image

    return mask



# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):
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
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findTape(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

def getTapeHeight(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return h

# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    #Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        #Sort contours by height (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: getTapeHeight(x), reverse=True)
        maxHeight = 0
        
        matches = []
        biggestCnts = []
        bottomOfScreenY = image_height - (image_height*.3) #only bother to process the target if it is in the top 70%
        #print("NEW LOOP")
        for cnt in cntsSorted:  
            if len(biggestCnts) >= 4:
                #we have the 4 tallest contours, stop looping
                break

            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Gets the centeroids of contour
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
                #skip this loop
                continue

            #get the left, top, width, and height of the contour
            boxX, boxY, boxW, boxH = cv2.boundingRect(cnt)
            #only bother to process the target if is is above the bottom of the screen and it is close to the same height as the biggest blob
            if(boxH >= (maxHeight * .7)):
                #(cy < bottomOfScreenY) and 
                #print ("X=" + str(boxX) + "  Y=" + str(boxY) + "  W=" + str(boxW) + "  H=" + str(boxH))

                if(maxHeight == 0):
                    maxHeight = boxH #will be set on first loop, first item in array will be tallest

                #draw a box around this contour if dubugging
                if (shuffleBoard.getBoolean("CameraDebug", False)):
                    cv2.rectangle(image, (boxX, boxY), (boxX + boxW, boxY + boxH), (23, 184, 80), 1)

                #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE, DRAWS ELLIPSE IF IN DEBUG MODE ##########
                rotation = getEllipseRotation(image, cnt)

                if [cx, cy] not in matches:
                    matches.append([cx, cy])
                    biggestCnts.append([cx, cy, rotation, cnt, boxH])

                ##### DRAW DEBUG CONTOUR######
                # Gets rotated bounding rectangle of contour
                #rect = cv2.minAreaRect(cnt)
                # Creates box around that rectangle
                #box = cv2.boxPoints(rect)
                # Not exactly sure
                #box = np.int0(box)
                # Draws rotated rectangle
                #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)


                # Calculates yaw of contour (horizontal position in degrees)
                #yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                # Calculates yaw of contour (horizontal position in degrees)
                #pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                
                # Draws a vertical white line passing through center of contour
                #cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
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
                #boundingRect = cv2.boundingRect(cnt)

                #cv2.circle(image, center, radius, (23, 184, 80), 1)

                # Appends important info to array
                #if [cx, cy, rotation, cnt, rh] not in biggestCnts:


        # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # Target Checking
        for i in range(len(biggestCnts) - 1):
            #Rotation of two adjacent contours
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            #x coords of contours
            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]

            rh1 = biggestCnts[i][4]
            rh2 = biggestCnts[i + 1][4]
            # If contour angles are opposite
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                avgHeight = (rh1+rh2)/2
                #ellipse negative tilt means rotated to right
                #Note: if using rotated rect (min area rectangle)
                #      negative tilt means rotated to left
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        continue
                #Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                #Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget, avgHeight] not in targets:
                    targets.append([centerOfTarget, yawToTarget, avgHeight])
    
    #Check if there are targets seen
    if (len(targets) > 0):
        # pushes that it sees vision target to network tables
        shuffleBoard.putBoolean("tapeDetected", True)
        print("true")
        #Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        #Draws yaw of target + line where center of target is
        #cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    #(255, 255, 255))
        if (shuffleBoard.getBoolean("CameraDebug", False)):
            cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)
            cv2.line(image, (int(shuffleBoard.getNumber("centerOffset", 15)+finalTarget[0]), screenHeight), (int(shuffleBoard.getNumber("centerOffset", 15)+finalTarget[0]), 0), (255,255,0), 2)
        
        currentAngleError = finalTarget[1]
        # pushes vision target angle to network tables
        networkTable.putNumber("tapeYaw", currentAngleError)
        shuffleBoard.putNumber("targetX", finalTarget[0])
        shuffleBoard.putNumber("targetY", finalTarget[2])
    else:
        # pushes that it deosn't see vision target to network tables
        shuffleBoard.putBoolean("tapeDetected", False)
        print("false")
        shuffleBoard.putNumber("targetX", -1)

    #cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image


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
    distance = math.fabs(heightOfTargetFromCamera / math.tan(math.radians(pitch)))

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

        if (shuffleBoard.getBoolean("CameraDebug", False)):
            cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
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

    #if (config.name == "Front"):
    #    print("Manually Setting Brightness for Front Cam")
    #    camera.setBrightness(20)
 
    return cs, camera

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        configFile = sys.argv[1]
    # read configuration
    if not readConfig():
        sys.exit(1)

    # start NetworkTables
    ntinst = NetworkTablesInstance.getDefault()
    #Name of network table - this is how it communicates with robot. IMPORTANT
    networkTable = NetworkTables.getTable('ChickenVision')
    shuffleBoard = NetworkTables.getTable('SmartDashboard')
    
    #leds = driverLights(shuffleBoard).start()

    if server:
        print("Setting up NetworkTables server")
        ntinst.startServer()
    else:
        print("Setting up NetworkTables client for team {}".format(team))
        ntinst.startClientTeam(team)


    # start cameras
    cameras = []
    streams = []

    foundDriver = False
    foundFront = False
    foundBack = False
    for cameraConfig in cameraConfigs:
        if cameraConfig.name == "Front":
            foundFront = True
        elif cameraConfig.name == "Back":
            foundBack = True
        elif cameraConfig.name == "Driver":
            foundDriver = True
        cs, cameraCapture = startCamera(cameraConfig)
        streams.append(cs)
        cameras.append(cameraCapture)
    #Get the first camera

    if len(cameras) == 0:
        print("No Cameras Attached")
    else:
        webcam = cameras[0]
        cameraServer = streams[0]
        #Start thread reading camera
        if foundDriver:
            driverStationCap = WebcamVideoStream(webcam, cameraServer, image_width, image_height, "DriverStation").start()
            driverStationCap.setStream("Driver")
            #driverStationCap.autoExpose = True
        if foundFront:
            visionCap = WebcamVideoStream(webcam, cameraServer, image_width, image_height, "VisionProcessing").start()
            visionCap.setStream("Front")
        
        # (optional) Setup a CvSource. This will send images back to the Dashboard
        # Allocating new images is very expensive, always try to preallocate
        img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
        #Start thread outputing stream
        streamViewer = VideoShow(image_width,image_height, cameraServer, frame=img).start()
        fps = FPS().start()
        #TOTAL_FRAMES = 200;
        print("Front (vision): " + str(foundFront) + "  Driver (front): " + str(foundDriver) + "  Back: " + str(foundBack)) 
        # loop forever
        while True:
            # Tell the CvSink to grab a frame from the camera and put it
            # in the source image.  If there is an error notify the output.
            #get image for driver camera - this will be front or back depending on which driver selected
            if foundDriver: #starts as front driver camera
                dsTimestamp, dsImg = driverStationCap.read()
            if foundFront: #starts as front vision camera     
                vTimestamp, vImg = visionCap.read()        
            #use "flipImage(img)" if the camera is upside down

            isProcessingFrontVision = shuffleBoard.getBoolean("Camera Toggle", True)
            isDebuggingVision = shuffleBoard.getBoolean("CameraDebug", True)

            processSuccess = False
            if isProcessingFrontVision:
                #do vision proccessing
                if foundFront:
                    if vTimestamp == 0: #failed to capture image
                        print("timestamp is 0, failed to capture image")
                        if isDebuggingVision:
                            streamViewer.notifyError(visionCap.getError())
                    else:
                        boxBlur = blurImg(vImg, front_green_blur)
                        threshold = threshold_video(front_lower_green, front_upper_green, boxBlur)
                        processed = findTargets(vImg, threshold)
                        processSuccess = True
                
                #send image back
                if isDebuggingVision and foundFront:
                    networkTable.putNumber("VideoTimestamp", vTimestamp)
                    if(processSuccess):
                        streamViewer.frame = processed #send back processed image
                elif foundDriver:
                    if dsTimestamp == 0: #failed to capture image
                        streamViewer.notifyError(driverStationCap.getError())
                    else:   
                        networkTable.putNumber("VideoTimestamp", dsTimestamp)                 
                        streamViewer.frame = dsImg #send back driver image
            elif foundBack: #back vision tracking
                #do vision proccessing
                if dsTimestamp == 0: #failed to capture image
                    streamViewer.notifyError(driverStationCap.getError())
                else:
                    boxBlur = blurImg(dsImg, back_green_blur)
                    threshold = threshold_video(back_lower_green, back_upper_green, boxBlur)
                    processed = findTargets(dsImg, threshold)
                    processSuccess = True

                    #send image back
                    if isDebuggingVision:
                        networkTable.putNumber("VideoTimestamp", vTimestamp)
                        if(processSuccess):
                            streamViewer.frame = processed #send back processed image
                    else:
                        networkTable.putNumber("VideoTimestamp", dsTimestamp)
                        streamViewer.frame = dsImg #send back driver image

            # update the FPS counter
            fps.update()
            #Flushes camera values to reduce latency
            ntinst.flush()

            #set the front/back camera the driver has selected - will be used in next loop
            try:
                if foundDriver and shuffleBoard.getBoolean("Camera Toggle", True):
                    driverStationCap.setStream("Driver")
                elif foundBack:
                    driverStationCap.setStream("Back")
            except:
                print("Failed to set camera stream to front/back")
                    
        #Doesn't do anything at the moment. You can easily get this working by indenting these three lines
        # and setting while loop to: while fps._numFrames < TOTAL_FRAMES
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))





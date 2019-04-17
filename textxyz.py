#Imports
import cv2
import numpy as np
import math

#CHICKEN CONSTANTS
#
#
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

front_green_blur = 1
front_lower_green = np.array([36,92,156])
front_upper_green = np.array([95, 255, 255])
#
#
#
#


# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define object points for a 9x6 grid
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
objp = objp * 26
#objp = np.array([[15,150,0],[-40,135,0],[0,0,0],[50,15,0],[290,150,0],[255,15,0],[305,0,0],[345,135,0]], dtype=np.float32)

#print(objp)

mtx = np.array([[653.90217311,   0,        302.49861937],[  0,         654.67708708, 254.90595383],[  0,          0,          1        ]])

dist = np.array([[ 2.29930082e-01, -3.71583204e+00,  1.35057973e-02, -6.10760675e-03, 1.71232055e+01]])

# mtx = np.array([[846.45853739 ,   0,         316.32230442],[  0,         847.93343997, 228.07883833],[  0,          0,          1        ]])
# dist = np.array([[-6.68957067e-02, -1.34039967e+00, -7.84647125e-03, -1.59645273e-02,1.05726650e+01]])
# Arrays to store object points and image points
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

vd = cv2.VideoCapture(1)

def testcam(imgp, img):
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)

    print("dist\n\n", dist)
    print("Camera calibration matrix\n\n", mtx)

    b, rvec, tvec, inliers = cv2.solvePnPRansac(objp, imgp, mtx, dist)
    print("Rvec\n", rvec)
    print("\nTvec", tvec)

    
    dst, jacobian = cv2.Rodrigues(rvec)
    x = tvec[0][0]
    y = tvec[2][0]
    t = (math.asin(-dst[0][2]))

    print("X", x, "Y", y, "Angle", t)
    print("90-t", (math.pi/2) - t)

    Rx = y * (math.cos((math.pi/2) - t))
    Ry = y * (math.sin((math.pi/2) - t))

    print("rx", Rx, "ry", Ry)

    #Save camera matrix and distortion coefficients to be used later
    np.save('cam_broke_mtx', mtx)
    np.save('cam_broke_dist', dist)

#CHICKEN CODE!!!
#
#
#
#





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
    #cv2.imshow("mask",mask)
    return mask

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

        #Forgot how exactly it works, but it works!
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)

def getTapeHeight(contour):
    x,y,w,h = cv2.boundingRect(contour)
    return h

# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)

# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):
    # Finds contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
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
        print("contours: ", len(contours))
        image = findTape(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    #Seen vision targets (correct angle, adjacent to each other)
    targets = []
    targetPoints = np.array([])

    if len(contours) >= 2:
        print("2 or more, 192")
        #Sort contours by height (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: getTapeHeight(x), reverse=True)
        maxHeight = 0
        
        matches = []
        biggestCnts = []
        bottomOfScreenY = image_height - (image_height*.01) #only bother to process the target if it is in the top 70%
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
            if( boxH >= (maxHeight * .75)): #(cy < bottomOfScreenY) and
                #print ("X=" + str(boxX) + "  Y=" + str(boxY) + "  W=" + str(boxW) + "  H=" + str(boxH))

                if(maxHeight == 0):
                    maxHeight = boxH #will be set on first loop, first item in array will be tallest

                #draw a box around this contour if dubugging
                #if (shuffleBoard.getBoolean("CameraDebug", False)):
                cv2.rectangle(image, (boxX, boxY), (boxX + boxW, boxY + boxH), (23, 184, 80), 1)
                #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE, DRAWS ELLIPSE IF IN DEBUG MODE ##########
                rotation = getEllipseRotation(image, cnt)

                if [cx, cy] not in matches:
                    matches.append([cx, cy])
                    biggestCnts.append([cx, cy, rotation, cnt, boxH])
                    print("cy: ",cy, "cx: ", cx, "h: ", boxH)

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
                    if (cx1 > cx2): #WARNING CHANGED < to >
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1): #WARNING CHANGED > to <
                        continue
                #Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                #Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget, avgHeight, biggestCnts[i][3], biggestCnts[i+1][3]] not in targets:
                    targets.append([centerOfTarget, yawToTarget, avgHeight, biggestCnts[i][3], biggestCnts[i+1][3]])
                    print("adding target")
                    
    #Check if there are targets seen
    if (len(targets) > 0):
        # pushes that it sees vision target to network tables
        #shuffleBoard.putBoolean("tapeDetected", True)
        print("true")
        #Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        
        rect1 = cv2.minAreaRect(finalTarget[3])
        box1 = cv2.boxPoints(rect1)
        box1 = np.int0(box1)
        cv2.drawContours(image,[box1],0,(0,191,255),2)

        rect2 = cv2.minAreaRect(finalTarget[4])
        box2 = cv2.boxPoints(rect2)
        box2 = np.int0(box2)
        cv2.drawContours(image,[box2],0,(0,191,255),2)
        
        targetP = np.concatenate([box1,box2])
        targetPoints = np.vstack(targetP[:, :]).astype(dtype=np.float32)
        print("target Points",targetPoints)
    
        # Puts the yaw on screen
        #Draws yaw of target + line where center of target is
        #cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    #(255, 255, 255))
        #if (shuffleBoard.getBoolean("CameraDebug", False)):
        #    cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)
        #    cv2.line(image, (int(shuffleBoard.getNumber("centerOffset", 15)+finalTarget[0]), screenHeight), (int(shuffleBoard.getNumber("centerOffset", 15)+finalTarget[0]), 0), (255,255,0), 2)
        
        #currentAngleError = finalTarget[1]
        # pushes vision target angle to network tables
        #networkTable.putNumber("tapeYaw", currentAngleError)
        #shuffleBoard.putNumber("targetX", finalTarget[0])
        #shuffleBoard.putNumber("targetY", finalTarget[2])
    else:
        # pushes that it deosn't see vision target to network tables
        #shuffleBoard.putBoolean("tapeDetected", False)
        print("false")
        #shuffleBoard.putNumber("targetX", -1)

    #cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)
    return image, targetPoints

#
#
#
#


while(True):

    ret, img = vd.read()
    cv2.imshow("Video cap", img)
   
    inp = cv2.waitKey(1)
    
    if inp == 115: #If input is 's'
        break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    # If found, add object points and image points
    if ret == True:
        objpoints.append(objp)

        #Refine image points
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        #Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        print(corners)
        testcam(corners, img)
    
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    #chicken code
    # boxBlur = blurImg(img, front_green_blur)
    # threshold = threshold_video(front_lower_green, front_upper_green, boxBlur)
    # processed, tarP = findTargets(img, threshold)
    # if(tarP.size >= 8):
    #     testcam(tarP, img)

    #cv2.imshow("Video processed", processed)
        

cv2.destroyAllWindows()



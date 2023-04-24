#!/usr/bin/env python3

# Copyright (C) 2018 Christian Berger
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# sysv_ipc is needed to access the shared memory where the camera image is present.
import sysv_ipc
# numpy and cv are needed to access, modify, or display the pixels
import numpy as np
import cv2 as cv
# OD4Session is needed to send and receive messages
import OD4Session
# Import the OpenDLV Standard Message Set.
import opendlv_standard_message_set_v0_9_10_pb2

################################################################################
# This dictionary contains all distance values to be filled by function onDistance(...).
distances = { "front": 0.0, "left": 0.0, "right": 0.0, "rear": 0.0 };

################################################################################
# This callback is triggered whenever there is a new distance reading coming in.
def onDistance(msg, senderStamp, timeStamps):
    #print ("Received distance; senderStamp= %s" % (str(senderStamp)))
    #print ("sent: %s, received: %s, sample time stamps: %s" % (str(timeStamps[0]), str(timeStamps[1]), str(timeStamps[2])))
    #print ("%s" % (msg))
    if senderStamp == 0:
        distances["front"] = msg.distance
    if senderStamp == 1:
        distances["left"] = msg.distance
    if senderStamp == 2:
        distances["rear"] = msg.distance
    if senderStamp == 3:
        distances["right"] = msg.distance


# Create a session to send and receive messages from a running OD4Session;
# Replay mode: CID = 253
# Live mode: CID = 112
# TODO: Change to CID 112 when this program is used on Kiwi.
session = OD4Session.OD4Session(cid=111)
# Register a handler for a message; the following example is listening
# for messageID 1039 which represents opendlv.proxy.DistanceReading.
# Cf. here: https://github.com/chalmers-revere/opendlv.standard-message-set/blob/master/opendlv.odvd#L113-L115
messageIDDistanceReading = 1039
session.registerMessageCallback(messageIDDistanceReading, onDistance, opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_DistanceReading)
# Connect to the network session.
session.connect()
(HEIGHT, WIDTH) = (720, 1280)
################################################################################
# The following lines connect to the camera frame that resides in shared memory.
# This name must match with the name used in the h264-decoder-viewer.yml file.
name = "/tmp/img.argb"
# Obtain the keys for the shared memory and semaphores.
keySharedMemory = sysv_ipc.ftok(name, 1, True)
keySemMutex = sysv_ipc.ftok(name, 2, True)
keySemCondition = sysv_ipc.ftok(name, 3, True)
# Instantiate the SharedMemory and Semaphore objects.
shm = sysv_ipc.SharedMemory(keySharedMemory)
mutex = sysv_ipc.Semaphore(keySemCondition)
cond = sysv_ipc.Semaphore(keySemCondition)
# not inverted:
inverted = True
# for blue (right) cones
hmin, smin, vmin = 110, 89, 55
hmax, smax, vmax = 141, 255, 255
# inverted:
if inverted:
    hmin, smin, vmin = 10, 9, 162
    hmax, smax, vmax = 41, 255, 255
cv.namedWindow('image')
cv.createTrackbar('Hmin','image',0,180,lambda x: x)
cv.createTrackbar('Smin','image',0,255,lambda x: x)
cv.createTrackbar('Vmin','image',0,255,lambda x: x)
cv.createTrackbar('Hmax','image',0,180,lambda x: x)
cv.createTrackbar('Smax','image',0,255,lambda x: x)
cv.createTrackbar('Vmax','image',0,255,lambda x: x)
cv.setTrackbarPos('Hmin','image',hmin)
cv.setTrackbarPos('Smin','image',smin)
cv.setTrackbarPos('Vmin','image',vmin)
cv.setTrackbarPos('Hmax','image',hmax)
cv.setTrackbarPos('Smax','image', smax)
cv.setTrackbarPos('Vmax','image', vmax)

# set default value for MAX HSV trackbars.
def findCones(img, color):
    if inverted:
        img = cv.bitwise_not(img)
    HEIGHT, WIDTH = img.shape[:2]

    mask = np.zeros_like(img)
    # FILL the mask with white
    mask[:] = (255, 255, 255)
    cv.rectangle(mask, (0, 0), (img.shape[1], int(img.shape[0]*0.56)), (0, 0, 0), -1)

    # create mask2
    mask2 = np.zeros_like(img)
    mask2[:] = (255, 255, 255)
    pts = np.array([(0, HEIGHT), (460, HEIGHT-110), (WIDTH-500, HEIGHT-110), (WIDTH, HEIGHT)])
    cv.fillPoly(mask2, [pts], (0, 0, 0))

    # combine mask1 and mask2
    mask3 = cv.bitwise_and(mask, mask2)

    # apply mask3
    masked = cv.bitwise_and(img, mask3)

    # convert to HSV and apply threshold
    hsv = cv.cvtColor(masked, cv.COLOR_BGR2HSV)
    blurred = cv.GaussianBlur(hsv, (5, 5), 0)

    # set lower and upper bounds of blue cones
    '''hmin = cv.getTrackbarPos('Hmin','image')
    smin = cv.getTrackbarPos('Smin','image')
    vmin = cv.getTrackbarPos('Vmin','image')
    hmax = cv.getTrackbarPos('Hmax','image')
    smax = cv.getTrackbarPos('Smax','image')
    vmax = cv.getTrackbarPos('Vmax','image')'''


    bhsvLow = (hmin, smin, vmin)
    bhsvHi = (hmax, smax, vmax)

    # apply color thresholding
    blueCones = cv.inRange(blurred, bhsvLow, bhsvHi)

    # morphological operations
    iterations = 3
    kernel = np.ones((3,3), np.uint8)
    dilate = cv.dilate(blueCones, kernel, iterations=iterations)
    erode = cv.erode(dilate, kernel, iterations=iterations)
    canny = cv.Canny(erode, 30, 90)

    # find contours and filter them
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    approxContourList = [cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True) for cnt in contours]
    filteredContours = [cnt for cnt in approxContourList if 3 <= len(cnt) <= 40]
    
    # remove contours that are too small
    filteredContours = [cnt for cnt in filteredContours if cv.contourArea(cnt) > 60]
    # remove contours that are too large
    #filteredContours = [cnt for cnt in filteredContours if cv.contourArea(cnt) < 1000]
    hulls = [cv.convexHull(cnt) for cnt in filteredContours]
    # create rectangle around contours 
    rectangles = [cv.boundingRect(cnt) for cnt in filteredContours]
    # if the rectangle is wider than it is tall, remove it
    rectangles = [rect for rect in rectangles if rect[3] > rect[2]]
    # find center of rectangle
    centers = [(int(rect[0] + rect[2]/2), int(rect[1] + rect[3]/2)) for rect in rectangles]
    
    return rectangles, centers
    

################################################################################
# Main loop to process the next image frame coming in.
hasFrame = False
while True:
    # Wait for next notification.
    # wait 100 ms for n key press
    key = cv.waitKey(2)
    if key == ord('n'):
        hasFrame = False
    if not hasFrame:
        hasFrame = True
        cond.Z()

        print ("Received new frame.")

        # Lock access to shared memory.
        mutex.acquire()
        # Attach to shared memory.
        shm.attach()
        # Read shared memory into own buffer.
        buf = shm.read()
        # Detach to shared memory.
        shm.detach()
        # Unlock access to shared memory.
        mutex.release()
    
    # Turn buf into img array (1280 * 720 * 4 bytes (ARGB)) to be used with OpenCV.
    img = np.frombuffer(buf, np.uint8).reshape(720, 1280, 4)
    img = img[:, :, :3]
    if inverted:
        hmin, smin, vmin = 10, 9, 162
        hmax, smax, vmax = 41, 255, 255
    blueRect, blueCenter = findCones(img, 'blue')
    if inverted:
        hmin, smin, vmin = 80, 43, 120
        hmax, smax, vmax = 130, 255, 255
    yellowRect, yellowCenter = findCones(img, 'yellow')
    ############################################################################
    # TODO: Add some image processing logic here.
    
    # Invert colors
    # draw centers on the original image
    centersImg = img.copy()
    for center in blueCenter:
        cv.circle(centersImg, center, 2, (0,0,255), 2)
    for center in yellowCenter:
        cv.circle(centersImg, center, 2, (0,255,255), 2)
    # draw lines between centers starting from the bottom
    blueCenter.sort(key=lambda tup: tup[1])
    yellowCenter.sort(key=lambda tup: tup[1])
    for i in range(len(blueCenter)-1):
        cv.line(centersImg, blueCenter[i], blueCenter[i+1], (0,0,255), 2)
    for i in range(len(yellowCenter)-1):
        cv.line(centersImg, yellowCenter[i], yellowCenter[i+1], (0,255,255), 2)

    numCones = 0
    if len(blueCenter) > len(yellowCenter):
        numCones = len(yellowCenter)
    else:
        numCones = len(blueCenter)

    # Draw center lines between the vertical lines
    for i in range(numCones-1):
        x1, y1 = blueCenter[i]
        x2, y2 = yellowCenter[i]
        x, y = (int((x1 + x2)/2), int(((y1 + y2)/2)))   # Middle point
        print("Coords: {0}, {1}".format(x, y))

        cv.line(centersImg, (250, 250), (middlePoint[0], middlePoint[1]), (180,255,0), 2)
        #cv.circle(centersImg, (x, y), 5, (255,255,0), 3)
    
    # draw rectangles on the original image
    for rect in blueRect:
        cv.rectangle(centersImg, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
    for rect in yellowRect:
        cv.rectangle(centersImg, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)
    

    # show the final output
    cv.imshow("Output", centersImg)
    # Draw a red rectangle
    #cv.rectangle(img, (50, 50), (100, 100), (0,0,255), 2)

    # TODO: Disable the following two lines before running on Kiwi:
    #cv.imshow("org image", img);
    #cv.waitKey(2);

    ############################################################################
    # Example: Accessing the distance readings.
    print ("Front = %s" % (str(distances["front"])))
    print ("Left = %s" % (str(distances["left"])))
    print ("Right = %s" % (str(distances["right"])))
    print ("Rear = %s" % (str(distances["rear"])))

    ############################################################################
    # Example for creating and sending a message to other microservices; can
    # be removed when not needed.
    angleReading = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_AngleReading()
    angleReading.angle = 123.45

    # 1038 is the message ID for opendlv.proxy.AngleReading
    session.send(1038, angleReading.SerializeToString());

    ############################################################################
    # Steering and acceleration/decelration.
    #
    # Uncomment the following lines to steer; range: +38deg (left) .. -38deg (right).
    # Value groundSteeringRequest.groundSteering must be given in radians (DEG/180. * PI).
    #groundSteeringRequest = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_GroundSteeringRequest()
    #groundSteeringRequest.groundSteering = 0
    #session.send(1090, groundSteeringRequest.SerializeToString());

    # Uncomment the following lines to accelerate/decelerate; range: +0.25 (forward) .. -1.0 (backwards).
    # Be careful!
    #pedalPositionRequest = opendlv_standard_message_set_v0_9_10_pb2.opendlv_proxy_PedalPositionRequest()
    #pedalPositionRequest.position = 0
    #session.send(1086, pedalPositionRequest.SerializeToString());


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
from ultralytics import YOLO
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
model = YOLO("models/best.pt")
model2 = YOLO("models/best2.pt")
class_name_dict = {0: 'b',
                   1: 'y'}
while True:
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

    img = np.frombuffer(buf, np.uint8).reshape(720, 1280, 4)
    img = img[:, :, :3]
    testimg = img.copy()
    testimg2 = img.copy()
    threshold = 0.1



    results = model(img)[0]
    results2 = model2(img)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv.rectangle(testimg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.putText(testimg, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv.LINE_AA)
    
    for result in results2.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv.rectangle(testimg2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv.putText(testimg2, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow("output",testimg)
    cv.imshow("output2", testimg2)
    cv.waitKey(2)

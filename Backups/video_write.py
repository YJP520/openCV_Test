# -*- coding: utf-8 -*-
"""
Copyright: Amovlab@www.amovauto.com
Author: Amovlab
Date:2021-11-01
"""

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv.VideoWriter('vtest.avi', fourcc, 20.0, (640,480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #frame = cv.flip(frame, 0)
    # write the flipped frame
    out.write(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()

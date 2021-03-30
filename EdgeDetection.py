'''
Created on Mar 7, 2021

@authors: charlie_tharas, jason_wu
'''

import cv2
import numpy as np
import dlib
from imutils import face_utils

# access camera (change index for diff cameras)
cap = cv2.VideoCapture(0)

# preset arrays for skin tones.
# this presents some problems. could use rework.
lowerbound = np.array([0, 50, 80])
upperbound = np.array([20, 255, 255])

# initialize some variables to 0/None
lastpt = None
eucl = 0
pts = [None, None]
letters = [[]]
created = False

# blink detection setup
face_detector = dlib.get_frontal_face_detector()
face_predictor = dlib.shape_predictor("shape_pred")
ls, le = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rs, re = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
def eye_ratio(eye):
    return (np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])) / (2.0 * np.linalg.norm(eye[0]-eye[3]))

# hyperparameters
eucl_thresh = 20 # threshold for invalidating point due to euclidian distance (error reduction for skin-tone-hued objects)
word_thresh = 32 # threshold for horizontal movement meaning the user must have moved backwards to write new word
blink_thresh = 0.18  # ratio between points on eye for blink detection

while (True): 
  
    # grab frame, keys
    ret, frame = cap.read() 
    k = cv2.waitKey(5) & 0xFF
    
    # convert to HSV (hue-saturation-value color scheme)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # escape exits loop
    if k == 27: 
        break
    
    # space allows user to select bounding box for where their hand may go (must avoid face/skin-tone-hued objects)
    if k == 32:
        x, y, w, ht = cv2.selectROI(frame, False)
        created = True
      
    if created:  
        # crop to bounding box
        hsv = hsv[int(y):int(y+ht), int(x):int(x+w)]
        
        # restrict to skin only (create and apply mask)
        skinreg = cv2.inRange(hsv, lowerbound, upperbound)
        blur = cv2.blur(skinreg, (3, 3))
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
        cv2.imshow('Threshold', thresh)
    
        try:
            # grab hand contours and contour hull
            contours, h = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = max(contours, key = lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(contours)
        except:
            continue
        
        # visualization
        cv2.drawContours(frame, [contours], -1, (0, 255, 255), 2)
        cv2.drawContours(frame, [hull], -1, (255, 0, 255), 3)
        
        # grab highest point on contour hull: when air-writing, this is the tip of the pointer finger
        pt = tuple(hull[hull[:, :, 1].argmin()][0])
        
        # grab euclidian distance from previous valid point
        if lastpt is not None:
            eucl = np.linalg.norm(np.array(pt)-np.array(lastpt))
        
        # validating points & letters
        if eucl is not 0 and eucl < frame.shape[1]/eucl_thresh:
            # print(pt, lastpt, eucl, pts[-1], pts[-2]) # debug
            pts.append(pt)
            
            # blink detection
            face_rects = face_detector(frame, 0)
            for i in face_rects:
                shape = face_predictor(frame, i)
                shape = face_utils.shape_to_np(shape)
                ratio = (eye_ratio(shape[ls:le]) + eye_ratio(shape[rs:re])) / 2.0
                # print(ratio) # debug
                
                if ratio > blink_thresh:
                    letters[-1].append(pt) # SAME WORD
                else:
                    # outputting the old word as a resized image
                    blankletter=np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)*255
                    for i in range(len(letters[-1])-2):
                        cv2.line(frame, letters[-1][i], letters[-1][i+1], (0, 0, 0), 2)
                    lx, ly, lw, lh = cv2.boundingRect(np.array(letters[-1]))
                    blankletter = blankletter[int(ly):int(ly+lh), int(lx):int(lx+lw)]
                    print(blankletter.shape)
                    blankletter = cv2.resize(blankletter, (28, 28))
                    blankletter = cv2.cvtColor(blankletter, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('Output', blankletter)
                    letters.append([pt]) # NEW LETTER
                    print("GOT BLINK")



        lastpt = pt
        
        # visualization 
        cv2.circle(frame, pt, 5, (255, 255, 0), -1)
        
        # current letter visualization
        try:
            for i in range(len(letters[-1])-2):
                cv2.line(frame, letters[-1][i], letters[-1][i+1], (255, 255, 0), 2)
        except IndexError:
            pass
        
    cv2.imshow('Original', frame)
    cv2.imshow('Cropped HSV', hsv)
    
# Close the window 
cap.release() 
print("LETTER COUNT", len(letters))
cv2.waitKey(25)

# iterate through and display letters (debugging)
for i in letters:
    blank = np.ones((frame.shape[0], frame.shape[1], 3), np.uint8)*255
    for ii in range(len(i)-2):
        cv2.line(blank, i[ii], i[ii+1], (255, 255, 0), 2)
    while True:
        cv2.imshow('Blank', blank)
        k = cv2.waitKey(5) & 0xFF
        if k == 32:
            break

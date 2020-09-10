#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:51:56 2020

@author: leosold
"""

import cv2
import os
import numpy as np
import pandas as pd
from skimage.draw import line
import matplotlib.pyplot as plt
import math
import time
import re
from datetime import datetime
import pickle
# from scipy import signal

tapeWidth = 19.  # width of marker tape (mm)
tapeSpacing = 21.
# file = "test.jpg"
armP0, armP1 = (515, 571), (511, 766)  # XY format
segmentWidthRange = [19,42]


def angle(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx <= 0:
        dx = abs(dx)
        dy = -dy
    alpha = math.degrees(np.arccos(dy/np.sqrt(dy**2 + dx**2)))  # 0 - 180
    if alpha >= 90:
        alpha = alpha - 180  # -90 - 90
    return alpha


def intersection(o0, o1, p0, p1):
    x1, x2, x3, x4 = o0[1], o1[1], p0[1], p1[1]
    y1, y2, y3, y4 = o0[0], o1[0], p0[0], p1[0]
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return (py, px)


def getColor(Phsv, b, vvar):
    h, s, v = Phsv[0], Phsv[1], Phsv[2]
    color = []
    if 85 <= h <= 149 and s >= 80 and vvar < 20: color.append('blu')
    if 31 <= h <= 84 and s >= 80 and vvar < 20: color.append('grn')
    if 11 <= h <= 30 and s >= 80 and vvar < 20: color.append('yel')
    if (h >= 150 or h <= 10) and s >= 80 and vvar < 20: color.append('red')
    if 20 <= h <= 90 and s >= 80 and vvar >= 20: color.append('stp')
    if s <= 60 and v <= b and vvar < 20: color.append('blk')
    if not color:
        color.append('na')
    # if s <= 50 and -21 <= v-b <= -79: color.append('grau')
    # if s <= 50 and v-b >= -20: color.append('weiß')
    return color


def cv2label(img, text, pos, fontColor, fontScale, thickness):
    rectangle_bgr = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # lineType = 2
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    # set the text start position
    text_offset_x = pos[0]
    text_offset_y = pos[1]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 8, text_offset_y - text_height - 8))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x+4, text_offset_y-4), font, fontScale=fontScale, color=fontColor, thickness=thickness)
    return img


def analyse_image(file, outfile, tapeWidth=tapeWidth, tapeSpacing=tapeSpacing, armP0=armP0, armP1=armP1, segmentWidthRange=segmentWidthRange):
    
    t0 = time.time()
    img = cv2.imread(file)
    img2 = img.copy()  # for overplotting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #img = cv2.resize(img, (int(img.shape[1]*5/10), int(img.shape[0]*5/10)))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    height, width = img.shape[0], img.shape[1]
    badImageScore = 0
    
    blur_gray = cv2.GaussianBlur(gray,(5, 5),0)
    # edges = cv2.Canny(blur_gray,50,150,apertureSize = 3)
    
    dst=gray.copy()
    
    # First check if image is bad
    if np.mean(hsv[:,:,2]) <= 100:
        print(file + " excluded - too dark")
        return
    # print(cv2.Laplacian(img2, cv2.CV_64F).var())
    
    # colorScore=cv2.normalize( cv2.multiply(hsv[:,:,1]/1.,hsv[:,:,2]/255.),dst, norm_type=cv2.NORM_MINMAX)*255
    colorScore=cv2.normalize( hsv[:,:,1]/1.,dst, norm_type=cv2.NORM_MINMAX)*255
    # plt.imshow(img)
    
    # Nach was suche ich?
    stakeKernel = np.full((3*25,3*35),0)
    stakeKernel[25:49,35:69]=255
    # plt.imshow(stakeKernel)
    res = cv2.matchTemplate(colorScore.astype(dtype=np.uint8),stakeKernel.astype(dtype=np.uint8),cv2.TM_CCORR)
    cv2.normalize(res,res, norm_type=cv2.NORM_MINMAX)
    # plt.imshow(res)
    # plt.hist(np.ravel(res))
    
    res_th = cv2.threshold(res, 0.5, 1,cv2.THRESH_BINARY)[1].astype(dtype=np.uint8)
    # plt.imshow(res_th)
    
    # cv2.imwrite(outfile, res_th*255)
    
    # erode to kill small spots
    kernel = np.full((5,13), 1)
    res_th = cv2.erode(res_th,kernel)
    

    # Return if no colors were found
    if np.max(res_th)-np.min(res_th) <= 0.5:
        badImageScore += 100
        print(file + " excluded - no colors found")
        return
    
    # Finde Zentren
    im2, contours, hierarchy = cv2.findContours(res_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # centerPoints=list(np.empty(len(contours)))
    centerPoints = []
    # cv2.contourArea(contours)
    for i,c in enumerate(contours):
        ccx = 0
        ccy = 0
        for p in c:
            ccx += p[0][0]
            ccy += p[0][1]
        ccx = int(ccx/len(c)+ stakeKernel.shape[1]*5/10)
        ccy = int(ccy/len(c)+ stakeKernel.shape[0]*5/10)
        centerPoints.append([ccx,ccy])
    centerPoints = np.array(centerPoints)   
            # centerPoints[i]=[ccx,ccy]
        # # calculate moments for each contour
        # M = cv2.moments(c)
        # # calculate x,y coordinate of center and add half kernel size
        # cX = int(M["m10"] / M["m00"] + stakeKernel.shape[1]*5/10)
        # cY = int(M["m01"] / M["m00"] + stakeKernel.shape[0]*5/10)
        # # cv2.circle(img, (cX, cY), 5, (255, 0, 0), -1)
        # centerPoints[i]=[cX,cY]
    
    # Check if centerPoints are above armP0/P1 to exclude marker tape on arm
    if armP0 is not None and armP1 is not None:
        centerPoints = np.delete(centerPoints, centerPoints[:,1] >= np.min([armP0[1],armP1[1]]),0)

            
    # Create combination matrix for point-slopes
    slopeMatrix = np.reshape([angle(p0, p1) for p0 in centerPoints for p1 in centerPoints], (len(centerPoints), len(centerPoints)))
    # Find collinear points
    approxStakeSlope = np.median(np.ravel(slopeMatrix)[~np.isnan(np.ravel(slopeMatrix))])
    centerPointsOnStake = centerPoints[np.array([bool(abs(np.median(pslopes[~np.isnan(pslopes)])-approxStakeSlope) <= 5) for pslopes in slopeMatrix]),: ]
    # return if not enough collinear points are found
    if len(centerPointsOnStake) <= 2:
        badImageScore += 100
        print(file + " excluded - too few collinear points on stake")
        return

    # find stake by fittin 2D line through points on stake
    vx,vy,x,y = cv2.fitLine(centerPointsOnStake, cv2.DIST_L2, 0, 0.01, 0.01)
    x0, x1 = x-1000., x+1000.    # Achtung, vx und vy werden später benötigt
    y0, y1 = y-1000*vy/vx, y+1000*vy/vx
    t, stakeP0, stakeP1 = cv2.clipLine((0,0,img.shape[1], img.shape[0]), (x0,y0), (x1,y1)) # wenn t = false dann liegt die Stanfe außerhalb...
    # cv2.line(img,stakeP0,stakeP1,(0,0,255),2) # Stake line, extended to entire image
    
    # print(time.time()-t0) #2
    
    if armP0 is None or armP1 is None:
        # # Suche nach Arm: dunkel - hell - dunkel
        # ringKernel = np.full((20,80),0)
        # ringKernel[0:20,20:60]=255
        # # plt.imshow(ringKernel)
        # res = cv2.matchTemplate(gray.astype(dtype=np.uint8),ringKernel.astype(dtype=np.uint8),cv2.TM_CCORR)
        # cv2.normalize(res,res, norm_type=cv2.NORM_MINMAX)
        # plt.imshow(res)
        # plt.hist(np.ravel(res))

        gray=np.int16(gray)
        blur_gray = np.int16(blur_gray)
        
        # Gabor #! define helper Arm angle cw from vertikal
        gabor1v = cv2.getGaborKernel((160, 160), 20.0, 0.1*np.pi, 80.0, 0.3, 0, ktype=cv2.CV_32F)
        gabor1h = cv2.getGaborKernel((160, 160), 20.0, 0.6*np.pi, 80.0, 0.3, 0, ktype=cv2.CV_32F)
        gaborv = cv2.filter2D(gray, cv2.CV_32F, gabor1v, (cv2.BORDER_CONSTANT, 0))/np.sum(gabor1v)
        gaborh = cv2.filter2D(gray, cv2.CV_32F, gabor1h, (cv2.BORDER_CONSTANT, 0))/np.sum(gabor1h)
        # plt.imshow(gaborv-gaborh)
        # plt.imshow(img)
        # plt.hist(np.ravel(gaborv-gaborh))
        
        # apply threshold to derive mask, #! define threshold
        res_th = cv2.threshold(gaborv-gaborh, 80, 1,cv2.THRESH_BINARY)[1].astype(dtype=np.uint8)
        # plt.imshow(res_th)
        # https://stackoverflow.com/questions/32430393/having-difficulties-detecting-small-objects-in-noisy-background-any-ways-to-fix
        # https://answers.opencv.org/question/187251/algorithm-recommendation-for-texture-analysissegmentation/
        # https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
        
        
        # skeletonize
        skel = np.zeros(res_th.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(res_th, cv2.MORPH_OPEN, element, (cv2.BORDER_CONSTANT, 0))
            #Step 3: Substract open from the original image
            temp = cv2.subtract(res_th, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(res_th, element, (cv2.BORDER_CONSTANT, 0))
            skel = cv2.bitwise_or(skel,temp)
            res_th = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(res_th)==0:
                break
        # plt.imshow(skel)
        skel = cv2.dilate(skel, element, (cv2.BORDER_CONSTANT, 0))
        
        # remove top and bottom
        skel[0:int(height*0.5),:]=0
        skel[-35:,:]=0
        
        # print(time.time()-t0)
        
        # Do the hough transform
        lines = cv2.HoughLines(skel, 1, np.pi / 180, 50, None, 0, 0)
        # if lines is not None:
        #     for i in range(0, len(lines)):
        #         rho = lines[i][0][0]
        #         theta = lines[i][0][1]
        #         a = math.cos(theta)
        #         b = math.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        #         cv2.line(img2, pt1, pt2, (255,0,255), 1)
        
        # only get best Arm line
        rho, theta = lines[0][0][0], lines[1][0][1]
        a, b = math.cos(theta), math.sin(theta)
        x0, y0 = a * rho, b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        # cv2.line(img2, pt1, pt2, (255,0,0), 2)
        # plt.imshow(img2)
        
        # clip Arm line to bottom half of image !maybe higher?
        t, armP0, armP1 = cv2.clipLine((0,int(height*0.5),width, int(height*0.5)), pt1, pt2) # wenn t = false dann liegt die Stanfe außerhalb...
        # t, armP0, armP1 = cv2.clipLine((int(height*0.5),0,height, width), pt1, pt2) # wenn t = false dann liegt die Stanfe außerhalb...
        # cv2.line(img2, armP0, armP1, (255,255,255), 2)
        # plt.imshow(img2)
        
        
        
        grayVariability = abs(cv2.GaussianBlur(gray,(5, 41),0) - blur_gray)
        grayVariability2 = cv2.GaussianBlur(grayVariability,(11, 11),0)
        # plt.imshow(grayVariability2)
        
        # extract Arm pixels
        discreteArm = line(*armP0, *armP1)
        # discreteArm = np.array(list(zip(*line(*armP0, *armP1))))
        # extraction: gray[np.array([discreteArm[1],discreteArm[0]]).T]
        score = [np.median(grayVariability2[discreteArm[1],discreteArm[0]+shift]) for shift in range(-100,100)]
        # plt.plot(score)
        # plt.hist(score)
        
        # thresholding to find "good" shifts, ! assuming that it doesnt get better left and right
        threshold,t = cv2.threshold(np.uint8(score),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        armLR = np.array(range(-100,100))[np.ravel(cv2.boxFilter(np.array(score),0,(1,20),0)<=threshold)][[0,-1]] # finds first and last occurence of values below threshold
        
        # How does the arm tip look like? !fixed size of ring
        tipKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))[0:11,10:20]
        res = cv2.matchTemplate(gray.astype(dtype=np.uint8),np.uint8(tipKernel*255),cv2.TM_CCOEFF)
        res = np.append(res,np.zeros((tipKernel.shape[0]-1,res.shape[1])), axis=0) #resize to match original proportions
        res = np.append(res,np.zeros((res.shape[0], tipKernel.shape[1]-1,)), axis=1)
        # ringMatch = np.zeros(gray.shape, dtype=res.dtype) # more straightforward but not so good
        # ringMatch[:res.shape[0], :res.shape[1]] = res
        # plt.imshow(res)
        # plt.imshow(img)
        threshold,t = cv2.threshold(res/np.max(res),0.5,255,cv2.THRESH_BINARY)
        # plt.imshow(t)
        
        # Find ring as best match on the arm
        # step through pixel rows of arm and find maximum of row maxima
        ringMatchArm = [np.max(res[row, discreteArm[0][i]+armLR[0]:discreteArm[0][i]+armLR[1]]) for i, row in enumerate(discreteArm[1])]
        ringMatchMax = np.argmax(ringMatchArm) #return index of max
        ringMatchPix = (discreteArm[0][ringMatchMax]+int(np.mean(armLR)) , discreteArm[1][ringMatchMax])
        # cv2.circle(img2, ringMatchPix, 5, (0, 255, 0), -1)
    
    else: # If armP0 or armP1 are given by definition
        if armP0[1] < armP1[1]:
            ringMatchPix = armP0
        else:
            ringMatchPix = armP1
        armLR = [0,0]
        
    # Find bottom of stake: intersection of stake line and normal projection of ringMatch (normal to arm, projected onto stake line)
    stakeBottom = intersection(stakeP0, stakeP1, ringMatchPix, (ringMatchPix[0]+1, ringMatchPix[1]-(armP0[0]-armP1[0])/(armP0[1]-armP1[1])))
    
    # # Abstand P3 von P1-P2
    # p1=np.array([0,0])
    # p2=np.array([10,0])
    # p3=np.array([0,100])
    # np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    
    #expand stake by 5px to left and right, 4 points clockwise
    rx0, rx1, rx2, rx3 = stakeP0[0]-20*vy, stakeP1[0]-20*vy, stakeP1[0]+20*vy, stakeP0[0]+20*vy
    ry0, ry1, ry2, ry3 = stakeP0[1]+20*vx, stakeP1[1]+20*vx, stakeP1[1]-20*vx, stakeP0[1]-20*vx
    stakeImgLength = math.sqrt((stakeP0[0]-stakeP1[0])**2 + (stakeP0[1]-stakeP1[1])**2)
    M = cv2.getAffineTransform(np.float32([(rx0, ry0), (rx1, ry1), (rx2, ry2)]), np.float32([(0,0), (0,stakeImgLength), (40,stakeImgLength)]))
    stakeImg = cv2.warpAffine(img, M, (width, height))[0:int(stakeImgLength),0:40] # bilinear interpolation
    # plt.imshow(stakeImg)
    # plt.imshow(img)
    # cv2.imwrite("out.jpg", stakeImg)
    
    # Translate stake bottom point to new coordinates
    sI_stakeBottom = M.dot(np.array((*stakeBottom,1)))
    
    # calculate and sum up edges (laplacian) on s and v image channel, create 1d array with edge peaks
    stakeImgHSV = cv2.cvtColor(stakeImg, cv2.COLOR_BGR2HSV)
    del_stakeImg = stakeImg.copy()
    
    sI_colorKernel = np.array([-1.,-1,-1,-1,-1,0,1, 1, 1, 1,1]).reshape((11,1))/11.
    sI_edgesS = cv2.filter2D(stakeImgHSV[:,:,1], cv2.CV_32F, sI_colorKernel, (cv2.BORDER_CONSTANT, 0))
    sI_edgesV = cv2.filter2D(stakeImgHSV[:,:,2], cv2.CV_32F, sI_colorKernel, (cv2.BORDER_CONSTANT, 0))
    
    # plt.imshow(abs(sI_edgesV))
    # plt.imshow(np.abs(sI_edgesS)+7*np.abs(sI_edgesV))
    sI_edgesScore = np.mean(np.abs(sI_edgesS)+7*np.abs(sI_edgesV), axis=1)
    # sI_edgesScore = cv2.boxFilter(sI_edgesScore,-1,(1,15))
    # plt.plot(sI_edgesScore)

    
    
    # plt.plot(np.mean(stakeImgHSV[:,:,0], axis=1))
    # plt.plot(np.mean(stakeImgHSV[:,:,1], axis=1))
    # plt.plot(np.mean(stakeImgHSV[:,:,2]-sI_baseV, axis=1))
    # plt.imshow(stakeImgCrop)
    # cv2.imwrite("out.jpg", stakeImgCrop)
    
    # # find highest values in +-15rows
    # temp = []
    # for i,val in enumerate(sI_edgesScore[15:-15]):
    #     if val == np.max(sI_edgesScore[i+15-15:i+15+15]):
    #         temp.append(i+15)
    # sI_edgesPeaks = np.array(temp)
    
    # local maxima code
    temp = np.squeeze(cv2.GaussianBlur(sI_edgesScore,(1,35),5))-np.squeeze(cv2.GaussianBlur(sI_edgesScore,(1,55),15))
    # plt.plot(temp)
    # np.diff(np.sign(np.diff(np.squeeze(cv2.GaussianBlur(kk,(1,35),7))))) == 2
    sI_edgesPeaks = np.array(np.where(np.diff(np.sign(np.diff(temp))) == -2))+1
    # filter out low-contrast-edges
    sI_edgesPeaks = sI_edgesPeaks[temp[sI_edgesPeaks] >= 10]
    # plt.scatter(sI_edgesPeaks,np.full(len(sI_edgesPeaks),20))
    # plt.clf()

    
    # make 1d array from HSV stake and cut away stake-in-ice part
    # measures are in pixels from top to bottom here (?)
    # stakeImgCrop = stakeImg[:int(sI_stakeBottom[1]),:,:]
    # sI_stakeImgHSV1d = cv2.resize(stakeImgHSV,(1, int(stakeImgHSV.shape[0])))
    sI_stakeImgHSV1d = cv2.resize(stakeImgHSV[:int(sI_stakeBottom[1]), :, :],(1, int(stakeImgHSV[:int(sI_stakeBottom[1]), :, :].shape[0])))
    sI_stakeImgBGR1d = cv2.resize(stakeImg[:int(sI_stakeBottom[1]), :, :],(1, int(stakeImg[:int(sI_stakeBottom[1]), :, :].shape[0])))

    sI_edgesPeaks = sI_edgesPeaks[sI_edgesPeaks <= sI_stakeBottom[1]]
    if len(sI_edgesPeaks) <= 2:
        print(file + "excluded - too few marker edges found on stake")
        return
    if abs(sI_edgesPeaks[-1]-sI_stakeBottom[1]) <= 5: sI_edgesPeaks = sI_edgesPeaks[:-1] # remove lowest peak if it appx. coincides with lower stake end
    if len(sI_edgesPeaks) <= 2: # again, maybe point was removes
        print(file + "excluded - too few marker edges found on stake")
        return
    
    print(file)
    # print("peaks")
    # print(sI_edgesPeaks)
    if file == "./testfolder/2020-06-19_10-04.jpg":
        print(" ")
    
    sI_baseV = np.percentile(sI_stakeImgHSV1d[:,:,2], 25)
    # plt.plot(sI_stakeImgHSV1d[:,:,0])
    # plt.plot(sI_stakeImgHSV1d[:,:,1])
    # plt.plot(sI_stakeImgHSV1d[:,:,2]-sI_baseV)
    
    # timage = cv2.cvtColor(cv2.resize(sI_stakeImgHSV1d,(50, int(sI_stakeImgHSV1d.shape[0]))), cv2.COLOR_HSV2RGB)
    # plt.imshow(timage)
    for x in sI_edgesPeaks: cv2.circle(del_stakeImg, (25, x), 5, (0, 255, 0), -1)
    # plt.imshow(del_stakeImg)
    
    sI_segments = pd.DataFrame(columns=['pxheight','pxwidth','color'])

    # sI_segments = [[],[],[]]
    for i, segment in enumerate(sI_edgesPeaks[:-1]):
        temp = [sI_edgesPeaks[i],sI_edgesPeaks[i+1]-1] # sI_edgesPeaks measure from stake top end
        Pgrb = np.mean(sI_stakeImgBGR1d[temp[0]:temp[1]+1,:,:], axis=0)
        Phsv = cv2.cvtColor(np.uint8(np.reshape(Pgrb, (1,1,3))), cv2.COLOR_BGR2HSV)
        vvar = np.subtract(*np.percentile(sI_stakeImgHSV1d[temp[0]:temp[1]+1,:,2], [75, 25]))  # Brightness IQR as robust variability measure
        print(Phsv,vvar)
        # print(temp, Phsv, vvar, getColor(np.ravel(Phsv), sI_baseV, vvar))
        tt = pd.DataFrame([[sI_stakeBottom[1]-sI_edgesPeaks[i+1],  # pixelheight of lower segment bound from BOTTOM
                           sI_edgesPeaks[i+1]-sI_edgesPeaks[i],  # pixel width of segment
                           getColor(np.ravel(Phsv), sI_baseV, vvar)[0]]], index = [i], columns=['pxheight','pxwidth','color'])
        sI_segments = sI_segments.append(tt)
        # sI_segments[0].append(sI_stakeBottom[1]-sI_edgesPeaks[i+1])  # pixelheight of lower segment bound from BOTTOM
        # sI_segments[1].append(sI_edgesPeaks[i+1]-sI_edgesPeaks[i])  # pixel width of segment
        # sI_segments[2].append(getColor(np.ravel(Phsv), sI_baseV, vvar)[0])

    # Remove segments that are too thin or too wide
    sI_segments = sI_segments[(sI_segments['pxwidth'] >= segmentWidthRange[0]) & (sI_segments['pxwidth'] <= segmentWidthRange[1])]

    # plt.scatter(sI_segments[0][0::2], sI_segments[1][0::2])
    # print("segments")
    # print(sI_segments)
    
    # find out if markers are odd or even segments
    ii = np.where(sI_segments['color']!='na')
    # check if colors were  found
    if np.squeeze(ii).size == 0:
        print(file + " excluded - no colors spotted")
        return
    
    sI_segmentsCol = np.array(np.squeeze(ii)) #! streifen aussortieren?
    sI_segmentsColOffset = sI_segmentsCol.reshape(-1)[0] % 2
    
    # check if this holds for ALL colored markers (Verzählt?)
    if not np.all(sI_segmentsCol % 2 == sI_segmentsColOffset):
        print(file + " excluded - odd/even pattern interruptet")
        return #raise Exception('ERROR: segment edtection failed')
    
    # define markers and spacing width
    tt = np.full(len(sI_segments), tapeSpacing)
    tt[sI_segmentsColOffset::2] = tapeWidth
    sI_segments['mmwidth']=tt
    
    # regression with vertical midpoints as X and mm-per-px as Y
    coef = np.polyfit(np.array(sI_segments['pxheight'])+np.array(sI_segments['pxwidth'], dtype=float)*0.5,  np.array(sI_segments['mmwidth'], dtype=float)/np.array(sI_segments['pxwidth'], dtype=float),1)
    sI_scaleFun = np.poly1d(coef) 
    
    # plot(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5,  np.array(sI_segments[3])/np.array(sI_segments[1]), 'yo', np.array(sI_segments[0])+np.array(sI_segments[1])*0.5, sI_scaleFun(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5), '--k')
    
    # calculate distance of lower marker bounds to lower stake end in mm
    sI_segments['mmheight'] = sI_segments['pxheight'] * sI_scaleFun(np.array(sI_segments['pxheight'])*0.5)
    # sI_segments


    # # assign chunk number (if stake is partly hidden)
    chunk = np.ones(len(sI_segments))
    ttt = np.insert(np.abs(np.diff(sI_segments['pxheight'])),0,0)
    sss = np.array(sI_segments['pxwidth'])
    for i, t in enumerate(ttt):
        if abs(t - sss[i]) >= 1:
            chunk[i+1:] += 1
    sI_segments['chunk'] = np.int8(chunk)
    # for r in sI_segments

    # remove spaces between markers from list (not helpful for matching)
    sI_segments_ret = sI_segments[sI_segmentsColOffset::2]

    # store coordinates for overplotting and output on img2
    iM = cv2.invertAffineTransform(M)

    out_edgesPeaks = [iM.dot(np.array([int(stakeImg.shape[1]/2),y,1])) for y in sI_edgesPeaks]
    for xy in out_edgesPeaks:
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.circle(img2, xxyy, 3, (255, 0, 0), -1)
    # segments with labels
    out_segmentsXY = [iM.dot(np.array([int(stakeImg.shape[1]),sI_stakeBottom[1]-y,1])) for y in sI_segments['pxheight']]
    tt_colors = np.array(sI_segments['color'])
    tt_height = np.array(sI_segments['mmheight'])
    for i, xy in enumerate(out_segmentsXY):
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.line(img2, xxyy, (xxyy[0]+50,xxyy[1]), (0,255,0), 1)
        cv2label(img2, str(tt_colors[i])+ ": " +str(np.format_float_positional(tt_height[i], precision=1))+"mm", (xxyy[0]+50,xxyy[1]), (0,255,0), 0.5, 1)
    # arm, stakebottom, ringmatch
    cv2.line(img2, (armP0[0]+int(np.mean(armLR)),armP0[1]), (armP1[0]+int(np.mean(armLR)),armP1[1]), (0,255,0), 1)
    cv2.circle(img2, ringMatchPix, 3, (0, 0, 255), -1)
    cv2.circle(img2, tuple(np.int_(np.rint(stakeBottom))), 3, (0, 255, 0), -1)
    # plt.imshow(img2)
    
    # stake line and arm line: stakeP0, stakeP1, armP0+..., armP1+mean(armLR)
    # ring match: ringMatchPix
    # stake bottom: stakeBottom
    # stake center points: centerPoints, centerPointsOnStake (außenliegende aussortiert)
    
    cv2.imwrite(outfile, img2)
    return sI_segments_ret

#######################################################
# Function call
folder = "./testfolder"
outfolder = "./testfolder_out"
results = {}
for file in sorted(os.listdir(folder)):
    if file.endswith(".jpg"):
        # print(os.path.join(folder, file))
        
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', file)
        timestamp = datetime.strptime(match.group(), '%Y-%m-%d_%H-%M')#.date()
        # print(timestamp)
        results[timestamp] = analyse_image(os.path.join(folder, file), os.path.join(outfolder, 'out_'+file), tapeWidth=tapeWidth, tapeSpacing=tapeSpacing, armP0=armP0, armP1=armP1, segmentWidthRange=segmentWidthRange)
# Saving the objects:
with open('ASDA-results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(results, f)


print("Ende?")
print("echt")
# lines = cv2.HoughLinesP(skel, 1, np.pi / 180, 50, None, 100, 50)
# if lines is not None:
#     for i in range(0, len(lines)):
#         l = lines[i][0]
#         cv2.line(img2, (l[0], l[1]), (l[2], l[3]), (255,0,255), 2)

# # find line through points
# centerPoints = np.asarray(np.where(skel == 1)).T[::10] # list of points
# slopeMatrix = np.reshape([angle(p0, p1) for p0 in centerPoints for p1 in centerPoints], (len(centerPoints), len(centerPoints)))
# # Find collinear points
# approxStakeSlope = np.median(np.ravel(slopeMatrix)[~np.isnan(np.ravel(slopeMatrix))])
# centerPointsOnStake = np.array(centerPoints)[np.array([bool(abs(np.median(pslopes[~np.isnan(pslopes)])-approxStakeSlope) <= 10) for pslopes in slopeMatrix]) ]

# #---
# b=cv2.threshold(a, 75, 255,cv2.THRESH_BINARY)[1]
# plt.imshow(b)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# c = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
# plt.imshow(c)
# d=cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel)
# plt.imshow(img_crop)

# d=d.astype(dtype=np.uint8)
# lines = cv2.HoughLines(cv2.threshold(d, 127, 1,cv2.THRESH_BINARY)[1], 1, np.pi / 180, 300, None, 0, 0)
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         aa = math.cos(theta)
#         bb = math.sin(theta)
#         x0 = aa * rho
#         y0 = bb * rho
#         pt1 = (int(x0 + 1000*(-bb)), int(y0 + 1000*(aa)))
#         pt2 = (int(x0 - 1000*(-bb)), int(y0 - 1000*(aa)))
#         cv2.line(d, pt1, pt2, 255, 3)
# plt.imshow(d)
# #plt.hist(np.ravel(ringKernel))
# #max(np.ravel(a))

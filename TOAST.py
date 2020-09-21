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


tapeWidth = 19.  # width of marker tape (mm)
tapeSpacing = 21.
armP0, armP1 = (515, 571), (511, 766)  # XY format
segmentWidthRange = [19, 42]

minBrightness = 100
maxDegCollin = 5 # maximum angle between points to be comsidered as collinear
minCollinPoints = 2 # minimum number of collinear points on stake

def angle(p0, p1):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx <= 0:
        dx = abs(dx)
        dy = -dy
    alpha = math.degrees(np.arccos(dy / np.sqrt(dy ** 2 + dx ** 2)))  # 0 - 180
    if alpha >= 90:
        alpha = alpha - 180  # -90 - 90
    return alpha


def intersection(o0, o1, p0, p1):
    x1, x2, x3, x4 = o0[1], o1[1], p0[1], p1[1]
    y1, y2, y3, y4 = o0[0], o1[0], p0[0], p1[0]
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return (py, px)


def getColor(Phsv, b, vvar):
    # GRY and WHT not implemented!
    h, s, v = Phsv[0], Phsv[1], Phsv[2]
    color = []
    if 85 <= h <= 149 and s >= 95 and vvar < 20: color.append('blu')
    if 31 <= h <= 84 and s >= 95 and vvar < 20: color.append('grn')
    if 11 <= h <= 30 and s >= 95 and vvar < 20: color.append('yel')
    if (h >= 150 or h <= 10) and s >= 95 and vvar < 20: color.append('red')
    if 20 <= h <= 90 and s >= 95 and vvar >= 20: color.append('stp')
    if s <= 40 and v <= b and vvar < 20: color.append('blk')
    if not color:
        color.append('na')
    return color


def cv2label(img, text, pos, fontColor, fontScale, thickness):
    # plot labels on image
    rectangle_bgr = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
    # set the text start position
    text_offset_x = pos[0]
    text_offset_y = pos[1]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 8, text_offset_y - text_height - 8))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x + 4, text_offset_y - 4), font, fontScale=fontScale, color=fontColor,
                thickness=thickness)
    return img

#######################################################

def analyse_image(file, outfile, tapeWidth=tapeWidth, tapeSpacing=tapeSpacing, armP0=armP0, armP1=armP1,
                  segmentWidthRange=segmentWidthRange):
    img = cv2.imread(file)
    img2 = img.copy()  # for overplotting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = gray.copy()

    height, width = img.shape[0], img.shape[1]

    # First check if image is too dark
    if np.mean(hsv[:, :, 2]) <= minBrightness:
        print(file + " excluded - too dark")
        return

    # Get normalized saturation
    colorScore = cv2.normalize(hsv[:, :, 1] / 1., dst, norm_type=cv2.NORM_MINMAX) * 255

    # Match template: shape of marker (is this actually a good idea?)
    stakeKernel = np.full((3 * 25, 3 * 35), 0)
    stakeKernel[25:49, 35:69] = 255
    # plt.imshow(stakeKernel)
    res = cv2.matchTemplate(colorScore.astype(dtype=np.uint8), stakeKernel.astype(dtype=np.uint8), cv2.TM_CCORR)
    cv2.normalize(res, res, norm_type=cv2.NORM_MINMAX)
    # plt.imshow(res)

    # threshold on marker-match
    res_th = cv2.threshold(res, 0.5, 1, cv2.THRESH_BINARY)[1].astype(dtype=np.uint8)
    # plt.imshow(res_th)

    # erode to kill small spots
    kernel = np.full((5, 13), 1)
    res_th = cv2.erode(res_th, kernel)

    # Return if no colors were found
    # shape of kernel creates rather circular patterns
    if np.max(res_th) - np.min(res_th) <= 0.5:
        print(file + " excluded - no colors found")
        return

    # find center points
    contours, hierarchy = cv2.findContours(res_th, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_NONE)
    # centerPoints=list(np.empty(len(contours)))
    centerPoints = []
    # cv2.contourArea(contours)
    for i, c in enumerate(contours):
        ccx = 0
        ccy = 0
        for p in c:
            ccx += p[0][0]
            ccy += p[0][1]
        ccx = int(ccx / len(c) + stakeKernel.shape[1] * 5 / 10)
        ccy = int(ccy / len(c) + stakeKernel.shape[0] * 5 / 10)
        centerPoints.append([ccx, ccy])
    centerPoints = np.array(centerPoints)

    # Check if centerPoints are above armP0/P1 to exclude marker tape on arm
    centerPoints = np.delete(centerPoints, centerPoints[:, 1] >= np.min([armP0[1], armP1[1]]), 0)

    # Create combination matrix for point-slopes to exclude non-colinear centerPoints
    slopeMatrix = np.reshape([angle(p0, p1) for p0 in centerPoints for p1 in centerPoints],
                             (len(centerPoints), len(centerPoints)))
    # Find collinear points
    approxStakeSlope = np.median(np.ravel(slopeMatrix)[~np.isnan(np.ravel(slopeMatrix))])
    centerPointsOnStake = centerPoints[np.array(
        [bool(abs(np.median(pslopes[~np.isnan(pslopes)]) - approxStakeSlope) <= maxDegCollin) for pslopes in slopeMatrix]), :]
    # return if not enough collinear points are found
    if len(centerPointsOnStake) <= minCollinPoints:
        print(file + " excluded - too few collinear points on stake")
        return

    # find stake by fittin 2D line through points on stake
    vx, vy, x, y = cv2.fitLine(centerPointsOnStake, cv2.DIST_L2, 0, 0.01, 0.01)
    x0, x1 = x - 1000., x + 1000.  # Warning! vx and vy will be used later
    y0, y1 = y - 1000 * vy / vx, y + 1000 * vy / vx
    t, stakeP0, stakeP1 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), (x0, y0),
                                       (x1, y1))
    # cv2.line(img,stakeP0,stakeP1,(0,0,255),2) # Stake line, extended to entire image

    if armP0[1] < armP1[1]:
        ringMatchPix = armP0
    else:
        ringMatchPix = armP1
    armLR = [0, 0]

    # Find bottom of stake: intersection of stake line and normal projection of ringMatch (normal to arm, projected onto stake line)
    stakeBottom = intersection(stakeP0, stakeP1, ringMatchPix,
                               (ringMatchPix[0] + 1, ringMatchPix[1] - (armP0[0] - armP1[0]) / (armP0[1] - armP1[1])))

    # expand stake by 5px to left and right, 4 points clockwise
    rx0, rx1, rx2, rx3 = stakeP0[0] - 20 * vy, stakeP1[0] - 20 * vy, stakeP1[0] + 20 * vy, stakeP0[0] + 20 * vy
    ry0, ry1, ry2, ry3 = stakeP0[1] + 20 * vx, stakeP1[1] + 20 * vx, stakeP1[1] - 20 * vx, stakeP0[1] - 20 * vx
    stakeImgLength = math.sqrt((stakeP0[0] - stakeP1[0]) ** 2 + (stakeP0[1] - stakeP1[1]) ** 2)
    M = cv2.getAffineTransform(np.float32([(rx0, ry0), (rx1, ry1), (rx2, ry2)]),
                               np.float32([(0, 0), (0, stakeImgLength), (40, stakeImgLength)]))
    stakeImg = cv2.warpAffine(img, M, (width, height))[0:int(stakeImgLength), 0:40]  # bilinear interpolation
    # plt.imshow(stakeImg)
    # plt.imshow(img)
    # cv2.imwrite("out.jpg", stakeImg)

    # Translate stake bottom point to new coordinates
    sI_stakeBottom = M.dot(np.array((*stakeBottom, 1)))

    # calculate and sum up edges (laplacian) on s and v image channel, create 1d array with edge peaks
    stakeImgHSV = cv2.cvtColor(stakeImg, cv2.COLOR_BGR2HSV)

    sI_colorKernel = np.array([-1., -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]).reshape((11, 1)) / 11.
    sI_edgesS = cv2.filter2D(stakeImgHSV[:, :, 1], cv2.CV_32F, sI_colorKernel, (cv2.BORDER_CONSTANT, 0))
    sI_edgesV = cv2.filter2D(stakeImgHSV[:, :, 2], cv2.CV_32F, sI_colorKernel, (cv2.BORDER_CONSTANT, 0))

    # plt.imshow(abs(sI_edgesV))
    # plt.imshow(np.abs(sI_edgesS)+7*np.abs(sI_edgesV))
    sI_edgesScore = np.mean(np.abs(sI_edgesS) + 7 * np.abs(sI_edgesV), axis=1)
    # sI_edgesScore = cv2.boxFilter(sI_edgesScore,-1,(1,15))
    # plt.plot(sI_edgesScore)

    # local maxima code
    temp = np.squeeze(cv2.GaussianBlur(sI_edgesScore, (1, 35), 5)) - np.squeeze(
        cv2.GaussianBlur(sI_edgesScore, (1, 55), 15))
    # plt.plot(temp)
    # np.diff(np.sign(np.diff(np.squeeze(cv2.GaussianBlur(kk,(1,35),7))))) == 2
    sI_edgesPeaks = np.array(np.where(np.diff(np.sign(np.diff(temp))) == -2)) + 1
    # filter out low-contrast-edges
    sI_edgesPeaks = sI_edgesPeaks[temp[sI_edgesPeaks] >= 10]
    # plt.scatter(sI_edgesPeaks,np.full(len(sI_edgesPeaks),20))
    # plt.clf()

    # make 1d array from HSV stake and cut away stake-in-ice part
    # !!! measures switch from from-top to from-bottom
    sI_stakeImgHSV1d = cv2.resize(stakeImgHSV[:int(sI_stakeBottom[1]), :, :],
                                  (1, int(stakeImgHSV[:int(sI_stakeBottom[1]), :, :].shape[0])))
    sI_stakeImgBGR1d = cv2.resize(stakeImg[:int(sI_stakeBottom[1]), :, :],
                                  (1, int(stakeImg[:int(sI_stakeBottom[1]), :, :].shape[0])))

    sI_edgesPeaks = sI_edgesPeaks[sI_edgesPeaks <= sI_stakeBottom[1]]
    if len(sI_edgesPeaks) <= 2:
        print(file + "excluded - too few marker edges found on stake")
        return
    if abs(sI_edgesPeaks[-1] - sI_stakeBottom[1]) <= 5: sI_edgesPeaks = sI_edgesPeaks[
                                                                        :-1]  # remove lowest peak if it appx. coincides with lower stake end (shadow problems)
    if len(sI_edgesPeaks) <= 2:  # again, maybe point was removes
        print(file + "excluded - too few marker edges found on stake")
        return

    # print(file)
    # if file == "./InputImages/2020-06-19_10-04.jpg":
    #     print(" ")

    sI_baseV = np.percentile(sI_stakeImgHSV1d[:, :, 2], 25)  # dark base-brightness for black-detection
    sI_segments = pd.DataFrame(columns=['pxheight', 'pxwidth', 'color'])  # storage

    for i, segment in enumerate(sI_edgesPeaks[:-1]):
        temp = [sI_edgesPeaks[i], sI_edgesPeaks[i + 1] - 1]  # sI_edgesPeaks measure from stake top end
        Pgrb = np.mean(sI_stakeImgBGR1d[temp[0]:temp[1] + 1, :, :], axis=0)
        Phsv = cv2.cvtColor(np.uint8(np.reshape(Pgrb, (1, 1, 3))), cv2.COLOR_BGR2HSV)
        vvar = np.subtract(*np.percentile(sI_stakeImgHSV1d[temp[0]:temp[1] + 1, :, 2],
                                          [75, 25]))  # Brightness IQR as robust variability measure
        # print(Phsv,vvar)
        tt = pd.DataFrame([[sI_stakeBottom[1] - sI_edgesPeaks[i + 1],  # pixelheight of lower segment bound from BOTTOM
                            sI_edgesPeaks[i + 1] - sI_edgesPeaks[i],  # pixel width of segment
                            getColor(np.ravel(Phsv), sI_baseV, vvar)[0]]], index=[i],
                          columns=['pxheight', 'pxwidth', 'color'])
        sI_segments = sI_segments.append(tt)

    # Remove segments that are too thin or too wide
    sI_segments = sI_segments[
        (sI_segments['pxwidth'] >= segmentWidthRange[0]) & (sI_segments['pxwidth'] <= segmentWidthRange[1])]
    # plt.scatter(sI_segments[0][0::2], sI_segments[1][0::2])

    # find out which segments are actually markers
    ii = np.where(sI_segments['color'] != 'na')
    # check if colors were  found
    if np.squeeze(ii).size == 0:
        print(file + " excluded - no colors spotted")
        return
    # check if markers are odd or even segments
    sI_segmentsCol = np.array(np.squeeze(ii))  # ! streifen aussortieren?
    sI_segmentsColOffset = sI_segmentsCol.reshape(-1)[0] % 2

    # check if this holds for ALL colored markers (Verzählt?)
    if not np.all(sI_segmentsCol % 2 == sI_segmentsColOffset):
        print(file + " excluded - odd/even pattern interruptet")
        return  # raise Exception('ERROR: segment edtection failed')

    # define markers and spacing width
    tt = np.full(len(sI_segments), tapeSpacing)
    tt[sI_segmentsColOffset::2] = tapeWidth
    sI_segments['mmwidth'] = tt

    # GET SCALE: linear (!) regression: vertical midpoints as X and mm-per-px as Y
    coef = np.polyfit(np.array(sI_segments['pxheight']) + np.array(sI_segments['pxwidth'], dtype=float) * 0.5,
                      np.array(sI_segments['mmwidth'], dtype=float) / np.array(sI_segments['pxwidth'], dtype=float), 1)
    sI_scaleFun = np.poly1d(coef)

    # plot(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5,  np.array(sI_segments[3])/np.array(sI_segments[1]), 'yo', np.array(sI_segments[0])+np.array(sI_segments[1])*0.5, sI_scaleFun(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5), '--k')

    # calculate distance of lower marker bounds to lower stake end in mm
    sI_segments['mmheight'] = sI_segments['pxheight'] * sI_scaleFun(np.array(sI_segments['pxheight']) * 0.5)

    # assign chunk number (if stake is partly hidden by drops on lens)
    chunk = np.ones(len(sI_segments))
    ttt = np.insert(np.abs(np.diff(sI_segments['pxheight'])), 0, 0)
    sss = np.array(sI_segments['pxwidth'])
    ttt[0] = sss[0]
    for i, t in enumerate(ttt):
        if abs(t - sss[i]) >= 1:
            chunk[i + 1:] += 1
    sI_segments['chunk'] = np.int8(chunk)
    print('number of chunks: '+str(np.max(chunk)))
    # print(chunk)

    # remove spaces between markers from list (not helpful for matching)
    sI_segments_ret = sI_segments[sI_segmentsColOffset::2]

    # store coordinates for overplotting and output on img
    iM = cv2.invertAffineTransform(M) # invert transform matrix to map coordinates back on image
    out_edgesPeaks = [iM.dot(np.array([int(stakeImg.shape[1] / 2), y, 1])) for y in sI_edgesPeaks]
    for xy in out_edgesPeaks:
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.circle(img2, xxyy, 3, (255, 0, 0), -1)
    # segments with labels
    out_segmentsXY = [iM.dot(np.array([int(stakeImg.shape[1]), sI_stakeBottom[1] - y, 1])) for y in
                      sI_segments['pxheight']]
    tt_colors = np.array(sI_segments['color'])
    tt_height = np.array(sI_segments['mmheight'])
    for i, xy in enumerate(out_segmentsXY):
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.line(img2, xxyy, (xxyy[0] + 50, xxyy[1]), (0, 255, 0), 1)
        cv2label(img2, str(tt_colors[i]) + ": " + str(np.format_float_positional(tt_height[i], precision=1)) + "mm",
                 (xxyy[0] + 50, xxyy[1]), (0, 255, 0), 0.5, 1)
    cv2.line(img2, (armP0[0] + int(np.mean(armLR)), armP0[1]), (armP1[0] + int(np.mean(armLR)), armP1[1]), (0, 255, 0),
             1)
    cv2.circle(img2, ringMatchPix, 3, (0, 0, 255), -1)
    cv2.circle(img2, tuple(np.int_(np.rint(stakeBottom))), 3, (0, 255, 0), -1)
    cv2.imwrite(outfile, img2)
    print(file + " probably     ok")
    return sI_segments_ret


#######################################################
# function call

folder = "./InputImages"
outfolder = "./OutputImages"

# Clean output directory before writing new images to disk
filelist = [ f for f in os.listdir(outfolder) if f.endswith(".jpg") ]
for f in filelist:
    os.remove(os.path.join(outfolder, f))

results = {}
for file in sorted(os.listdir(folder)):
    if file.endswith(".jpg"):
        match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', file)
        timestamp = datetime.strptime(match.group(), '%Y-%m-%d_%H-%M')  # .date()
        # print(timestamp)
        results[timestamp] = analyse_image(os.path.join(folder, file), os.path.join(outfolder, 'out_' + file),
                                           tapeWidth=tapeWidth, tapeSpacing=tapeSpacing, armP0=armP0, armP1=armP1,
                                           segmentWidthRange=segmentWidthRange)

# Save results:
with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(results, f)

print("Dummy Ende")

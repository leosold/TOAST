#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import pandas as pd
# from skimage.draw import line
# import matplotlib.pyplot as plt
import math
import time
import re
from datetime import datetime
import pickle
# import statsmodels.formula.api as smf


tape_width = 19.  # width of marker tape (mm)
tape_spacing = 21.
arm_p0, arm_p1 = (515, 571), (511, 766)  # XY format
segment_width_range = (19, 42)
wb_rect_yyxx = (320, 500, 560, 880)

min_brightness = 100
max_deg_collin = 5  # maximum angle between points to be comsidered as collinear
min_collin_points = 2  # minimum number of collinear points on stake


def angle(p0, p1):
    """

    Parameters
    ----------
    p0 : [x, y] coordinates of point P0
    p1 : [x, y] coordinates of point P1

    Returns
    -------
    slope of line P0P1, converted to range [-90, 90]

    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx <= 0:
        dx = abs(dx)
        dy = -dy
    alpha = math.degrees(np.arccos(dy / np.sqrt(dy ** 2 + dx ** 2)))  # 0 - 180
    if alpha >= 90:
        alpha = alpha - 180  # -90 - 90
    return alpha


def intersection(o0, o1, p0, p1):
    """

    Parameters
    ----------
    o0 : line o, point 0
    o1 : line o, point 1
    p0 : line p, point 0
    p1 : line p, point 1

    Returns
    -------
    intersection coordinates [x, y] of lines o and p
    """
    x1, x2, x3, x4 = o0[1], o1[1], p0[1], p1[1]
    y1, y2, y3, y4 = o0[0], o1[0], p0[0], p1[0]
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return py, px


def get_color(Phsv, b, vvar):
    """

    Parameters
    ----------
    Phsv : color triple HSV
    b :    "dark" reference brightness 
    vvar : brightness variability

    Returns
    -------
    list of colors that match the input
    
    """

    # todo: GRY and WHT not implemented!
    h, s, v = Phsv[0], Phsv[1], Phsv[2]
    color = []
    if 85 <= h <= 149 and s >= 95 and vvar < 20:
        color.append('blu')
    if 31 <= h <= 84 and s >= 95 and vvar < 20:
        color.append('grn')
    if 11 <= h <= 30 and s >= 95 and vvar < 20:
        color.append('yel')
    if (h >= 150 or h <= 10) and s >= 95 and vvar < 20:
        color.append('red')
    if 20 <= h <= 90 and s >= 95 and vvar >= 20:
        color.append('stp')
    if s <= 40 and v <= b and vvar < 20:
        color.append('blk')
    if not color:
        color.append('na')
    return color


def odd_even_score(arr):
    odd = np.zeros(len(arr), dtype=np.int8)
    odd[::2] = 1
    even = np.ones(len(arr), dtype=np.int8)
    even[::2] = 0

    # The higher the score the more different bits from odd/even
    odd_score = np.sum(np.abs(arr - odd))
    even_score = np.sum(np.abs(arr - even))

    return np.min([odd_score, even_score])


def get_color_arr(arr):
    """

    Parameters
    ----------
    arr : HSV color array for all segments, 4th value is variability measure

    Returns
    -------
    list of colors

    """

    # find saturation threshold that gives the best odd/even-pattern
    try_sat_thres = range(255, 0, -2)
    try_sat_score = []
    for sat in try_sat_thres:
        ii = arr[:, 1] >= sat
        try_sat_score.append(odd_even_score(ii))
    sat_thres = try_sat_thres[np.argmin(try_sat_score)] # returns first occurance

    # find lower brightness threshold (for black) to give even better pattern
    try_lowval_thres = range(255, 0, -5)
    try_lowval_score = []
    for lowval in try_lowval_thres:
        ii = np.logical_or(arr[:, 2] <= lowval, arr[:, 1] >= sat_thres)
        try_lowval_score.append(odd_even_score(ii))
    low_thres = try_lowval_thres[np.argmin(try_lowval_score)]

    # find upper brightness threshold (for white) to give even better pattern
    try_upval_thres = range(low_thres, 255, 5)
    try_upval_score = []
    for upval in try_upval_thres:
        ii = np.logical_or(arr[:, 2] >= upval,
                            np.logical_or(arr[:, 2] <= low_thres, arr[:, 1] >= sat_thres))
        try_upval_score.append(odd_even_score(ii))
    up_thres = try_upval_thres[np.argmin(try_upval_score)]

    ii = np.logical_or(arr[:, 2] >= up_thres,
                            np.logical_or(arr[:, 2] <= low_thres, arr[:, 1] >= sat_thres))

    black_sat_thres = np.median(arr[ii, 1]) * 0.5
    black_val_thres = np.median(arr[~ii, 2]) * 0.5

    # todo: GRY not implemented!
    col_list = []
    for a in arr:
        h, s, v, vvar = a[0], a[1], a[2], a[3]
        color = 'na'
        if s > sat_thres or v < low_thres or v > up_thres:
            if s >= black_sat_thres * 4/3 and black_val_thres <= v <= black_val_thres * 4:
                if 20 <= h <= 90 and vvar >= 20:
                    color = 'stp'
                if vvar < 20:
                    if 85 <= h <= 149:
                        color = 'blu'
                    if 31 <= h <= 84:
                        color = 'grn'
                    if 11 <= h <= 30:
                        color = 'yel'
                    if (h >= 150 or h <= 10):
                        color = 'red'
            if s < black_sat_thres and v > black_val_thres * 4:
                color = 'wht'
            if s < black_sat_thres and v < black_val_thres:
                color = 'blk'

        col_list.append(color)
    return col_list


def cv2label(img, text, pos, fontcolor, fontscale, thickness):
    """

    Parameters
    ----------
    img :       img to plot label on
    text :      label text
    pos :       label [x, y] position
    fontcolor : (R,G,B) font color
    fontscale : font scale
    thickness : font thickness

    Returns
    -------
    img with label

    """
    rectangle_bgr = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=fontscale, thickness=1)[0]
    # set the text start position
    text_offset_x = pos[0]
    text_offset_y = pos[1]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y),
                  (text_offset_x + text_width + 8,
                   text_offset_y - text_height - 8))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x + 4, text_offset_y - 4), font,
                fontScale=fontscale, color=fontcolor, thickness=thickness)
    return img

#######################################################


def analyse_image(file, outfile, tape_width=tape_width,
                  tape_spacing=tape_spacing, arm_p0=arm_p0, arm_p1=arm_p1,
                  segment_width_range=segment_width_range, wb_rect_yyxx=wb_rect_yyxx):
    """

    Parameters
    ----------
    file :          image file
    outfile :       output image file name
    tape_width :    width of marker tape
    tape_spacing :  spacing of tape markers
    arm_p0 :        [x,y] coordinates of one Point on arm
    arm_p1 :        [x,y] of other Point on arm (one must be ring)
    segment_width_range : acceptable width range [min,max] of segments
    wb_rect_yyxx :  y and x ranges for whitebalance region

    Returns
    -------
    dataframe with details for each segment (marker)

    """
    t0 = time.time()
    img = cv2.imread(file)

    # whitebalance: greyworld assumption only in given rectangle
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_lab[wb_rect_yyxx[0]:wb_rect_yyxx[1], wb_rect_yyxx[2]:wb_rect_yyxx[3], 1])  # calculate averages for whitebalance
    avg_b = np.average(img_lab[wb_rect_yyxx[0]:wb_rect_yyxx[1], wb_rect_yyxx[2]:wb_rect_yyxx[3], 2])
    img_lab[:, :, 1] = img_lab[:, :, 1] - ((avg_a - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)  # substract on LAB color space
    img_lab[:, :, 2] = img_lab[:, :, 2] - ((avg_b - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)  # convert back to BGR

    img2 = img.copy()  # for overplotting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = gray.copy()

    height, width = img.shape[0], img.shape[1]

    # First check if image is too dark
    if np.mean(hsv[:, :, 2]) <= min_brightness:
        print(file + " excluded - too dark")
        return

    # Get normalized saturation
    color_score = cv2.normalize(hsv[:, :, 1] / 1., dst,
                               norm_type=cv2.NORM_MINMAX) * 255

    # Match template: shape of marker (is this actually a good idea?)
    stake_kernel = np.full((3 * 25, 3 * 35), 0)
    stake_kernel[25:49, 35:69] = 255
    # plt.imshow(stakeKernel)
    res = cv2.matchTemplate(color_score.astype(dtype=np.uint8),
                            stake_kernel.astype(dtype=np.uint8), cv2.TM_CCORR)
    cv2.normalize(res, res, norm_type=cv2.NORM_MINMAX)
    # plt.imshow(res)

    # threshold on marker-match
    res_th = cv2.threshold(res, 0.5, 1,
                           cv2.THRESH_BINARY)[1].astype(dtype=np.uint8)
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
    center_points = []
    # cv2.contourArea(contours)
    for i, c in enumerate(contours):
        ccx = 0
        ccy = 0
        for p in c:
            ccx += p[0][0]
            ccy += p[0][1]
        ccx = int(ccx / len(c) + stake_kernel.shape[1] * 5 / 10)
        ccy = int(ccy / len(c) + stake_kernel.shape[0] * 5 / 10)
        center_points.append([ccx, ccy])
    center_points = np.array(center_points)

    # Check if centerPoints are above arm_p0/P1 to exclude marker tape on arm
    center_points = np.delete(
        center_points, center_points[:, 1] >= np.min([arm_p0[1], arm_p1[1]]),
        0)

    # Create combination matrix for point-slopes to exclude non-colinear centerPoints
    slope_matrix = np.reshape(
        [angle(p0, p1) for p0 in center_points for p1 in center_points],
        (len(center_points), len(center_points)))

    # Find collinear points
    approx_stake_slope = np.median(np.ravel(slope_matrix)[~np.isnan(np.ravel(slope_matrix))])
    center_points_on_stake = center_points[np.array(
        [bool(abs(np.median(pslopes[~np.isnan(pslopes)]) - approx_stake_slope)
              <= max_deg_collin) for pslopes in slope_matrix]), :]
    # return if not enough collinear points are found
    if len(center_points_on_stake) <= min_collin_points:
        print(file + " excluded - too few collinear points on stake")
        return

    # find stake by fitting 2D line through points on stake
    vx, vy, x, y = cv2.fitLine(center_points_on_stake, cv2.DIST_L2, 0, 0.01, 0.01)
    x0, x1 = x - 1000., x + 1000.  # Warning! vx and vy will be used later
    y0, y1 = y - 1000 * vy / vx, y + 1000 * vy / vx
    t, stake_p0, stake_p1 = cv2.clipLine(
        (0, 0, img.shape[1], img.shape[0]), (x0, y0), (x1, y1))
    # Stake line, extended to entire image
    # cv2.line(img,stakeP0,stakeP1,(0,0,255),2)

    if arm_p0[1] < arm_p1[1]:
        ring_match_pix = arm_p0
    else:
        ring_match_pix = arm_p1
    armLR = [0, 0]

    # Find bottom of stake: intersection of stake line and normal projection
    # of ringMatch (normal to arm, projected onto stake line)
    stake_bottom = intersection(
        stake_p0, stake_p1, ring_match_pix, (ring_match_pix[0] + 1,
                                             ring_match_pix[1] -
                                             (arm_p0[0] - arm_p1[0]) /
                                             (arm_p0[1] - arm_p1[1])))

    # expand stake by 20px to left and right, 4 points clockwise
    rx0, rx1, rx2, rx3 = stake_p0[0] - 20 * vy, stake_p1[0] - 20 * vy, \
                         stake_p1[0] + 20 * vy, stake_p0[0] + 20 * vy
    ry0, ry1, ry2, ry3 = stake_p0[1] + 20 * vx, stake_p1[1] + 20 * vx, \
                         stake_p1[1] - 20 * vx, stake_p0[1] - 20 * vx

    stake_img_length = math.sqrt((stake_p0[0] - stake_p1[0]) ** 2 +
                                 (stake_p0[1] - stake_p1[1]) ** 2)
    M = cv2.getAffineTransform(
        np.float32([(rx0, ry0), (rx1, ry1), (rx2, ry2)]),
        np.float32([(0, 0), (0, stake_img_length), (40, stake_img_length)]))

    # bilinear interpolation
    stake_img = cv2.warpAffine(img, M, (width, height))[0:int(stake_img_length), 0:40]
    # plt.imshow(stake_img)
    # plt.imshow(img)
    # cv2.imwrite("out.jpg", stake_img)

    # Translate stake bottom point to new coordinates
    sI_stakeBottom = M.dot(np.array((*stake_bottom, 1)))

    # convert to HSV
    stake_img_HSV = cv2.cvtColor(stake_img, cv2.COLOR_BGR2HSV)

    # calculate and sum up edges (laplacian) on s and v image channel,
    sI_color_Kernel = np.array([-1., -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]).reshape((11, 1)) / 11.
    sI_edgesS = cv2.filter2D(stake_img_HSV[:, :, 1], cv2.CV_32F, sI_color_Kernel,
                             (cv2.BORDER_CONSTANT, 0))
    sI_edgesV = cv2.filter2D(stake_img_HSV[:, :, 2], cv2.CV_32F, sI_color_Kernel,
                             (cv2.BORDER_CONSTANT, 0))

    # plt.imshow(abs(sI_edgesV))
    # plt.imshow(np.abs(sI_edgesS)+7*np.abs(sI_edgesV))
    sI_edges_score = np.mean(np.abs(sI_edgesS) + 7 * np.abs(sI_edgesV), axis=1)
    # sI_edges_score = cv2.boxFilter(sI_edges_score,-1,(1,15))
    # plt.plot(sI_edges_score)

    # local maxima code
    temp = np.squeeze(cv2.GaussianBlur(sI_edges_score, (1, 35), 5)) - \
           np.squeeze(cv2.GaussianBlur(sI_edges_score, (1, 55), 15))
    # plt.plot(temp)
    # np.diff(np.sign(np.diff(np.squeeze(cv2.GaussianBlur(kk,(1,35),7))))) == 2
    sI_edges_peaks = np.array(np.where(np.diff(np.sign(np.diff(temp))) == -2)) + 1
    # filter out low-contrast-edges
    sI_edges_peaks = sI_edges_peaks[temp[sI_edges_peaks] >= 10]
    # plt.scatter(sI_edges_peaks,np.full(len(sI_edges_peaks),20))
    # plt.clf()

    # Standardize S and V channels
    s_val = np.max(stake_img_HSV[:int(sI_stakeBottom[1]), :, 1])
    # s_val = np.percentile(stake_img_HSV[:int(sI_stakeBottom[1]), :, 1], 80)
    v_val = np.percentile(stake_img_HSV[:int(sI_stakeBottom[1]), :, 2], 50)
    stake_img_HSV[:, :, 1] = np.clip(stake_img_HSV[:, :, 1] / s_val * 255, 0, 255).astype(int)
    stake_img_HSV[:, :, 2] = np.clip(stake_img_HSV[:, :, 2] / v_val * 150, 0, 255).astype(int)
    cv2.imwrite(os.path.join(outfolder, 'stake_'+os.path.basename(file)), cv2.cvtColor(stake_img_HSV,cv2.COLOR_HSV2BGR))

    # make 1d array from HSV stake and cut away stake-in-ice part
    # !!! measures switch from from-top to from-bottom
    sI_stakeImgHSV1d = cv2.resize(stake_img_HSV[:int(sI_stakeBottom[1]), :, :],
                                  (1, int(stake_img_HSV[:int(sI_stakeBottom[1]), :, :].shape[0])))
    sI_stakeImgBGR1d = cv2.resize(stake_img[:int(sI_stakeBottom[1]), :, :],
                                  (1, int(stake_img[:int(sI_stakeBottom[1]), :, :].shape[0])))

    sI_edges_peaks = sI_edges_peaks[sI_edges_peaks <= sI_stakeBottom[1]]
    if len(sI_edges_peaks) <= 2:
        print(file + "excluded - too few marker edges found on stake")
        return
    # remove lowest peak if it appx. coincides with lower stake end
    # (shadow problems)
    if abs(sI_edges_peaks[-1] - sI_stakeBottom[1]) <= 5: sI_edges_peaks = \
        sI_edges_peaks[:-1]
    if len(sI_edges_peaks) <= 2:  # again, because maybe a point was removed
        print(file + "excluded - too few marker edges found on stake")
        return

    # # dark base-brightness for black-detection
    # sI_baseV = np.percentile(sI_stakeImgHSV1d[:, :, 2], 25)
    # # storage
    # sI_segments = pd.DataFrame(columns=['pxheight', 'pxwidth', 'color'])
    #
    # for i, segment in enumerate(sI_edges_peaks[:-1]):
    #     # sI_edges_peaks measure from stake top end
    #     temp = [sI_edges_peaks[i], sI_edges_peaks[i + 1] - 1]
    #     Pgrb = np.mean(sI_stakeImgBGR1d[temp[0]:temp[1] + 1, :, :], axis=0)
    #     Phsv = cv2.cvtColor(np.uint8(np.reshape(Pgrb, (1, 1, 3))),
    #                         cv2.COLOR_BGR2HSV)
    #     # Brightness IQR as robust variability measure
    #     vvar = np.subtract(
    #         *np.percentile(sI_stakeImgHSV1d[temp[0]:temp[1] + 1, :, 2], [75,
    #                                                                      25]))
    #     # print(Phsv,vvar)
    #     tt = pd.DataFrame([[sI_stakeBottom[1] - sI_edges_peaks[i + 1],  # pixelheight of lower segment bound from BOTTOM
    #                         sI_edges_peaks[i + 1] - sI_edges_peaks[i],  # pixel width of segment
    #                         get_color(np.ravel(Phsv), sI_baseV, vvar)[0]]],
    #                       index=[i], columns=['pxheight', 'pxwidth', 'color'])
    #     sI_segments = sI_segments.append(tt)

    # new color detection
    sI_segments = pd.DataFrame(columns=['pxheight', 'pxwidth', 'color'])
    sI_segments_hsvV = np.zeros((len(sI_edges_peaks[:-1]),4), dtype=np.uint8)
    for i, segment in enumerate(sI_edges_peaks[:-1]):
        # sI_edges_peaks measure from stake top end
        temp = [sI_edges_peaks[i], sI_edges_peaks[i + 1] - 1]
        sI_segments_hsvV[i,0:3] = cv2.cvtColor(
            np.array(np.mean(sI_stakeImgBGR1d[temp[0]:temp[1] + 1, :, :], axis=0),
                     ndmin=3, dtype=np.uint8),cv2.COLOR_BGR2HSV)


        # Brightness IQR as robust variability measure
        sI_segments_hsvV[i,3] = np.subtract(
            *np.percentile(sI_stakeImgHSV1d[temp[0]:temp[1] + 1, :, 2], [75, 25]))

        tt = pd.DataFrame([[sI_stakeBottom[1] - sI_edges_peaks[i + 1],  # pixelheight of lower segment bound from BOTTOM
                            sI_edges_peaks[i + 1] - sI_edges_peaks[i]]],  # px width of segment
                            index=[i], columns=['pxheight', 'pxwidth'])
        sI_segments = sI_segments.append(tt)

    # color recognition
    sI_segments['color'] = get_color_arr(sI_segments_hsvV)

    # Remove segments that are too thin or too wide
    # DANGEROUS - is this a good place to sort them out?
    sI_segments = sI_segments[
        (sI_segments['pxwidth'] >= segment_width_range[0]) &
        (sI_segments['pxwidth'] <= segment_width_range[1])]

    # plt.scatter(sI_segments[0][0::2], sI_segments[1][0::2])

    # find out which segments are actually markers
    ii = np.where(sI_segments['color'] != 'na')
    # check if colors were  found
    if np.squeeze(ii).size == 0:
        print(file + " excluded - no colors spotted")
        return
    # check if markers are odd or even segments
    sI_segments_col = np.array(np.squeeze(ii))
    sI_segments_col_offset = sI_segments_col.reshape(-1)[0] % 2

    # check if this holds for ALL colored markers ("Verzählt?")
    if not np.all(sI_segments_col % 2 == sI_segments_col_offset):
        print(file + " excluded - odd/even pattern interruptet")
        return  # raise Exception('ERROR: segment edtection failed')

    # define markers and spacing width
    tt = np.full(len(sI_segments), tape_spacing)
    tt[sI_segments_col_offset::2] = tape_width
    sI_segments['mmwidth'] = tt

    # GET SCALE: linear (!) regression: vertical midpoints as X, mm-per-px as Y
    x = np.array(sI_segments['pxheight']) + np.array(sI_segments['pxwidth'], dtype=float) * 0.5
    y = np.array(sI_segments['mmwidth'], dtype=float) / np.array(sI_segments['pxwidth'], dtype=float)

    coef, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
    sI_scale_fun = np.poly1d(coef)

    ## Preparing for annoying questions on regression stats
    ## Idea from https://joshualoong.com/2018/10/03/Fitting-Polynomial-Regressions-in-Python/
    # df = pd.DataFrame(columns=['x','y']) # just for the summary
    # df['x'] = x
    # df['y'] = y
    # sI_scale_fun_summary = smf.ols(formula='y ~ sI_scale_fun(x)', data=df).fit()

    # UNCERTAINTY: https://de.wikipedia.org/wiki/Standardfehler_der_Regression
    # residuals: sum of squared y deviations: mm-per-px
    # SQRT of residuals divided by n-2
    if len(sI_segments) >= 3: # DOF check
        scale_error = np.sqrt( residuals/(len(x)-2) )
    else:
        scale_error = np.nan
    # plot(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5,
    # np.array(sI_segments[3])/np.array(sI_segments[1]), 'yo',
    # np.array(sI_segments[0])+np.array(sI_segments[1])*0.5,
    # sI_scale_fun(np.array(sI_segments[0])+np.array(sI_segments[1])*0.5),
    # '--k')

    # calculate distance of lower marker bounds to lower stake end in mm
    sI_segments['mmheight'] = sI_segments['pxheight'] * \
                              sI_scale_fun(np.array(sI_segments['pxheight'])
                                           * 0.5)
    # store UNCERTAINTY
    sI_segments['mmheight_scale_error'] = sI_segments['pxheight'] * scale_error
    print(sI_segments.iloc[-2:-1]['pxheight'] * scale_error)

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
    sI_segments_ret = sI_segments[sI_segments_col_offset::2]

    # store coordinates for overplotting and output on img
    # invert transform matrix to map coordinates back on image
    iM = cv2.invertAffineTransform(M)
    out_edges_peaks = [iM.dot(np.array([int(stake_img.shape[1] / 2), y, 1]))
                       for y in sI_edges_peaks]
    for xy in out_edges_peaks:
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.circle(img2, xxyy, 3, (255, 0, 0), -1)
    # segments with labels
    out_segments_XY = [iM.dot(np.array([int(stake_img.shape[1]),
                                        sI_stakeBottom[1] - y, 1])) for y in
                       sI_segments['pxheight']]
    tt_colors = np.array(sI_segments['color'])
    tt_height = np.array(sI_segments['mmheight'])
    for i, xy in enumerate(out_segments_XY):
        xxyy = tuple(np.int_(np.rint(xy)))
        cv2.line(img2, xxyy, (xxyy[0] + 50, xxyy[1]), (0, 255, 0), 1)
        cv2label(img2, str(tt_colors[i]) + ": " +
                 str(np.format_float_positional(tt_height[i], precision=1)) +
                 "mm", (xxyy[0] + 50, xxyy[1]), (0, 255, 0), 0.5, 1)
    cv2.line(img2, (arm_p0[0] + int(np.mean(armLR)), arm_p0[1]),
             (arm_p1[0] + int(np.mean(armLR)), arm_p1[1]), (0, 255, 0), 1)
    cv2.circle(img2, ring_match_pix, 3, (0, 0, 255), -1)
    cv2.circle(img2, tuple(np.int_(np.rint(stake_bottom))), 3, (0, 255, 0), -1)
    cv2.imwrite(outfile, img2)
    print(file + " probably     ok")
    return sI_segments_ret


if __name__ == "__main__":

    folder = "./InputImages"
    outfolder = "./OutputImages"
    results = {}
    for file in sorted(os.listdir(folder)):
        if file.endswith(".jpg"):
            match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}', file)
            timestamp = datetime.strptime(match.group(), '%Y-%m-%d_%H-%M')
            # print(timestamp)
            results[timestamp] = analyse_image(os.path.join(folder, file), os.path.join(outfolder, 'out_' +
                                                                                        file), tape_width=tape_width,
                                               tape_spacing=tape_spacing, arm_p0=arm_p0, arm_p1=arm_p1,
                                               segment_width_range=segment_width_range)

    # Save results:
    with open('results.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(results, f)

    print("Dummy Ende")

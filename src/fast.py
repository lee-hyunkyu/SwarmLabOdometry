"""Performs FAST corner detection without machine generated code."""

import numpy as np
import cv2
import matplotlib import pyplot as plt

# TODO: Convert image to array or 2D list?

def shape(array):
    """ Returns a list of 2D array dimensions """
    rows = 0
    for row in range(len(array)):
        row += 1

    cols = 0
    for cols in range(len(array[0])):
        cols += 1

    return [rows, cols]

def zeros(n):
    """ Returns the n x n zero matrix """
    return [[0 for i in range(0, n)] for j in range(0, n)]

def circle(row, col):
    """ 
    Returns a list of the pixels ((x,y, point index) tuples) that make up the circumference of a pixel's search region.
    Circle circumference = 16 pixels
    """
    point1 = (row+3, col)
    """
    point2 = (row+3, col+1, 2)
    point3 = (row+3, col-1, 3)
    point4 = (row+2, col+2, 4)
    """
    point5 = (row+1, col+3)
    """
    point6 = (row, col+3, 6)
    point7 = (row-1, col+3, 7)
    point8 = (row+2, col+2, 8)
    """
    point9 = (row-3, col)
    """
    point10 = (row-3, col+1, 10)
    point11 = (row-3, col-1, 11)
    point12 = (row-2, col-2, 12)
    """
    point13 = (row+1, col-3)
    """
    point14 = (row, col-3, 14)
    point15 = (row-1, col-3, 15)
    point16 = (row-2, col-2, 16)
    return [point1, point2, point3, point4, point5, point6, point7, \
            point8, point9, point10, point11, point12, point13, point14, point15, point16]"""
    return [point1, point5, point9, point13]

def is_corner(image, row, col, ROI, threshold):
    """
    We use the high speed test to detect a corner:
    Check if points 1, 9, 5, and 13 meet the threshold. 
    At least 3 of these must be the criteria in order for hte pixel p to be
    considered a corner.

    This does not reject as many candidates as checking every point in the circle,
    but it runs much faster and we can set the threshold to be a high value to filter
    out more non-corners
    """
    intensity = image[row][col]
    point1 = ROI[0]
    point9 = ROI[2]
    row1, col1 = point1
    intensity1 = image[row1][col1]
    row9, col9 = point9
    intensity9 = image[row9][col9]
    point5 = ROI[1]
    point13 = ROI[3]
    row5, col5 = point5
    intensity5 = image[row5][col5]
    row13, col13 = point13
    intensity13 = image[row13][col13]
    count = 0 
    if abs(int(intensity1) - int(intensity)) > threshold:
        count += 1
    if abs(int(intensity9) - int(intensity)) > threshold:
        count += 1
    if abs(int(intensity5) - int(intensity)) > threshold:
        count += 1
    if abs(int(intensity13) - int(intensity)) > threshold:
        count += 1

    return count >= 3
     
def detect(image, threshold=25):
    """
    corners = fast.detect(image, threshold) performs the detection
    on the image and returns the corners as a list of (x,y) tuples and the 
    scored as a list of of integers. The score is computed using binary search
    over all possible thresholds

    Setting nonmax = 0 performs corner detection but suppresses nonmaximal suppression 
    (edge thinning technique)

    This function does not search the entire frame for corners. It only searches a portion
    in the middle in order to speed up the process.

    ***Parameters: 
        image == numpy array (currently). NOTE: Image must be grayscale
        threshold == int
        nonmax = 0 or 1
    """

    corners = []
    rows = image.shape[0]
    cols = image.shape[1]
    startSearchRow = int(0.25*rows)
    endSearchRow = int(0.75*rows) # search the middle half of the frame
    startSearchCol = int(0.25*cols)
    endSearchCol = int(0.75*cols)

    for row in range(startSearchRow, endSearchRow):
        for col in range(startSearchCol, endSearchCol):
            ROI = circle(row, col) 
            if is_corner(image, row, col, ROI, threshold):
                corners.append((col, row))
    return corners;

def test():
    image = cv2.imread('/Users/timmytimmyliu/research/maap/images/template.png');
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = detect(image)


test()

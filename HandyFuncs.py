import os
from pathlib import Path

from numpy.typing import ArrayLike
import numpy as np
import random

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import *

import pygestalt as gs
from pygestalt import sampler

import skimage as ski

import math

from IPython.utils.io import capture_output
from contextlib import contextmanager

@contextmanager
def silence_output():
    """
    Context manager to silence output in Jupyter notebooks.
    
    Example:
        with silence_output():
            noisy_function()  # Output is silenced
        print("This will be printed normally")
    """
    with capture_output() as captured:
        yield captured

def blockPrint():
    # Disable
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    # Restore
    sys.stdout = sys.__stdout__

def coord(x,ImageDimensions = 1512):
    y = x/ImageDimensions
    return y

def rotate_bezier(shape, center, angle_degrees):
    """
    Rotate a Bezier curve's control points around a center point.
    
    Parameters:
    shape (list): List of [x,y] coordinates defining the Bezier curve
    center (list): [x,y] coordinates of the rotation center point
    angle_degrees (float): Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
    list: New list of rotated [x,y] coordinates
    """
    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)
    
    # Calculate sine and cosine once
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    # Center point coordinates
    cx, cy = center
    
    # Initialize result list
    rotated_shape = []
    
    for point in shape:
        # Translate point to origin
        x, y = point
        translated_x = x - cx
        translated_y = y - cy
        
        # Rotate point
        rotated_x = translated_x * cos_angle - translated_y * sin_angle
        rotated_y = translated_x * sin_angle + translated_y * cos_angle
        
        # Translate back
        final_x = rotated_x + cx
        final_y = rotated_y + cy
        
        rotated_shape.append([final_x, final_y])
    
    return rotated_shape

def scaleShape (shape, factor):
    """
    Scale a shape.
    First it brings the shape down to the axes. This prevents any issues with negative numbers
    that arise when subtracting the center value.
    Then it multiplies each value by the scaling factor.
    Finally, it adds brings it back up by the amount moved earlier.
    Thus the minimum X and Y values are unchanged because transformations happened when they were at 0.
    Only works with shapes with all positive values. 
    """
    xList = []
    yList = []
    for num in shape:
        x, y = num
        xList.append(x)
        yList.append(y)

    xMin, yMin = min(xList), min(yList)
    #print(xMin,yMin)

    scaledShape = []
    for num in shape:
        x, y = num
        x2, y2 = x - xMin, y - yMin
        x3, y3 = x2 * factor, y2 * factor
        xfinal, yfinal = x3 + xMin, y3 + yMin
        scaledShape.append([xfinal, yfinal])

    return scaledShape

def translate (shape, distance, nShapes):
    """
    Select a shape and create a list of all those shapes.
    Each shape is still defined as a list of lists defining the start/end/anchor points.
    You must also input the distace between the shapes (check unit, either graph scale or pixels)
    And input the total number of images.
    A while loop is used rather than for, so that i == 0 at the start, 
    thus not translating the first image.
    """

    array = np.empty(shape = (nShapes,len(shape),2),dtype='O')

    for n in range(nShapes):
        for i in range(len(shape)):
            array[n,i,0],array[n,i,1] = shape[i][0] + distance * n , shape[i][1]

    return array

def borderChecker (border:str,xCoord,imageDimPx = 1512, bboxVal = coord(180-53.5)):
   
    leftmost = min(xCoord)
    rightmost = max(xCoord)
    if border in ('L','Left','l','left','LEFT'):
        print('Testing Left')
        if leftmost != bboxVal:
            print('!!! Border Conflict !!!')
            for i in range(len(xCoord)):
                xCoord[i] += (bboxVal - leftmost)
            conflict = True
        else:
            print('!!! No Border Conflict !!!')
            conflict = False
    elif border in ('R','Right','r','right','RIGHT'):
        print('Testing Right')
        if rightmost != (1 - bboxVal):
            print('!!! Border Conflict !!!')
            for i in range(len(xCoord)):
                xCoord[i] -= (bboxVal - rightmost)
            conflict = True
        else:
            print('!!! No Border Conflict !!!')
            conflict = False
    elif border == 'M':
        print('Testing Midline')
        if rightmost != (.5 - bboxVal):
            xCoord -= (.5 - bboxVal - rightmost)
        conflict = True

    else:
        print("ERROR: invalid 'border' argument. Did you mean 'L' or 'R' ?")
        conflict = None
        quit()

    leftmost = min(xCoord)
    rightmost = max(xCoord)
    center = (leftmost + rightmost) / 2

    return xCoord, center, conflict, leftmost, rightmost


def vertCentering (Top:bool, imageWidthPix, Ycoords):

    midY = (min(Ycoords) + max(Ycoords)) / 2
    if Top:
        diff = .75 - midY
    else: 
        diff = .25 - midY
    Ycoords += diff
    if Top:
        print('Equal distance between Y-border and shape boundary:',round(min(Ycoords) - .5,2) == round(1 - max(Ycoords),2))
    else:
        print('Equal distance between Y-border and shape boundary:',round(min(Ycoords),2) == round(.5 - max(Ycoords),2))
    if min(Ycoords) < 0 or max(Ycoords) > 1:
        print('!!! Scaling Factor and/or Rotation lead to conflict !!!')
    return Ycoords

def nShapeCaseOveride (nShapes, xCoordsLeft, bbox):
    mini, maxi = min(xCoordsLeft), max(xCoordsLeft)
    mid = (mini + maxi) / 2
    print(round(mini,2,),'    ',round(maxi,2))
    Lquad, center, Rquad = .25, .5, .75

    if nShapes == 2:
        diff = mid - Lquad
        xCoordsLeft -= diff
    mini, maxi = min(xCoordsLeft), max(xCoordsLeft)
    print(round(mini,2,),'    ',round(.5-maxi,2))
    return xCoordsLeft

def godFunction (VCenter:bool = True,
                shapeList = None, 
                rotationDeg = 0, 
                scalingFactor = 1, 
                imageWidthPix = 1512, 
                nShapes = 8, 
                nCoordPairs = 100, 
                Top:bool = False,
                bboxSize = coord(180-53.5),
                spread:bool = False,
                Right:bool = False,
                override:bool = False,
                Yflip:bool = False):
    
    """Args:
            shapeList: A list of coordinates in the form [[x1,y1],[x2,y2]...] or a np.array
            rotationDeg: Float or Scalar, specifying degrees to rotate shape about the shape's center
            scalingFactor: Float or Scalar,  specifying how much larger/smaller the shape should be. Keeps minimum X and Y coordinates the same IF VCenter == False
            imageWidthPix: Float or Scalar, this assumes the image will be a square
            nShapes: Scalar, how many shapes to draw
            nCoordPairs: The number of coordiates to output in bx and by
            VCenter: Bool, whether or not to center the shapes in the specified hemisphere
            Top: Bool, whether the images should be drawn in the top hemisphere or not
            bboxSize: The total distance wanted between shapes at the extremes and their control lines (e.g. image boundaries, midlines). coord(180-53.5) is the farthest left point from Mandoh's original back_C
            spread: Bool, determines whether the shapes will be spread across the entire X-axis, or justt one specified quadrant with Top
        Explanation: 
            This hell of a function is my own personal child. It will in this order:
            1) Scale the shape from it's minimum X and Y coords
            2) Rotate the shape about it's center
            3) Check to see if it overlaps the left image border (or specified X-coord)
            4) Adjusts X-coords to not overlap
            5) Duplicate the shape to the far right
            6) Checks to make sure there is equal distace between shapes and their corresponding borders
            7) Finds the total distance between L and R shapes
            8) Determines the inter-shape distance using total distance and nShapes
        Outputs:
            arrayFinal: an array of only X-coords for each shape
            rightmostShape: a vector of only X-coords
            leftmostShape: same as above, but for the left shape
            by: a vector of only Y-coords, assumes that all shapes should have the same Y-coords"""

    shape = np.array(shapeList)

    # First scale
    shape_scale = np.array(scaleShape(shape=shape,factor=scalingFactor))
    midX, midY = (shape_scale[0][0] + shape_scale[-1][0]) / 2, (shape_scale[0][1] + shape_scale[-1][1]) / 2
    center = [midX,midY]
    # then rotate
    transformed = np.array(rotate_bezier(shape=shape_scale, center=center, angle_degrees=rotationDeg))

    # After trasforming the anchor points we need to convert to actual Bezier line position
    bx, by = gs.sampler.bezier_curve_position(np.linspace(0,1,nCoordPairs), transformed).T

    if Top:
        by += coord(imageWidthPix/2)
    if VCenter:
        with silence_output():
            by = vertCentering(Top=Top,imageWidthPix=imageWidthPix,Ycoords=by)
    
    if override:
        with silence_output():
            bx = nShapeCaseOveride(nShapes = nShapes, xCoordsLeft = bx, bbox = bboxSize)

    # Find the distance from L border
    with silence_output():
        leftmostShape, centerL, Lconflict, Lleft, Lright = borderChecker('L',bx, bboxVal=bboxSize)

    # Now send it to the R image border, simultaneously mirroring it again
    rightmostShape = np.zeros_like(leftmostShape)
    for i in range(len(leftmostShape)):
        rightmostShape[i] = leftmostShape[i] - centerL
        rightmostShape[i] = rightmostShape[i] * -1
        rightmostShape[i] = rightmostShape[i] + centerL
        if spread:
            rightmostShape[i] = 1-rightmostShape[i]
        else:
            rightmostShape[i] = .5-rightmostShape[i]

    # Check if it goes of the R image border
    if not spread:
        border = 'M'
    else:
        border = 'R'
    with silence_output():
        rightmostShape, centerR, Rconflict, Rleft, Rright = borderChecker(border,rightmostShape, bboxVal=bboxSize)

    arrayFinal = np.zeros((nShapes,nCoordPairs), dtype='f')

    # if Lconflict and not Rconflict:
    #     print('Equal distance between X-border and shape boundary:', round(Lleft,5) == round(1-Rright,5))
    
    # Make the coordinates equally spaced between the two L and R shapes
    interShapeDist = (centerR-centerL)  / (nShapes - 1) # How much to translate each shape along X-axis

    # Now in order to get the actual X-coords for the shape we need to use PyGestalt

    for i in range(nShapes):
        arrayFinal[i] = leftmostShape + i * interShapeDist

    if Right and not spread:
        for i in range(nShapes):
            center = (min(arrayFinal[i]) + max(arrayFinal[i])) / 2
            arrayFinal[i] -= center
            arrayFinal[i] *= -1
            arrayFinal[i] += center
            arrayFinal[i] = 1 - arrayFinal[i]

    if Yflip:
        mid = (min(by) + max(by)) / 2
        by -= mid
        by *= -1
        by += mid

    if spread:
        if np.max(arrayFinal) > 1 or np.min(arrayFinal) < 0:
            print('!!! Scaling Factor and/or Rotation lead to X-axis conflict !!!')
    else:
        if not Right:
            if np.max(arrayFinal) > .5 or np.min(arrayFinal) < 0:
                print('!!! Scaling Factor and/or Rotation lead to X-axis conflict !!!')
        else:
            if np.max(arrayFinal) > 1 or np.min(arrayFinal) < .5:
                print('!!! Scaling Factor and/or Rotation lead to X-axis conflict !!!')
    if Top and max(by) > 1 or Top and min(by) < .5:
        print('!!! Scaling Factor and/or Rotation lead to Y-axis conflict!!!')
    elif not Top and max(by) > .5 or not Top and min(by) < 0:
        print('!!! Scaling Factor and/or Rotation lead to Y-axis conflict!!!')
            
    return arrayFinal, rightmostShape, leftmostShape, by, interShapeDist


def equidistance (shapeList,
                  nShapes:int,
                  scalingFactor = 1,
                  rotationDeg = 0, 
                  nCoordPairs:int = 100, 
                  imageWidthPix = 1512, 
                  Top:bool = False, 
                  VCenter:bool = True,
                  spread:bool = False,
                  Right:bool = False,
                  Flip:bool = False
                  ):

    shape = np.array(shapeList)

    if Right & spread:
        Right = False

    # First scale
    shape_scale = np.array(scaleShape(shape=shape,factor=scalingFactor))
    midX, midY = (shape_scale[0][0] + shape_scale[-1][0]) / 2, (shape_scale[0][1] + shape_scale[-1][1]) / 2
    center = [midX,midY]
    # then rotate
    transformed = np.array(rotate_bezier(shape=shape_scale, center=center, angle_degrees=rotationDeg))

    # After trasforming the anchor points we need to convert to actual Bezier line position
    bx, by = gs.sampler.bezier_curve_position(np.linspace(0,1,nCoordPairs), transformed).T

    if Flip:
        bin = np.average(by)
        by -= bin
        by *= -1
        by += bin
    if Top:
        by += .5
    if VCenter:
        with silence_output():
            by = vertCentering(Top=Top,imageWidthPix=imageWidthPix,Ycoords=by)

    w = 1
    if not spread:
        w = .5

    bxW = max(bx) - min(bx)
    b = w / (4 * nShapes)
    d = (w / nShapes) + bxW

    if nShapes == 1:
        bx -= np.average(bx) - w / 2
        if Right and not spread:
            bx += .5
        return bx, by
    
    bx -= min(bx) - b
    bin = (max(bx) + min(bx)) / 2
    bxF = bx - bin
    bxF *= -1
    bxF += bin
    bxF = w - bxF

    d = (bxF - bx) / (nShapes - 1)
    arrayFinal = np.zeros((nShapes,nCoordPairs), dtype='f')
    for i in range(nShapes):
        if i == 0:
            arrayFinal[0] = bx
        elif i == nShapes-1:
            arrayFinal[-1] = bxF
        elif i not in (0, nShapes-1):
            arrayFinal[i] = bx + d * i
    if Right and not spread:
        arrayFinal += .5
    
    return  arrayFinal, by


def precise_segments(middle, angle, segment_length, n_segments):
    """
    Arguments:
    n_segment: how many line segments you want MUST BE EVEN NUMBER??
    middle: the center of the middle line segment, not the center of the global shape
    segment_length: how long each segment should be
    angle: the angle offset between each segment, in degrees
    """
    middle = np.array(middle)
    sl = segment_length / 2
    phi = 180 - angle
    n_points = n_segments+1
    pts = np.zeros(shape = (n_points,2), dtype=float)

    pts[0] = middle + [0,sl] # upper point of vertical segment
    pts[1] = middle - [0,sl] # lower point of vertical segment
    pts[2] = rotate_bezier(shape=[pts[1]], center=pts[0], angle_degrees=phi)[0]
    pts[3] = rotate_bezier(shape=[pts[0]], center=pts[1], angle_degrees=-phi)[0]

    if n_segments > 3:
        for i in range(4,len(pts)):
            if i%2:
                pts[i] = rotate_bezier([pts[i-4]],pts[i-2],-phi)[0]
            else:
                pts[i] = rotate_bezier([pts[i-4]],pts[i-2],+phi)[0]
    
    sorted = pts[pts[:,1].argsort()]
    bx, by = sorted[...,0], sorted[...,1]
    xC = middle[0]
    bxC = (bx.min()+bx.max())/2
    bx += (xC - bxC)
    return(bx,by)

def spatialJitterUni(C,r,j,seed=None):
    """
    Adds equal jitter in both directions
    Returns C modified by a uniform distribution that ranges between +/- (r*j)
    Args:
    C: 2D array of the X and Y coordinates
    j: jitter amount, should be between 0 and 1
    r: radius of the 'bubbles' the line segments occupy
    """
    # -.5 centers distribution on 0
    # *2 scales it -:+1
    # *2*r turns radius -> diameter
    # *j proportion of the bubble the line segments can jitter in
    return C + 2*2*r*j*(np.random.RandomState(seed).rand(C.shape[0],C.shape[1]) - 0.5)

def spatialJitterNorm(C,r,j,seed=None):
    """
    Adds equal jitter in both directions
    Returns C modified by a uniform distribution that ranges between +/- (r*j)
    Args:
    C: 2D array of the X and Y coordinates
    j: jitter amount, should be between 0 and 1
    r: radius of the 'bubbles' the line segments occupy
    """
    return C + np.random.RandomState(seed).normal(loc= 100, scale = r*j, size = C.shape)

def regularPolygon(degrees, shape:str, radius=0.25, center:ArrayLike=[.5,.5], plot:bool=False):

    """
    External angles always add up to 360. This is the basis for generating this array.
    Args: most are pretty straightforward...
    degrees: the measure of the exteral angle of each line segment
    shape: a string saying C|BC to indicate which way the shape should open
    """

    shape = shape.upper()
    if shape not in ['C','BC']:
        raise ValueError("Shape must be 'C' or 'BC' (case-insensitive).")

    nSides = 360/degrees
    if nSides%1 == 0:
        nSides = int(nSides)
    else:
        raise ValueError("Angle must divide 360 evenly to form regular polygon.")
    
    Cx, Cy = center
    angles = np.linspace(0, 2*np.pi, nSides+1, endpoint=True)

    x = Cx + radius * np.cos(angles)
    y = Cy + radius * np.sin(angles)
    Ps = np.column_stack([x,y])

    if shape == 'C':
        indices = np.argsort(Ps[:,0])[:7]
    else:
        indices = np.argsort(Ps[:,0])[-8:]

    Ps = Ps[indices]
    Ps[...,0] += -1*(Ps[...,0].mean()) + Cx

    ## Now get the tangents
    Tx = Cx + radius * np.sin(angles)
    Ty = Cy + radius * -np.cos(angles)
    Ts = np.column_stack([Tx,Ty])

    Ts = Ts[indices]
    Ts[...,0] += -1*(Ts[...,0].mean()) + Cx
    # rather than mean-centering, 
    # we want to center over the middle point of the C
    Ts[:,0] -= (Ts[0,0] - Ps[0,0])




    if plot:
        plt.plot(Ps[...,0],Ps[...,1],'b.', label = 'Points')
        plt.plot(Ts[...,0],Ts[...,1],'r.', label = 'Tangents')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend()
    
    return Ps, Ts

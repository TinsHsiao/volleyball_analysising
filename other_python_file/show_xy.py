# From Python
# It requires OpenCV installed for Python

import sys
import cv2
import os
from sys import platform
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

bodyHeightPixel = -1
ground_point = 5000.0


def set_ground_position(gp, feet1, feet2) : # use feet as ground position
    if feet1 != 0 and feet2 != 0 :
        if min(feet1, feet2) < gp :
            gp = min(feet1, feet2)
            print(gp)
    elif feet1 != 0 and feet2 == 0 and feet1 < gp:
        gp = feet1
    elif feet2 != 0 and feet1 == 0 and feet2 < gp:
        gp = feet2



def plot_y_trend(y) :
    plt.plot(y)
    plt.ylabel('Y Position in frame')
    plt.xlabel('time')
    plt.show()

def plot_x_trend(x) :
    plt.plot(x)
    plt.ylabel('X Position in frame')
    plt.xlabel('time')
    plt.show()

def plot_x_y_position(x, y) :
    plt.plot(x, y)
    plt.ylabel('Y Position in frame')
    plt.xlabel('X Position in frame')
    plt.show()   

def set_body_height(bhp, head, feet) :
    if bhp == -1 :
        bhp = head - feet 
    elif ( head - feet ) > bhp :
        bhp = head - feet 
    #print(bhp)
    return bhp


def calculate_jump_height(bhp, body_height, y) :
    i = 0 
    h_index = 0 
    height = 0
    while True :
        if i == np.size(y) :
            break
        elif y [i] > height :
            height = y[i]
            h_index = i
        i = i + 1
    
    true2pixel = body_height / bhp # a pixel == true long cm
    return true2pixel, height*true2pixel - body_height, h_index

    
def plot_center_height(str, true2pixel, height_index) :
    str_new = []
    #print(str)
    #print("pass2")
    for i in range(np.size(str)) :
        str_new.append(str[i] * true2pixel)
    #print("pass3")
    plt.plot(str_new)
    #print("pass4")
    #plt.plot(str*true2pixel)
    #plt.plot(height_index,str[height_index]*true2pixel,'ro') 
    plt.ylabel('Center position in axis-y (unit=cm)')
    plt.xlabel('time')
    plt.show()

    

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/test_video9.mp4", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()



    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cap = cv2.VideoCapture(args[0].image_path)
    plot = np.ones((1920, 1080), dtype='float64')
    begining = time.time()
    y = []
    x = []
    i = 0

    judge = True
    while cap.isOpened():
        # Process Image
        datum = op.Datum()
        ret, imageToProcess = cap.read()
        if not ret:
            print("Can't receive frame")
            break
        
        elif judge == True :
            datum.cvInputData = imageToProcess
            transY= imageToProcess.shape[0]
            transX = imageToProcess.shape[1]
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            judge = False

        elif judge == False :
            datum.cvInputData = imageToProcess
            transY = imageToProcess.shape[0]
            transX = imageToProcess.shape[1]
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        if ( np.ndim(np.array(datum.poseKeypoints)) != 0 ) :
            list_x = np.array(datum.poseKeypoints)[0]
            x.append(np.array(datum.poseKeypoints)[0][8][0])
            y.append(transY - np.array(datum.poseKeypoints)[0][8][1])
            #print(transY - np.array(datum.poseKeypoints)[0][19][1])

            #if frame_data_can_use(np.array(datum.poseKeypoints)[0]) == True :
            if np.array(datum.poseKeypoints)[0][15][1] != 0 and np.array(datum.poseKeypoints)[0][16][1] != 0 and np.array(datum.poseKeypoints)[0][19][1] != 0:
                bodyHeightPixel = set_body_height( bodyHeightPixel, transY - np.array(datum.poseKeypoints)[0][15][1], transY - np.array(datum.poseKeypoints)[0][19][1])
            elif np.array(datum.poseKeypoints)[0][16][1] != 0 and np.array(datum.poseKeypoints)[0][19][1] != 0 :
                bodyHeightPixel = set_body_height( bodyHeightPixel, transY - np.array(datum.poseKeypoints)[0][16][1], transY - np.array(datum.poseKeypoints)[0][19][1])
            #print(bodyHeightPixel)
            set_ground_position( ground_point, transY - np.array(datum.poseKeypoints)[0][19][1], transY - np.array(datum.poseKeypoints)[0][22][1])

        i = i + 1
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(2)
        end = time.time()

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

    true2pixel, jump_height, h_index = calculate_jump_height(bodyHeightPixel, 175, y)
    plot_center_height(y, true2pixel, h_index)
    plot_y_trend(y)
    plot_x_trend(x)
    plot_x_y_position(x, y)
    print( "Jumping Height = ",  round( jump_height, 2 ), " cm" ) 



except Exception as e:
    print(e)
    sys.exit(-1)

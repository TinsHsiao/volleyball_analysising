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
import threading


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class data_process :
    g_p = 5000 # ground truth point
    h_pixel = -1 # body height pixels in video
    c_p_mostheight_index = 0 # center in the highest
    center2feet = 0
    center_lowest_index = 0
    
    def set_ground_position(gp, feet1, feet2) : # use feet as ground position
        if feet1 != 0 and feet2 != 0 :
            if min(feet1, feet2) < gp :
                gp = min(feet1, feet2)
        elif feet1 != 0 and feet2 == 0 and feet1 < gp:
            gp = feet1
        elif feet2 != 0 and feet1 == 0 and feet2 < gp:
            gp = feet2
        data_process.g_p = gp

    def set_center2feet(center, feet1, feet2):
        if feet1 != 0 and feet2 != 0 :
            if data_process.center2feet < (center - min(feet1, feet2)) :
                data_process.center2feet = center - min(feet1, feet2)
        elif feet1 != 0 and feet2 == 0 and data_process.center2feet < (center - min(feet1, feet2)):
            data_process.center2feet = center - min(feet1, feet2)
        elif feet2 != 0 and feet1 == 0 and data_process.center2feet < (center - min(feet1, feet2)):
            data_process.center2feet = center - min(feet1, feet2)
        return data_process.center2feet
        
    def find_center_lowest_index(str_y) :
        center_lowest = 10000
        for i in range(np.size(str_y)) :
            if str_y[i] < center_lowest :
                data_process.center_lowest_index = i

    def calculate_approach_rate(fps, str_x, true2pixel) : 

        distance = abs( str_x[data_process.center_lowest_index]-str_x[0] ) * true2pixel
        time = (1/fps) * data_process.center_lowest_index
        velocity = (distance/100)/time    # m / s
        return velocity
    

    def set_body_height(head, feet) :
        if data_process.h_pixel == -1 :
            data_process.h_pixel = head - feet 
        elif ( head - feet ) > data_process.h_pixel :
            data_process.h_pixel = head - feet 
        return data_process.h_pixel
    
    
    def calculate_jump_height(body_height, y) :
        i = 0 
        h_index = 0 
        height = 0
        while True :
            if i == np.size(y) :
                break
            elif y [i] - data_process.g_p > height :
                height = y[i] - data_process.g_p
                h_index = i
            i = i + 1
        
        true2pixel = body_height / data_process.h_pixel # a pixel == true long cm
        return true2pixel, height*true2pixel - body_height, h_index
    
    def plot_jump_height(fps, str, true2pixel) :
        str_new = []
        time = []
        height = 0
        for i in range(np.size(str)) :
            time.append((1/fps)*i)
            if (str[i]-data_process.center2feet) < 0 :
                str_new.append(0)
            else :
                str_new.append((str[i]-data_process.center2feet) * true2pixel)
                if str_new[i] > height :
                    height = str_new[i]

        plt.plot(time, str_new)
        plt.title("The Play's Jumping Height")
        plt.ylabel('Jumping Height[cm]')
        plt.xlabel('Time[s]')
        # plt.grid()
        #plt.show()
        plt.savefig('result/Jumping_height.jpg')
        plt.close()
        return height


    
    def plot_center_height(fps, str, true2pixel) :
        str_new = []
        time = []
        for i in range(np.size(str)) :
            time.append((1/fps)*i)
            str_new.append(str[i] * true2pixel)

        plt.plot(time, str_new)
        plt.title("The Player's Body Center Height ")
        plt.ylabel('Height[cm]')
        plt.xlabel('Time[s]')
        #plt.grid()
        #plt.show()
        plt.savefig('result/center_height.jpg')
        plt.close()

    def plot_in_world(fps, str, true2pixel) :
        str_new = []
        time = []
        for i in range(np.size(str)) :
            time.append((1/fps)*i)
            str_new.append(str[i] * true2pixel)

        plt.plot(time, str_new)
        plt.title("Player's position in x axis")
        plt.ylabel('cm')
        plt.xlabel('Time[s]')
        # plt.grid()
        plt.show()

    def plot_y_trend(y) :
        plt.plot(y)
        plt.title("Y Trend")
        plt.ylabel('y')
        plt.xlabel('time')
        plt.show()

    def plot_x_trend(x) :
        plt.plot(x)
        plt.ylabel('X Trend')
        plt.xlabel('time')
        plt.show()

    def plot_x_y_position(x, y, true2pixel) :
        str_x = []
        str_y = []
        time = []
        for i in range(np.size(x)) :
            str_x.append((x[i]-x[0])*true2pixel)
            str_y.append((y[i]-y[0])*true2pixel)
        plt.plot(str_x, str_y)
        plt.title("The Player's Position")
        plt.annotate("Origin : The Player's Initial Location.", xy=(19,33.9), xytext = (21,35),
           arrowprops=dict(facecolor='black', shrink=0.05))
        plt.ylabel('Height[cm]')
        plt.xlabel('Distance From Origin [cm] ')
        # plt.grid()
        #plt.show()
        plt.savefig("result/The_Player's_Position.jpg")
        plt.close()   
    
    def get_angle_point(human, pos):
        pnts = []
    
        if pos == 'left_elbow':
            pos_list = (5,6,7)
        elif pos == 'left_hand':
            pos_list = (1,5,7)
        elif pos == 'left_knee':
            pos_list = (12,13,14)
        elif pos == 'left_ankle':
            pos_list = (5,12,14)
        elif pos == 'right_elbow':
            pos_list = (2,3,4)
        elif pos == 'right_hand':
            pos_list = (1,2,4)
        elif pos == 'right_knee':
            pos_list = (9,10,11)
        elif pos == 'right_ankle':
            pos_list = (2,9,11)
        else:
            print('Unknown  [%s]', pos)
            return pnts
    
        for i in range(3):
            if human[pos_list[i]][2] <= 0.1:
                print('component [%d] incomplete'%(pos_list[i]))
                return pnts
    
            pnts.append((int( human[pos_list[i]][0]), int( human[pos_list[i]][1])))
        return pnts

    def angle_between_points( p0, p1, p2 ):
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        if a * b == 0:
            return -1.0 
        
        return  np.arccos( (a+b-c) / np.sqrt(4*a*b) ) * 180 /np.pi

    def angle_right_elbow(human):
        pnts = data_process.get_angle_point(human, 'right_elbow')
        if len(pnts) != 3:
            print('component incomplete')
            return 0
        angle = 0
        if pnts is not None:
            angle = data_process.angle_between_points(pnts[0], pnts[1], pnts[2])
            print('Right Elbow angle:%f'%(angle))
        return angle
    
    def plot_knee_angle(fps, angle) :
        time = []
        for i in range(np.size(angle)) :
            time.append((1/fps)*i)
        plt.plot(time, angle)
        plt.ylabel('The Left Knee Angle[degree]')
        plt.ylim((0, 360))
        plt.xlabel("Time[s]")
        plt.show()


class pose :

    jump_height = 0
    approach_speed = 0

    def pose(path) :
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
            #parser.add_argument("--image_path", default= "data/test_video9.mp4", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
            parser.add_argument("--image_path", default= path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
            center_y = []
            center_x = []
            head_y = []
            pose.right_elbow_angle = []
            i = 0
            begining = time.time()
            end = 0
            frame_num = 0
            judge = True
            dp = data_process
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter('result/output.mp4', fourcc, 20.0, (640,368))
            fps_for_video = 56

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
                    # transY = imageToProcess.shape[0]
                    # transX = imageToProcess.shape[1]
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                # Display Image
                # print("Body keypoints: \n" + str(datum.poseKeypoints))


                if ( np.ndim(np.array(datum.poseKeypoints)) != 0 ) :
                    ky = np.array(datum.poseKeypoints)[0]
                    pose.right_elbow_angle.append(dp.angle_right_elbow(datum.poseKeypoints[0]))
                    center_x.append(ky[8][0])
                    center_y.append(transY - ky[8][1])
                    head_y.append(transY-ky[0][1])
                    #print(transY - np.array(datum.poseKeypoints)[0][19][1])
                    #if frame_data_can_use(np.array(datum.poseKeypoints)[0]) == True :

                    if ky[15][1] != 0 and ky[16][1] != 0 and ky[19][1] != 0:
                        dp.set_body_height( transY - min(ky[15][1], ky[16][1]), transY - ky[19][1])
                    elif ky[16][1] != 0 and ky[19][1] != 0 :
                        dp.set_body_height( transY - ky[16][1], transY - ky[19][1])
                    elif ky[15][1] != 0 and ky[19][1] != 0 :
                        dp.set_body_height( transY - ky[15][1], transY - ky[19][1])

                    
                    dp.set_ground_position( dp.g_p, transY - ky[19][1], transY - ky[22][1])
                    dp.set_center2feet(transY-ky[8][1], transY - ky[19][1], transY - ky[22][1])

                out.write(datum.cvOutputData)
                # write an output video

                frame_num = frame_num + 1
                i = i + 1
                cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
                

                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
            
            out.release()
            cap.release()
            optime = round(time.time()-begining, 2) # 處理所有frame時間

            fps = frame_num/optime
            print("Video FPS:" + str(round(fps_for_video)))
            print("Openpose process FPS:" + str(round(frame_num/optime)))
                

            true2pixel, c_jump_height, dp.c_p_mostheight_index = dp.calculate_jump_height(175, head_y) # 175 == bodyheight
            # ==== Need to cal video distortion ===
            true2pixel = true2pixel*0.6 
            c_jump_height = c_jump_height*0.6
            # =====================================

            # dp.plot_in_world(fps_for_video, center_x, true2pixel)
            dp.find_center_lowest_index(center_y)
            pose.approach_speed = round( dp.calculate_approach_rate(fps_for_video, center_x, true2pixel), 2)
            print( "The Speed Of Approaching = ", pose.approach_speed, " m/s")
            pose.jump_height = dp.plot_jump_height(fps_for_video, center_y, true2pixel)
            dp.plot_center_height(fps_for_video, center_y, true2pixel)
            dp.plot_x_y_position(center_x, center_y, true2pixel)
            print( "The Jumping Height= ",  round( pose.jump_height, 2 ), " cm" ) 
            # dp.plot_angle(fps_for_video, left_knee_angle)




        except Exception as e:
            print(e)
            sys.exit(-1)

if __name__ == "__main__":
    p = pose
    path = "C:\\Users\\tinah\\OneDrive\\文件\\CSIE\\Seminar\\openpose\\build\\examples\\tutorial_api_python\\data\\test_video9.mp4"
    p.pose(path)
    



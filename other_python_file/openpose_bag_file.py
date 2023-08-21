#####################################################
##               Read bag from file                ##
#####################################################


# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
from sys import platform
import argparse
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, default='Demos/20230517_141907.bag')
# Parse the command line arguments to an object
args = parser.parse_args()

# Safety if no parameter have been given

if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()


#  def record(point_str):
#     feet = 0
#     head = 0
#     for i in range (point_str.shape) :
#         center = point_str[0][8][1]

#         head = point_str[0][15][1]
#         if point_str[0][22][1] > feet :
#         feet = point_str[0][22][1]

#         elif point_str[0][19][1] > feet :
#         feet = point_str[0][19][1]

#     pixel = 181 / (head-feet) 
    

try:
    # Create pipeline
    pipeline = rs.pipeline()
   
    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    #rs.config.enable_device_from_file(config, args.input)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, rs.format.bgr8, 30)


    pipeline.start(config)

    # colorizer = rs.colorizer()

    params = dict()
    params["model_folder"] = "../../../models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()


    while True :
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_frame = frames.get_depth_frame()
        depth_frame_image = np.asanyarray(depth_frame.get_data())
        datum = op.Datum()
        datum.cvInputData = color_image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        #print(depth_frame_image)
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        point_str = datum.poseKeypoints


        
        cv2.namedWindow("OpenPose 1.7.0 - Tutorial Python API", 0)
        cv2.resizeWindow("OpenPose 1.7.0 - Tutorial Python API", 1024, 768)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pass



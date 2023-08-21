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
parser.add_argument("-i", "--input", type=str, default='20230324_092705.bag')
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

try:
    # Create pipeline
    pipeline = rs.pipeline()
   
    # Create a config object
    config = rs.config()
    
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    pipeline.start(config)

    colorizer = rs.colorizer()

    params = dict()
    params["model_folder"] = "../../../models/"
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    while True :
        frames = pipeline.wait_for_frames()
        rgb_frame = frames.get_color_frame()
        rgb_color_image = np.asanyarray(rgb_frame.get_data())
        datum = op.Datum()
        datum.cvInputData = rgb_color_image
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        
        cv2.namedWindow("OpenPose 1.7.0 - Tutorial Python API", 0)
        cv2.resizeWindow("OpenPose 1.7.0 - Tutorial Python API", 1280, 960)
        cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pass



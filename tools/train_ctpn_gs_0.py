
#from __future__ import print_function

import sys
import cv2
import os
import shutil
import stat
import subprocess

from cfg import Config as cfg

# Make sure that caffe is on the python path:
current_path = os.path.dirname(os.path.abspath(__file__))
project_home = os.path.dirname(current_path)
os.chdir(project_home)
# sys.path.insert(0, 'caffe')


caffe_home = os.path.normpath(os.path.join(os.getcwd(), "caffe"))
os.chdir(caffe_home)
sys.path.insert(0, 'python')

import caffe

from src.other import draw_boxes, resize_im, CaffeModel

from src.detectors import TextProposalDetector, TextDetector
import os.path as osp
from src.utils.timer import Timer

DEMO_IMAGE_DIR="demo_images/"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

demo_imnames=os.listdir(DEMO_IMAGE_DIR)
timer=Timer()

for im_name in demo_imnames:
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "Image: %s"%im_name

    im_file=osp.join(DEMO_IMAGE_DIR, im_name)
    im=cv2.imread(im_file)

    timer.tic()

    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)

    print "Number of the detected text lines: %s"%len(text_lines)
    print "Time: %f"%timer.toc()

    im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, wait=False)

print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
cv2.waitKey(0)


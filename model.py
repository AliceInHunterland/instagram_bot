from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import model as  Mo
import matplotlib.pyplot as plt
import utilz as U
import numpy as np
from parser_utils import get_parser
import pickle
import random 
import cv2
import copy
from PIL import Image
import numpy
from blend_modes import *
import pafy
from natsort import natsorted, ns
import shutil
# from google.colab.patches import cv2_imshow
# from google.colab import output

## Get AI options
options =type('Expando', (object,), {})()
options.img_h = 224
options.img_w = 224
options.s_h = 56
options.s_w = 56
options.kshot = 5
options.nway = 5
options.learning_rate = 0.0001

#we are ready for 5 class 5 shot queries.
#each folder in folder_with_sample_folders must have 
#up to 5 jpg color images 
#with a black and white mask png file per image
#query must be a colored image
folder_with_sample_folders = "/content/data/"
model_weights_path = './VGG_b345_5_fewshot_DOGLSTM.h5'

print("model loading....")
model = Mo.my_model(encoder = 'VGG_b345', input_size = (options.img_h, options.img_w, 3), k_shot = options.kshot, learning_rate = options.learning_rate)
model.summary()
model.load_weights(model_weights_path)
overall_miou = 0.0

## Get an episode for test 



def data(opt, folder_with_sample_folders):
    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype = np.float32)
    smasks  = np.zeros([opt.nway, opt.kshot, opt.s_h, opt.s_w,        1], dtype = np.float32)

    setX = os.listdir(folder_with_sample_folders)
    setX = natsorted(setX, alg=ns.PATH | ns.IGNORECASE)
    #print(opt.nway)
    for idx in range(opt.nway):
        #print(idx)
        for idy in range(0, opt.kshot): # For support set 
            #print(idy)
            s_img = ''
            if os.path.isfile(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.jpg' ):
              s_img = cv2.imread(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.jpg' ) 
            elif os.path.isfile(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.jpeg' ):
              s_img = cv2.imread(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.jpeg' )
            else:
              continue
            s_msk = cv2.imread(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.png' )
            print(folder_with_sample_folders + setX[idx] + '/' + str(idy+1) + '.png' )
            s_img = cv2.resize(s_img,(opt.img_h, opt.img_w))
            s_msk = s_msk /255
            s_msk = cv2.resize(s_msk,(opt.s_h, opt.s_w))

            s_msk = np.where(s_msk > 0.5, 1., 0.)

            support[idx, idy] = s_img
            smasks[idx, idy]  = s_msk[:, :, 0:1] 

    support = support /255.
    return support, smasks   




print("model initiated")


def make_img(opt,path):
    support, smask = data(options, folder_with_sample_folders)
    Ss_mask = None
    OSs_mask = None
    print("Started reading file")

    frame = cv2.imread(path)
    height, width, channels = frame.shape
    frame = frame[0:height, int(width/2)-int(height/2):int(width/2)+int(height/2)]
    frame = cv2.resize(frame,(opt.img_h, opt.img_w))
    query   = np.zeros([opt.nway, opt.img_h, opt.img_w, 3], dtype = np.float32)      
                    
    for idx in range(opt.nway):
          query[idx] = frame.copy()
      
    query   = query   /255
    Ss_mask = model.predict([support, smask, query])
    Es_mask = Ss_mask
  
    Es_mask = np.where(Es_mask > 0.5, 1 , 0.)
    Es_mask = Es_mask * 255



    def store(O, tiles):
                  O = cv2.resize(O, (options.img_h, options.img_w)).astype(numpy.uint8)
                  ret, thresh = cv2.threshold(O, 1, 255, cv2.THRESH_BINARY)
                  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  O = cv2.cvtColor(O, cv2.COLOR_GRAY2BGRA)
                  O[:, :, 0] = 0
                  O[:, :, 2] = 0
                  O[:, :, 3] = 255
                  tiles = cv2.cvtColor(tiles, cv2.COLOR_BGR2BGRA)
                  s = O.astype(float) 
                  opacity = 0.6
                  tiles = tiles.astype(float)
                  blended = soft_light(tiles, s, opacity).astype(numpy.uint8)
                  blended = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
                  blended = cv2.drawContours(blended, contours, -1, (240, 240, 255), 2)
#                   cv2_imshow(blended)

                  print("saving results")
                  tempPath='/content/tempres.jpg'
                  blended = cv2.resize(blended, (521,521))
                  cv2.imwrite(tempPath, blended)
                  print("done")
                  return tempPath

    O1 = Es_mask[0]
    path = store(O1,frame)
    return path

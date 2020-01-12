from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import cv2
from PIL import Image, ImageDraw, ImageFilter
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from LIP_model import *

import random
from colorsys import rgb_to_hsv, hsv_to_rgb, hls_to_rgb, rgb_to_hls

def h_dist(h1, h2):
    """ distance between color hues in angular space,
    where 1.0 == 0.0 (so distance must wrap around if > 1)"""
    return min(abs(h1+1-h2), abs(h1-h2), abs(h1-1-h2))

def rgb2hsv(t):
    """ convert PIL-like RGB tuple (0 .. 255) to colorsys-like
    HSL tuple (0.0 .. 1.0) """
    r,g,b = t
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return rgb_to_hsv(r,g,b)

def hsv2rgb(t):
    """ convert a colorsys-like HSL tuple (0.0 .. 1.0) to a
    PIL-like RGB tuple (0 .. 255) """
    r,g,b = hsv_to_rgb(*t)
    r *= 255
    g *= 255
    b *= 255
    return (int(r),int(g),int(b))

def rgb2hls(t):
    """ convert PIL-like RGB tuple (0 .. 255) to colorsys-like
    HSL tuple (0.0 .. 1.0) """
    r,g,b = t
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return rgb_to_hls(r,g,b)

def hls2rgb(t):
    """ convert a colorsys-like HSL tuple (0.0 .. 1.0) to a
    PIL-like RGB tuple (0 .. 255) """
    r,g,b = hls_to_rgb(*t)
    r *= 255
    g *= 255
    b *= 255
    return (int(r),int(g),int(b))

def reload_img(img):
    sizew, sizeh = img.size
    maxsize = ((sizew/2)**2 + (sizeh/2)**2)**0.5
    imgdata = list(img.getdata())
    return imgdata

N_CLASSES = 20
INPUT_SIZE = (384, 384)
# DATA_DIRECTORY = './datasets/examples'
# DATA_LIST_PATH = './datasets/examples/list/val.txt'
DATA_DIRECTORY = './datasets/check_image'
DATA_LIST_PATH = './datasets/check_image/list/val.txt'
NUM_STEPS = 17 # Number of images in the validation set.
RESTORE_FROM = './checkpoint/JPPNet-s2'
OUTPUT_DIR = './output/parsing/val'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    """Create the model and start the evaluation process."""
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    h, w = INPUT_SIZE
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIRECTORY, DATA_LIST_PATH, None, False, False, coord)
        image = reader.image
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch_origin = tf.stack([image, image_rev])
    image_batch = tf.image.resize_images(image_batch_origin, [int(h), int(w)])
    image_batch075 = tf.image.resize_images(image_batch_origin, [int(h * 0.75), int(w * 0.75)])
    image_batch125 = tf.image.resize_images(image_batch_origin, [int(h * 1.25), int(w * 1.25)])
    
    # Create network.
    with tf.variable_scope('', reuse=False):
        net_100 = JPPNetModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_075 = JPPNetModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
    with tf.variable_scope('', reuse=True):
        net_125 = JPPNetModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)

    
    # parsing net
    parsing_fea1_100 = net_100.layers['res5d_branch2b_parsing']
    parsing_fea1_075 = net_075.layers['res5d_branch2b_parsing']
    parsing_fea1_125 = net_125.layers['res5d_branch2b_parsing']

    parsing_out1_100 = net_100.layers['fc1_human']
    parsing_out1_075 = net_075.layers['fc1_human']
    parsing_out1_125 = net_125.layers['fc1_human']

    # pose net
    resnet_fea_100 = net_100.layers['res4b22_relu']
    resnet_fea_075 = net_075.layers['res4b22_relu']
    resnet_fea_125 = net_125.layers['res4b22_relu']

    with tf.variable_scope('', reuse=False):
        pose_out1_100, pose_fea1_100 = pose_net(resnet_fea_100, 'fc1_pose')
        pose_out2_100, pose_fea2_100 = pose_refine(pose_out1_100, parsing_out1_100, pose_fea1_100, name='fc2_pose')
        parsing_out2_100, parsing_fea2_100 = parsing_refine(parsing_out1_100, pose_out1_100, parsing_fea1_100, name='fc2_parsing')
        parsing_out3_100, parsing_fea3_100 = parsing_refine(parsing_out2_100, pose_out2_100, parsing_fea2_100, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_075, pose_fea1_075 = pose_net(resnet_fea_075, 'fc1_pose')
        pose_out2_075, pose_fea2_075 = pose_refine(pose_out1_075, parsing_out1_075, pose_fea1_075, name='fc2_pose')
        parsing_out2_075, parsing_fea2_075 = parsing_refine(parsing_out1_075, pose_out1_075, parsing_fea1_075, name='fc2_parsing')
        parsing_out3_075, parsing_fea3_075 = parsing_refine(parsing_out2_075, pose_out2_075, parsing_fea2_075, name='fc3_parsing')

    with tf.variable_scope('', reuse=True):
        pose_out1_125, pose_fea1_125 = pose_net(resnet_fea_125, 'fc1_pose')
        pose_out2_125, pose_fea2_125 = pose_refine(pose_out1_125, parsing_out1_125, pose_fea1_125, name='fc2_pose')
        parsing_out2_125, parsing_fea2_125 = parsing_refine(parsing_out1_125, pose_out1_125, parsing_fea1_125, name='fc2_parsing')
        parsing_out3_125, parsing_fea3_125 = parsing_refine(parsing_out2_125, pose_out2_125, parsing_fea2_125, name='fc3_parsing')


    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out1_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out2_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)
    parsing_out3 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out3_100, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_075, tf.shape(image_batch_origin)[1:3,]),
                                           tf.image.resize_images(parsing_out3_125, tf.shape(image_batch_origin)[1:3,])]), axis=0)

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2, parsing_out3]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    
    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, dimension=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    rand_int = [i/100 for i in random.sample(range(0, 100), NUM_STEPS)]
    # Iterate over training steps.
    for step in range(NUM_STEPS):
        parsing_ = sess.run(pred_all)
        if step % 100 == 0:
            print('step {:d}'.format(step))
            print (image_list[step])
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]
        mask_top = decode_labels(parsing_, [5,6,7,10], num_classes=N_CLASSES)
        mask_bottom = decode_labels(parsing_, [9, 12], num_classes=N_CLASSES)
        im_mask_top = Image.fromarray(mask_top[0]).convert('L')
        im_mask_bottom = Image.fromarray(mask_bottom[0]).convert('L')
        _mask_top = np.array(im_mask_top)
        _mask_top[_mask_top != 0 ] = 255
        _mask_bottom = np.array(im_mask_bottom)
        _mask_bottom[_mask_bottom != 0 ] = 255

        real_img = Image.open(image_list[step]).convert("RGB")
        top_real_img = Image.open(image_list[step]).convert("RGB")
        bottom_real_img = Image.open(image_list[step]).convert("RGB")

        imgdata_top = reload_img(top_real_img)
        for i in range(0,len(imgdata_top)):
            # hsv code start
            (h,s,v) = rgb2hsv(imgdata_top[i])
            h += float(i)/len(imgdata_top)
            v += 0.005
            imgdata_top[i] = hsv2rgb((h,s,v))
            # hsv code end

            # hls code
            # (h,l,s) = rgb2hls(imgdata_top[i])
            # h = rand_int[step]
            # s = 0.3
            # imgdata_top[i] = hls2rgb((h,l,s))
            # hls code end
        top_real_img.putdata(imgdata_top)
        im = Image.composite(top_real_img, real_img, Image.fromarray(_mask_top).convert('L'))
        if _mask_bottom.size > 0:
            imgdata_bottom = reload_img(bottom_real_img)
            for i in range(0,len(imgdata_bottom)):
                # bottom part only use hsv to avoid not matching color with top parts
                (h,s,v) = rgb2hsv(imgdata_bottom[i])
                h += float(i)/len(imgdata_bottom)
                v += 0.005
                imgdata_bottom[i] = hsv2rgb((h,s,v))
            bottom_real_img.putdata(imgdata_bottom)
            im = Image.composite(bottom_real_img, im, Image.fromarray(_mask_bottom).convert('L'))
            Image.fromarray(_mask_bottom).save('{}/{}_bottom.png'.format(OUTPUT_DIR, img_id))
        im.save('{}/{}_hsl.png'.format(OUTPUT_DIR, img_id))
        Image.fromarray(_mask_top).save('{}/{}_top.png'.format(OUTPUT_DIR, img_id))
        
        # parsing_im.save('{}/{}_vis.png'.format(OUTPUT_DIR, img_id))
        # cv2.imwrite('{}/{}.png'.format(OUTPUT_DIR, img_id), parsing_[0,:,:,0])

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()


##############################################################333

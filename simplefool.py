#!/usr/bin/env python
# -*- coding: utf-8 -*-
##########################     Decleration     ##########################
# The code is modified from the reversed engineer work in Neural Cleanse#
# simplefool.py: generating different universal adversarial noises (UAN)
#                on different tasks. In targeted attack setting, we regard
#                each class label as target label in turn. In nontargeted
#                attack setting, we only need to avoid the true label. The
#                objective function and the solution of pattern and mask
#                are detailed in visualizer.py
#########################################################################

import os
import time
from parameters import *   #this is the parameter
import numpy as np
import random
from tensorflow import set_random_seed
import tensorflow as tf
print(tf.__version__)
random.seed(123)
np.random.seed(123)
set_random_seed(123)
import keras #from tensorflow import keras #
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from visualizer import Visualizer
import utils_simplefool
from keras.applications.inception_v3 import InceptionV3

def load_dataset(DATA_NAME,data_file=('%s/%s'%(DATA_DIR,DATA_FILE))):
    if DATA_NAME=='cifar10':
        print('loading cifar 10 .....')
        dataset = utils_simplefool.load_dataset(data_file, keys=['img'])
        X_test = np.array(dataset['img'], dtype='float32')
        print('X_test:', X_test.shape)
        Y_test = np.zeros((X_test.shape[0], NUM_CLASSES))
        for inum in range(NUM_CLASSES):
            Y_test[inum * SINGLE_NUM:inum * SINGLE_NUM + SINGLE_NUM] = [keras.utils.to_categorical(inum,NUM_CLASSES)] * SINGLE_NUM

    if DATA_NAME=='imagenet':
        print('loading imagenet .....')
        dataset = utils_simplefool.load_dataset(data_file, keys=['X_val','Y_val'])
        X_test = np.array(dataset['X_val'], dtype='float32')  # 交通標志牌對應的是X_test和Y_test
        Y_test_ini = np.array(dataset['Y_val'], dtype='float32')
        Y_test=keras.utils.to_categorical(Y_test_ini, NUM_CLASSES)

    if DATA_NAME=='gtsrb':
        print('loading gtsrb .....')
        dataset = utils_simplefool.load_dataset(data_file, keys=['X_test','Y_test'])
        X_test = np.array(dataset['X_test'], dtype='float32')  # 交通標志牌對應的是X_test和Y_test
        Y_test = np.array(dataset['Y_test'], dtype='float32')

    print('X_test shape %s' % str(X_test.shape))
    print('Y_test shape %s' % str(Y_test.shape))
    return X_test, Y_test

def build_data_loader(X, Y):
    datagen = ImageDataGenerator()   #通过实时数据增强生成张量图像数据批次。数据将不断循环（按批次）
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)
    return generator

def visualize_pattern_and_mask(visualizer, gen, y_target, save_pattern_flag=True):
    visualize_start_time = time.time()

    # initialize with random pattern and mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0  #INPUT_SHAPE:(IMG_ROWS, IMG_COLS, IMG_COLOR)
    mask = np.random.random(MASK_SHAPE)

    # visualize universal mask and pattern
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('time consuming: %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern_and_mask(DATA_NAME,pattern, mask_upsample, y_target)

    return pattern, mask_upsample, logs


def save_pattern_and_mask(dataset_name, pattern, mask, y_target):
    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % (dataset_name,'pattern', y_target)))
    utils_simplefool.dump_image(pattern, img_filename, 'png')

    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % (dataset_name,'mask', y_target)))
    utils_simplefool.dump_image(np.expand_dims(mask, axis=2) * 255,
                                img_filename,
                              'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
        '%s/%s' % (RESULT_DIR,
                   IMG_FILENAME_TEMPLATE % (dataset_name,'fusion', y_target)))
    utils_simplefool.dump_image(fusion, img_filename, 'png')

    pass


def find_universal_adversarial_noise():
    X_test,Y_test=load_dataset(DATA_NAME)
    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    if DATA_NAME=='cifar10':
        model = load_model(model_file)
    if DATA_NAME=='imagenet':
        model = InceptionV3(weights=model_file)
    if DATA_NAME == 'gtsrb':
        model = load_model(model_file)

    # initialize visualizer 初始化可视化器
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,  #MINI_BATCH:NB_SAMPLE/BATCH_SIZE
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,  #0.99
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}
    y_target_list = range(0,NUM_CLASSES)
    for y_target in y_target_list:   #regard each class label as target label in turn.
        print('\n======= Generating UAN for target label %d ========' % y_target)
        _, _, logs = visualize_pattern_and_mask(
            visualizer, test_generator, y_target=y_target,
            save_pattern_flag=True)
        log_mapping[y_target] = logs
    pass


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_simplefool.fix_gpu_memory()
    find_universal_adversarial_noise()
    pass


if __name__ == '__main__':

    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)

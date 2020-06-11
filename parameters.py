#encoding:utf-8
##########################     Decleration     ##########################
# parameters.py: The parameters are all listed below. For different task,
#                canceled corresponding comment to attack.
#########################################################################


import numpy as np
#===========================================================#
#                         For CIFAR10                      #
#===========================================================#
DATA_NAME='cifar10'
DATA_DIR = 'data/'  # data folder
DATA_FILE = 'cifar10_train_5000.h5'  # dataset file
MODEL_DIR = 'models'  # model directory
MODEL_FILENAME = 'cifar10_clean.h5py'  # model file  # gtsrb_bottom_right_white_4_target_33.h5  #cifar10_clean.h5py
RESULT_DIR = 'results/cifar10'  # directory for storing results
# image filename template for visualization results
IMG_FILENAME_TEMPLATE = '%s_UAN_%s_target_%d.png'

# input size
IMG_ROWS = 32
IMG_COLS = 32
IMG_COLOR = 3
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES =10  # total number of classes in the model  #43
Y_TARGET = 0
SINGLE_NUM=5000 #  cifar10 training dataset each class contain 5000 images
INTENSITY_RANGE = 'raw'  # preprocessing method for the task, cifar, GTSRB uses raw pixel intensities

# #===========================================================#
# #                         For ImageNet                      #
# #===========================================================#
# DATA_NAME='imagenet'
# DATA_DIR = 'data'  # data folder
# DATA_FILE = 'imagenet_train_299_10000.h5'  # dataset file   ImageNet_train_299_10000.h5
# MODEL_DIR = 'models'  # model directory
# MODEL_FILENAME = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
# RESULT_DIR = 'results/imagenet'  # directory for storing results
# # image filename template for visualization results
# IMG_FILENAME_TEMPLATE = '%s_UAN_%s_target_%d.png'
#
# # input size
# IMG_ROWS = 299
# IMG_COLS = 299
# IMG_COLOR = 3
# INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
# NUM_CLASSES =1000  # total number of classes in the model
# SINGLE_NUM=10 #  cifar10 training dataset each class contain 5000 images
# INTENSITY_RANGE = 'inception'  # preprocessing method for the task, cifar, GTSRB uses raw pixel intensities


DEVICE = '0'  # specify which GPU to use

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization
LR = 0.1  # learning rate
STEPS = 1000  # total optimization iterations 总的优化迭代次数
NB_SAMPLE = 1000  # number of samples in each mini batch  #每个minibatch中的样本数量
MINI_BATCH = NB_SAMPLE / BATCH_SIZE  # mini batch size used for early stop  #1000/32=31.25 用于计算早停的mini_batch
INIT_COST = 1e-3  # initial weight used for balancing two objectives 用于平衡两个优化目标的初始cost

REGULARIZATION = 'l1'  # reg term to control the mask's norm

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
PATIENCE = 5  # patience for adjusting weight, number of mini batches
COST_MULTIPLIER = 1.5  # multiplier for auto-control of weight (COST)   #original was 2
SAVE_LAST = False  # whether to save the last result or best result

EARLY_STOP = True  # whether to early stop
EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 2 * PATIENCE  # patience for early stop   #original:EARLY_STOP_PATIENCE = 5 * PATIENCE
# the following part is not used in our experiment
# but our code implementation also supports super-pixel mask
UPSAMPLE_SIZE = 1  # size of the super pixel
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)
#pattern_initial=np.random.random(INPUT_SHAPE) * 255.0   #added by sainan
#encoding:utf-8
import keras
import cv2 as cv
import matplotlib.pyplot as plt
from utils_simplefool import *

data_file='./data/cifar10_test_1000.h5'  #'../data/single_class/cifar10_train_5000.h5'
NUM_CLASSES =10
DATASET_label=0
SINGLE_NUM=1000
# path_template='./results/cifar10_UAN_%s_target_%d.png'

def load_cifar10_to_test_UAN():
    logf=open('./cifar10_result.txt','a+')
    dataset = load_dataset(data_file, keys=['img'])
    All_test = np.array(dataset['img'], dtype='uint8')

    print('loading model!')
    model = keras.models.load_model('./models/cifar10_clean.h5py')

    trigger_path='./results/cifar10/cifar10_UAN_fusion_target_%s.png'
    mask_path='./results/cifar10/cifar10_UAN_mask_target_%s.png'
    pattern_path='./results/cifar10/cifar10_UAN_pattern_target_%s.png'

    y_target_list = range(NUM_CLASSES)
    for idx, y_target in enumerate(y_target_list):
        X_test=All_test.copy()
        print('*********************y_target*********************', y_target)
        triggerindex = y_target
        trigger = cv.imread(trigger_path % (str(triggerindex)))
        trigger = cv.cvtColor(trigger, cv.COLOR_BGR2RGB).astype('uint8')

        pattern = cv.imread(pattern_path % (str(triggerindex)))
        pattern = cv.cvtColor(pattern, cv.COLOR_BGR2RGB).astype('uint8')

        trigger_mask = cv.imread(mask_path % (str(triggerindex)))
        trigger_mask_zeroone = trigger_mask / 255.0
        all_one = np.ones((32, 32, 3), dtype='uint8')
        reverse = all_one - trigger_mask_zeroone


        for j in range(0, X_test.shape[0]):
            X_temp = image.array_to_img(X_test[j])
            plt.subplot(2, 3, 4)
            plt.imshow(X_temp)
            plt.title('raw')

            X_test[j] = reverse * X_test[j] + pattern * trigger_mask_zeroone

            imagex = image.array_to_img(X_test[j])
            imagex.save('./adv/t_%d_img_%d.png' % (y_target, j), 'png')   #save adversarial examples

            '''start: uncomment this part for more details '''
            plt.subplot(2, 3, 1)
            plt.imshow(image.array_to_img(X_test[j]))
            plt.title('adv')

            plt.subplot(2, 3, 2)
            plt.imshow(trigger.astype('uint8'))
            plt.title('fusion')
            plt.subplot(2, 3, 3)
            plt.imshow(pattern.astype('uint8'))
            plt.title('pattern')

            plt.subplot(2, 3, 5)
            plt.imshow(reverse)
            plt.title('1-mask')
            plt.subplot(2, 3, 6)
            plt.imshow(trigger_mask_zeroone)
            plt.title('mask')
            plt.show()
            '''end: uncomment this part for more details'''

        preds = model.predict(X_test)
        print('preds', np.argmax(preds, axis=1))

        c = 0
        for k in range(X_test.shape[0]):
            if np.argmax(preds[k]) == y_target:
                c = c + 1

        Y_test_original = np.array([keras.utils.to_categorical(DATASET_label, 10)] * SINGLE_NUM)
        Y_t = np.array([keras.utils.to_categorical(y_target, 10)] * 10000)#SINGLE_NUM)

        # test_loss,test_acc=model.evaluate(X_test,Y_t)
        # print('test_loss,test_acc:',test_loss,test_acc)

        print('Final Success attack Num:%2d,  Test number:%2d,  ACC:%2f\n' % (c,  X_test.shape[0], c * 1.0 / X_test.shape[0]))
        logf.writelines('y_target:' + str(y_target) + '\t' + 'attack_num:' + str(c) + '\t' + 'acc:' + str(c * 1.0 / X_test.shape[0]) + '\n')

    logf.close()

if __name__=='__main__':
    load_cifar10_to_test_UAN()
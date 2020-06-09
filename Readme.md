This repository contains code for "SimpleFool: Simple universal adversarial noises to fool deep neural networks".

Required python tool (under Ubuntu 16.04): Python 2.7, Keras (2.2.2), TensorFlow 1.9.0, numpy, opencv(3.4.1). 



# Code usage

```
--SimpleFool--data
			  |--cifar10_train_5000.h5
			  |--cifar10_test_1000.h5
			  |--......
			--models
			  |--cifar10_clean.h5py
			--results  #X is the target class
			  |--cifar10_UAN_fusion_target_X.png    #fusion is mask*pattern
			  |--cifar10_UAN_mask_target_X.png      
			  |--cifar10_UAN_pattern_target_X.png
			  |--......
			--adv   #stored images with UAN (ie, adversarial examples)
			--parameters.py
			--simplefool.py
			--utils_simplefool.py
			--visualizer.py
			--result.txt
```



change parameters in parameters.py and then run simplefool.py directly.




import os
import sys

visible_device = '-1'

os.environ['CUDA_VISIBLE_DEVICES'] = str(visible_device)
print('debug')
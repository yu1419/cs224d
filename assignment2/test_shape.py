import os
import getpass
import sys
import time
import numpy as np
import numpy as np
import tensorflow as tf
from q2_initialization import xavier_weight_init
import data_utils.utils as du
import data_utils.ner as ner
from utils import data_iterator
from model import LanguageModel

a = [1,2,3]
a = np.array(a)
a = np.reshape(a,(a.size,1))
with tf.Session() as s:

    print(a.shape)


print(a.tolist())
t=[]
for x in a:
    t.extend(x)
print(t)
a = tf.constant([[1,2,3],[2.22,2,2]])
with tf.Session():
    print(a[-1,:].eval())

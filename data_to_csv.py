import pandas as pd
import tensorflow as tf
import os

labels = ['food','not food']
datset = ['TRAIN', 'Validation', 'TEST']
data = []
for s in dataset:
    path = os.path.join('gs://food-resnet/dataset/',s)
    for l in labels:
        image_path = tf.gfile.Glob(os.path.join(path, l)):
            for file_path in tf.gfile.Glob(os.path.join(image_path,'/*')):
                data.append(s, file_path, l)

pd_data = pd.DataFrame(data)
pd_data.to_csv('data.csv', index=False, header=False, )

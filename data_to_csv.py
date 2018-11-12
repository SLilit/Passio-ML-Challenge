import pandas as pd
import tensorflow as tf
import os
import datalab.storage as gcs

labels = ['food','not food']
dataset = ['VALIDATION', 'TEST', 'TRAIN']
data = []
for s in dataset:
    path = os.path.join('gs://food_resnet/dataset/',s)
    for l in labels:
        for file_path in tf.gfile.Glob(os.path.join(path, l+'/*')):
            data.append((s, file_path, l))

pd_data = pd.DataFrame(data)
gcs.Bucket('food_resnet').item('dataset/data.csv').write_to(pd_data.to_csv(index=False, header=False),'text/csv')
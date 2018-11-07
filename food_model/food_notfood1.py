import numpy as np
import os
import tensorflow as tf 
#import google.datalab.storage as storage
from google.cloud import storage

tf.logging.set_verbosity(tf.logging.INFO)

client = storage.Client()
bucket = client.bucket('food-221614')
for blob in bucket.list_blobs(prefix='food_model/test/food'):
    print (os.path.join('gs://food-221614', blob.name))
       
#Data reading and preprocessing function
def imgs_input_fn(data, mode, batch_size=15):
    
    imagepaths, imagelabels = [], []
    buffer_size = 0
    client = storage.Client()
    bucket = client.bucket('food-221614')
    for blob in bucket.list_blobs(prefix=os.path.join(data,'food')):
        imagepaths.append(os.path.join('gs://food-221614', blob.name))
        imagelabels.append(0)
        buffer_size += 1
    
    for blob in bucket.list_blobs(prefix=os.path.join(data,'not food')):
        imagepaths.append(os.path.join('gs://food-221614', blob.name))
        imagelabels.append(1)
        buffer_size += 1
   
    imagepaths = tf.constant(imagepaths)
    imagelabels = tf.constant(imagelabels)
    
    def _parse_function(file, label):
        image_string = tf.read_file(file)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded,[244,244])
        image_resized.set_shape([244, 244, 3])
        
        return image_resized, label
    
    def augment(image, label):
        augment_image = tf.random_crop(image, [244,244,3])
        augment_image = tf.image.random_flip_left_right(image)
        augment_image = tf.contrib.image.rotate(augment_image, 50)
        
        return augment_image, label
        
    dataset = tf.data.Dataset.from_tensor_slices((imagepaths,imagelabels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    
    #Create batches 
    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None
        dataset = dataset.map(augment)
        dataset = dataset.repeat(num_epochs).shuffle(buffer_size=buffer_size)
    else:
        num_epochs = 1
    
    dataset = dataset.repeat(num_epochs).batch(batch_size)    
    iterator = dataset.make_one_shot_iterator() 
    features, labels = iterator.get_next()
    labels = tf.reshape(labels,[-1, 1])
    
    return features, labels

#def reset_graph(seed=42):
 #   tf.reset_default_graph()
  #  tf.set_random_seed(2)
   # np.random.seed(2)

#reset_graph()

def cnn_model_fn(features, labels, mode):
    
    input_layer = tf.reshape(features, [-1 , 244, 244, 3])
    
    #Convolutional and Pooling layers
    conv1 = tf.layers.conv2d(input_layer, 16, 3, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2)

    conv2 = tf.layers.conv2d(pool1, 32, 3, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)

    conv3 = tf.layers.conv2d(pool2, 64, 3, activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    
    #Dense layer
    flat = tf.contrib.layers.flatten(pool3)
    dense = tf.layers.dense(flat, 64, activation=tf.nn.relu)
    drop = tf.layers.dropout(dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    #Logits layer
    logits = tf.layers.dense(drop, 2)
    
    #Generate predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1),"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    #Calculate Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    
    #Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op) 
    
    #Add evaluation metrics
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


#food_classifier = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir='food_model')
eval_int = 50

food_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir='gs://food-221614/food_model/food_notfood_trained',
                                        config=tf.estimator.RunConfig(save_checkpoints_secs=eval_int))  

train_spec = tf.estimator.TrainSpec(input_fn= lambda: imgs_input_fn('food_model/train', mode=tf.estimator.ModeKeys.TRAIN), max_steps=60)

#exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

eval_spec = tf.estimator.EvalSpec(input_fn= lambda: imgs_input_fn('food_model/test', mode=tf.estimator.ModeKeys.EVAL),
                                   start_delay_secs=eval_int, throttle_secs=eval_int)
    
tf.estimator.train_and_evaluate(food_classifier, train_spec, eval_spec)


#Pred data reading and preprocessing function
#def img_input_fn(path):
#    imagepaths = []
#    
#    images = os.walk(path).__next__()[2]
#    for img in images:
#        img_path = os.path.join(path,img)
#        imagepaths.append(img_path)
#       
#    imagepaths = tf.constant(imagepaths)
#    
#    def _parse_function(file):
#        image_string = tf.read_file(file)
#        image_decoded = tf.image.decode_jpeg(image_string)
#        image_resized = tf.image.resize_images(image_decoded,[244,244])
#        image_resized.set_shape([244, 244, 3])
#        return image_resized
#    
#    dataset = tf.data.Dataset.from_tensor_slices(imagepaths)
#    dataset = dataset.map(_parse_function)
#    
#    iterator = dataset.make_one_shot_iterator() 
#    features = iterator.get_next()
#    
#    return features

#pred = food_classifier.predict(input_fn=lambda: img_input_fn("n"))

#for prediction in pred:
   
#    if prediction["classes"] == 0:
#        print("Food")
#    else:
#        print("Not Food")


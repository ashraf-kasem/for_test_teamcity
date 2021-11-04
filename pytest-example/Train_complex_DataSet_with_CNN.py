# importing important libraries

import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import ntpath
import random
import tensorflow as tf
from tensorflow import keras
import time
import os


# a function to build a CNN model based on Nvidia Model
def nvidia_model():
    # build a layered model
    model = Sequential()
    # adding 5 convolutional layers with input size equal to 66*200 and using elu as
    # activiation function, which is bettewr than relu
    # 24 is the number of filters in the first lyaer
    # 5*5 is the size of each filter
    # subsample is 2*2, is the way how the filter move each step
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 1), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # not in original model. added for more robustness
    # model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    # to flatten the data to be suitable for the fully conected layer
    model.add(Flatten())
    # not in original model. added for more robustness
    model.add(Dropout(0.2))
    # adding one layer contain of 100 neurons
    model.add(Dense(100, activation='elu'))
    # to avoid overfitting we use dropout layer, which cut to half
    # the number of neurons every iteration when training
    # also the importance of this layer is to generelise
    # the model for new data. 0.5 means half of the nodes will be dropped
    # model.add(Dropout(0.5))
    # adding one layer contain of 50 neurons
    model.add(Dense(50, activation='elu'))
    # model.add(Dropout(0.5))
    # adding one layer contain of 10 neurons
    model.add(Dense(10, activation='elu'))
    #   model.add(Dropout(0.5))
    # adding one output layer
    model.add(Dense(1))
    # defining the Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=1e-3)
    # Defining the loss function as menimum squer error
    model.compile(loss='mse', optimizer=optimizer)
    return model


# a function to preprocess all data set images
def img_preprocess(img):
    # first crop the image and keep just the area of interest
    # crop just the height (pixel 16 to pixel 135)
    img = img[120:, :, :]
    # convert the color type to YUV, as recommended by Nvidia
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # apply gaussian blur to smooth the image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # resize to 200*66 as recommended in Nvidia model
    img = cv2.resize(img, (200, 66))
    # for normalization, each pixel intensity is divided by 255
    img = img / 255
    return img


# a function to preprocess all data set images but detected for the map function
def parse_function(filename, label):
    img_raw = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_raw)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [66, 200])
    layer = tf.keras.layers.Rescaling(scale=1. / 255)
    img = layer(img)
    # print(img.shape)
    return img, label

# the following functions are parts of the main parse function
# which aim to replace the main parse function for testing purposes

# for rescaling and normalization
def normalize_image(im):
    # convert the image to a float datatype then divide by 255
    normalized_image = tf.cast(im, tf.float32) / 255.0
    return normalized_image








# a user defined function to work as a generator
# define a function to generate batches to run with the training process
# this will improve storage use, but it will affect the training process
# parameters: images paths, the responding steering angles, batch size
def batch_generator(image_paths, steering_ang, batch_size):
    while True:
        # build two lists to save the batch images and corresponding steering
        batch_img = []
        batch_steering = []
        # loop with the length of the batch to read the mages from the desk and
        # do the preprocessing ( I think here is the most input time is wasted )
        for i in range(batch_size):
            # generate random index
            random_index = random.randint(0, len(image_paths) - 1)
            # read the image based on the previous index
            im = mpimg.imread(image_paths[random_index], format='jpeg')
            # read the steering based on the previous index
            steering = steering_ang[random_index]
            # calling the preprocess function
            im = img_preprocess(im)
            # append the preprocessed mage to the batch list
            batch_img.append(im)
            # append the steering to the batch list
            batch_steering.append(steering)
            # converting the image to numpy array
            batch_img_arr = np.asarray(batch_img)
            # reshape to fit the model input ( 66,200,1  )
            # we used -1 for the first dimension because this is a batch of images
            # so the dimension should be after completing all iterations ( batchSize , 66, 200, 1)
            # so during the iteration we can use -1 (anything), or (i+1)
            batch_img_arr = batch_img_arr.reshape(-1, 66, 200, 1)
        # yield the batch images and the batch steering
        yield batch_img_arr, np.asarray(batch_steering)


# a function to read all steering and paths for images
# and put them in suitable arrays to deal with them correctly
def load_img_steering(datadir, data):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center = indexed_data[0]
        # center image append
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[1]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings


# user defined function to create a dataset, not simple
def create_iter_dataset() -> object:
    # reading the csv driving log, which contain the paths to all images
    # we used panda library
    datadir = 'Complex_dataSet'
    data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'))
    pd.set_option('display.max_colwidth', None)
    # to display some lines from the csv file
    # print(data.head())
    # function to split the long path of the images, and to leave just the names
    # of the images |  lambda path: ntpath.split(path)[1] |
    # updating the values of the csv data with the new images names
    data['Im_paths'] = data['Im_paths'].apply(lambda path: ntpath.split(path)[1])
    # print(data.head())
    # calling the previous function to load all steering and images paths data
    image_paths, steerings = load_img_steering(datadir + '/IMG', data)
    # build a training and validation data by splitting the data set
    # 20 percent for validation and the rest for training
    # of course with their labels ( steering angle )
    # random_state=6 just a random seed
    x_train, x_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.15, random_state=6)
    # print('Training Samples: {}\nValid Samples: {}'.format(len(x_train), len(x_valid)))

    # ask if you want prefetching and caching
    # use_prefetching_caching = input(" Do you want to use prefetching and caching ? y/n ")
    use_prefetching_caching = 'y'

    if use_prefetching_caching == 'y':
        # with prefetch and caching
        # ask if you want Maping
        # with_map = input(" Do you want to use DataSet Maping function ? y/n ")
        with_map = 'y'
        if with_map == 'y':
            # with map function
            # first convert the images paths and the labels an object from the Dataset class to the training set
            images = tf.constant(x_train)
            labels = tf.constant(y_train)
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.shuffle(len(x_train))
            dataset = dataset.repeat()
            dataset = dataset.map(parse_function).batch(32).cache().prefetch(tf.data.experimental.AUTOTUNE)
            # for images, labels in dataset.take(1):
            #     print(labels.shape)
            #     print(images.shape)

            # first convert the images paths and the labels an object from the Dataset class to the validation set
            images = tf.constant(x_valid)
            labels = tf.constant(y_valid)
            v_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            v_dataset = v_dataset.shuffle(len(x_valid))
            v_dataset = v_dataset.repeat()
            v_dataset = v_dataset.map(parse_function).batch(32).cache().prefetch(tf.data.experimental.AUTOTUNE)
            # for images, labels in v_dataset.take(1):
            #     print(labels.shape)
            #     print(images.shape)


        # without mapping
        else:
            # generating the dataset using from_generator function
            # parameters: "methode to work as a generator", output_types should represent the type of
            # training data and the labels, output_shapes: to represent the shape of the training data
            # and the labels, args are the parameters for the generator function
            dataset = tf.data.Dataset.from_generator(
                batch_generator, output_types=(tf.float64, tf.float64),
                output_shapes=((32, 66, 200, 1), (32,)),
                args=(x_train, y_train, 32)).prefetch(
                tf.data.experimental.AUTOTUNE).cache()
            # using the same generator to generate the validation set
            v_dataset = tf.data.Dataset.from_generator(
                batch_generator, output_types=(tf.float64, tf.float64),
                output_shapes=((32, 66, 200, 1), (32,)),
                args=(x_valid, y_valid, 32)).prefetch(
                tf.data.experimental.AUTOTUNE).cache()


    # without prefetching and caching
    else:
        # ask if you want to use mapping
        with_map = input(" Do you want to use DataSet Maping function ? y/n ")
        if with_map == 'y':
            # with map function
            # first convert the images paths and the labels an object from the Dataset class to the training set
            images = tf.constant(x_train)
            labels = tf.constant(y_train)
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            dataset = dataset.shuffle(len(x_train))
            dataset = dataset.repeat()
            dataset = dataset.map(parse_function).batch(32)
            # for images, labels in dataset.take(1):
            #     print(labels.shape)
            #     print(images.shape)

            # first convert the images paths and the labels an object from the Dataset class to the validation set
            images = tf.constant(x_valid)
            labels = tf.constant(y_valid)
            v_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            v_dataset = v_dataset.shuffle(len(x_valid))
            v_dataset = v_dataset.repeat()
            v_dataset = v_dataset.map(parse_function).batch(32)
            # for images, labels in v_dataset.take(1):
            #     print(labels.shape)
            #     print(images.shape)


        # without mapping
        else:
            # generating the dataset using from_generator function
            # parameters: "methode to work as a generator", output_types should represent the type of
            # training data and the labels, output_shapes: to represent the shape of the training data
            # and the labels, args are the parameters for the generator function
            dataset = tf.data.Dataset.from_generator(
                batch_generator, output_types=(tf.float64, tf.float64),
                output_shapes=((32, 66, 200, 1), (32,)),
                args=(x_train, y_train, 32))
            # using the same generator to generate the validation set
            v_dataset = tf.data.Dataset.from_generator(
                batch_generator, output_types=(tf.float64, tf.float64),
                output_shapes=((32, 66, 200, 1), (32,)),
                args=(x_valid, y_valid, 32))

    # return the two datasets
    return dataset, v_dataset


# function to create callbacks list for logging and monitoring
def create_callbacks(log_dir):
    # callbacks list contains callbacks objects as desired, as a start
    # we want to use TensorBoard virtualization tool to show the logs
    # so, the first object to the callbacks list is built using TensorBoard class
    # profile_batch specify from which patch you want to start profiling till what batch
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=log_dir,
                                    write_graph=True,
                                    write_images=True,
                                    profile_batch=(1, 100)),
    ]
    # return the callbacks objects list
    return callbacks


# to build the model and define the training process
def train(log_dir, epochs=1):
    # user defined function to build the model
    model = nvidia_model()
    # User defined function to create callbacks list for logging and monitoring
    callbacks = create_callbacks(log_dir=log_dir)
    # using normal generator without prefetch and cashing
    dataset, validation_set = create_iter_dataset()
    # because the input is generator we dont need a labels, lables will be
    # obtained from the input dss
    print("complex")
    # start the training process with the batch generator for the with training agumetation process
    # epochs=10, means number of iterations
    history = model.fit(dataset,
                        batch_size=32,
                        steps_per_epoch=100,
                        epochs=epochs,
                        validation_data=validation_set,
                        validation_steps=50,
                        verbose=1,
                        shuffle=1,
                        callbacks=callbacks)
    return model


# the main function, Start from here
def main():
    log_dir = 'logs'
    experiment_name = time.strftime('%Y-%m-%d-%H-%M')
    log_path = os.path.join(log_dir, experiment_name)

    detect_nan = {
        'use_check_numerics': False,  # use either enable_check_numerics ...
        'use_dump_debug_info': False,  # .. or enable_dump_debug_info, not both!
        'use_keras_callback': False,
    }
    # use either enable_check_numerics ...
    if detect_nan['use_check_numerics']:
        tf.debugging.enable_check_numerics(stack_height_limit=300, path_length_limit=500)
        # catching NaN nicely

    elif detect_nan['use_dump_debug_info']:
        # use either .. or enable_dump_debug_info, not both!
        w = tf.debugging.experimental.enable_dump_debug_info(log_path, tensor_debug_mode="FULL_HEALTH",
                                                             circular_buffer_size=-1)
        # TODO: returns a writer, w.FlushNonExecutionFiles() and w.FlushExecutionFiles()
        w.FlushExecutionFiles()
    # Calling tf.config.run_functions_eagerly(True) will make all
    # invocations of tf.function run eagerly instead of running as a traced graph function.
    tf.config.run_functions_eagerly(False)
    # start by calling the train method
    train(epochs=2, log_dir=log_path)


# to make sure that if this file is imported, the main function will not run
if __name__ == '__main__':
    main()

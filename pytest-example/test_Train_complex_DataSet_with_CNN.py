import pytest
import random
import Train_complex_DataSet_with_CNN
import tensorflow as tf
import numpy as np


# to test the dimensions of the processed images and labels
# in our model the image should be (66,200,1) and the label it should be a float64 scalar
@pytest.mark.dataset
def test_dataset():
    # get dataset and validation set
    dataset, val_set = Train_complex_DataSet_with_CNN.create_iter_dataset()
    # get one patch from the dataset
    for images, labels in dataset.take(1):
        # check one item from the images patch
        assert images[1].shape == (66, 200, 1)
        # check one item from the labels patch
        assert labels[1].dtype == 'float64' and labels[1].shape == ()


# to test the preprocessing function
# using input_value fixture function
@pytest.mark.preprocessing
def test_preprocessing(input_value):
    # the fixture function provide path and label
    image_path, label = input_value
    # use this path and label as a test for the image preprocessing function
    im, label = Train_complex_DataSet_with_CNN.parse_function(image_path, label)
    im_values = im.numpy()
    rand_value = random.randint(1, (66 * 200))
    # will not print out unless the -s option is activated
    # print(im_values.item(rand_value))
    # check shape and type
    assert im.shape == (66, 200, 1) and tf.is_tensor(im)
    # check scale if its between 0 and 1
    assert 0 <= im_values.item(rand_value) <= 1


# new tests on a separated preprocessing functions
# the general preprocessing function will be partitioned to
# many smaller functions, each one of them will be specific
# to do one transformation on the image

# test normalisation
@pytest.mark.norm
def test_normalize_image():
    # specify array of shape 2*2 all ones
    im = np.array([[1.0, 1.0], [1.0, 1.0]])
    # the expected image values after division
    expected_image = np.array([[0.00392157, 0.00392157], [0.00392157, 0.00392157]])
    # test the normalize_image with the  2*2 all ones image array
    result = Train_complex_DataSet_with_CNN.normalize_image(im).numpy()
    # some printing to check which will noy be shown when testing, only if -s option is on
    print("result", result)
    print("expected result", expected_image)
    # the test will give True if the number are close
    assert np.allclose(result, expected_image)





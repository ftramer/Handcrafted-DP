"""
Adapted from https://colab.research.google.com/github/google-research/simclr/blob/master/colabs/finetuning.ipynb
"""

import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
from sklearn.linear_model import LogisticRegression
import os


CROP_PROPORTION = 0.875  # Standard for ImageNet.


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf.cond(
      tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32),
              tf.cast(p, tf.float32)),
      lambda: func(x),
      lambda: x)


def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.
  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.
  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf.cast(tf.rint(
        crop_proportion / aspect_ratio * image_width_float), tf.int32)
    crop_width = tf.cast(tf.rint(
        crop_proportion * image_width_float), tf.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf.cast(
        tf.rint(crop_proportion * image_height_float), tf.int32)
    crop_width = tf.cast(tf.rint(
        crop_proportion * aspect_ratio *
        image_height_float), tf.int32)
    return crop_height, crop_width

  return tf.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.
  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.image.resize_bicubic([image], [height, width])[0]

  return image


def preprocess_for_eval(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.
  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf.reshape(image, [height, width, 3])
  image = tf.clip_by_value(image, 0., 1.)
  return image


def preprocess_image(image, height, width, is_training=False,
                     color_distort=True, test_crop=True):
  """Preprocesses the given image.
  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.
  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  #if is_training:
  #  return preprocess_for_train(image, height, width, color_distort)
  #else:
  return preprocess_for_eval(image, height, width, test_crop)


(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
xtrain = xtrain.astype(np.float32) / 255.0
xtest = xtest.astype(np.float32) / 255.0
ytrain = ytrain.reshape(-1)
ytest = ytest.reshape(-1)


def _preprocess(x):
  x = preprocess_image(x, 224, 224, is_training=False, color_distort=False)
  return x


batch_size = 100
x = tf.placeholder(shape=(batch_size, 32, 32, 3), dtype=tf.float32)
x_preproc = tf.map_fn(_preprocess, x)
print(x_preproc.get_shape().as_list())

hub_path = 'gs://simclr-checkpoints/simclrv2/pretrained/r50_2x_sk1/hub/'
module = hub.Module(hub_path, trainable=False)
features = module(inputs=x_preproc, signature='default')
print(features.get_shape().as_list())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("model loaded!")

features_train = []
for i in range(len(xtrain)//batch_size):
    x_batch = xtrain[i*batch_size:(i+1)*batch_size]
    f = sess.run(features, feed_dict={x: x_batch})
    features_train.append(f)

features_train = np.concatenate(features_train, axis=0)
print(features_train.shape)

features_test = []
for i in range(len(xtest)//batch_size):
    x_batch = xtest[i*batch_size:(i+1)*batch_size]
    f = sess.run(features, feed_dict={x: x_batch})
    features_test.append(f)

features_test = np.concatenate(features_test, axis=0)
print(features_test.shape)

os.makedirs("transfer/features/", exist_ok=True)
np.save("transfer/features/simclr_r50_2x_sk1_train.npy", features_train)
np.save("transfer/features/simclr_r50_2x_sk1_test.npy", features_test)

mean = np.mean(features_train, axis=0)
var = np.var(features_train, axis=0)

features_train_norm = (features_train - mean) / np.sqrt(var + 1e-5)
features_test_norm = (features_test - mean) / np.sqrt(var + 1e-5)

for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train, ytrain)
    print(C, clf.score(features_train, ytrain), clf.score(features_test, ytest))

    clf = LogisticRegression(random_state=0, max_iter=1000, C=C).fit(features_train_norm, ytrain)
    print(C, clf.score(features_train_norm, ytrain), clf.score(features_test_norm, ytest))

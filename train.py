from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib.request
import tarfile
import hashlib
import struct
import random
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from six.moves import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    if not os.path.exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = next(os.walk(image_dir))[1]
    for sub_dir in sub_dirs:
        if sub_dir == image_dir:
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                             (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    if label_name not in image_lists:
        print('Label does not exist {}.'.format(label_name))
    label_lists = image_lists[label_name]
    if category not in label_lists:
        print('Category does not exist {}.'.format(category))
    category_list = label_lists[category]
    if not category_list:
        print('Label {} has no images in the category {}.'.format(label_name, category))
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category):
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def create_inception_graph(model_dir):
    graph = tf.Graph()
    with graph.as_default():
        model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
        with tf.io.gfile.GFile(model_filename, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()  # For compatibility with TensorFlow 2.x
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    'pool_3/_reshape:0', 'DecodeJpeg/contents:0',
                    'ResizeBilinear:0']))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# Example usage:
model_dir = 'inception/'
with tf.compat.v1.Session() as sess:
    graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = create_inception_graph(model_dir)
    # Then you can use these tensors and graph for further processing


def maybe_download_and_extract():
    dest_directory = FLAGS.model_dir
    os.makedirs(dest_directory, exist_ok=True)  # Create directory if it doesn't exist
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(dest_directory)

def ensure_dir_exists(dir_name):
    os.makedirs(dir_name, exist_ok=True)  # Create directory if it doesn't exist


def write_list_of_floats_to_file(list_of_floats, file_path):
    s = struct.pack('d' * len(list_of_floats), *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)

def read_list_of_floats_from_file(file_path):
    with open(file_path, 'rb') as f:
        s = f.read()
        num_floats = len(s) // struct.calcsize('d')
        return struct.unpack('d' * num_floats, s)

bottleneck_path_2_bottleneck_values = {}

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor):
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
    if not os.path.exists(image_path):
        raise FileNotFoundError('File does not exist {}'.format(image_path))
    image_data = gfile.GFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file {}: {}'.format(image_path, e))

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_dir, category, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print('{} bottleneck files created.'.format(how_many_bottlenecks))


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                            image_index, image_dir, category,
                                            bottleneck_dir, jpeg_data_tensor,
                                            bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_dir, category,
                                              bottleneck_dir, jpeg_data_tensor,
                                              bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
        if not gfile.Exists(image_path):
            raise FileNotFoundError('File does not exist {}'.format(image_path))
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
        bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_input_tensor,
                                         bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):

    jpeg_data = tf.keras.layers.Input(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random.uniform(shape=(), minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_shape = tf.cast(tf.multiply(tf.cast(tf.shape(decoded_image_4d)[1:3], tf.float32), scale_value), tf.int32)
    precropped_image = tf.image.resize(decoded_image_4d, precrop_shape)
    cropped_image = tf.image.random_crop(precropped_image,
                                         [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                          MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random.uniform(shape=(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result

train_summaries = []

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar('mean', mean)
        train_summaries.append(mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.compat.v1.summary.scalar('stddev', stddev)
        tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
        tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
        tf.compat.v1.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    
    with tf.name_scope('input'):
        bottleneck_input = tf.compat.v1.placeholder_with_default(
                bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
                name='BottleneckInputPlaceholder')

        ground_truth_input = tf.compat.v1.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')
    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.random.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                           stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.compat.v1.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.compat.v1.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
              final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.compat.v1.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.io.gfile.exists(FLAGS.summaries_dir):
        tf.io.gfile.rmtree(FLAGS.summaries_dir)
    tf.io.gfile.makedirs(FLAGS.summaries_dir)
    # Set up the pre-trained graph.
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
            create_inception_graph(model_dir))

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        print('Only one valid folder of images found at ' + FLAGS.image_dir +
              ' - multiple classes are needed for classification.')
        return -1

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)

    with tf.compat.v1.Session(graph=graph) as sess:

        if do_distort_images:
            # We will be applying distortions, so setup the operations we'll need.
            (distorted_jpeg_data_tensor,
             distorted_image_tensor) = add_input_distortions(
                     FLAGS.flip_left_right, FLAGS.random_crop,
                     FLAGS.random_scale, FLAGS.random_brightness)
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        bottleneck_tensor)

        # Add the new layer that we'll be training.
        (train_step, cross_entropy, bottleneck_input, ground_truth_input,
         final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(
                final_tensor, ground_truth_input)

        # default_graph = tf.compat.v1.get_default_graph()
        # print("Operations in the default graph:")
        # for op in default_graph.get_operations():
        #     print(op.name)

        

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)

        validation_writer = tf.compat.v1.summary.FileWriter(
                FLAGS.summaries_dir + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if do_distort_images:
                (train_bottlenecks,
                 train_ground_truth) = get_random_distorted_bottlenecks(
                         sess, image_lists, FLAGS.train_batch_size, 'training',
                         FLAGS.image_dir, distorted_jpeg_data_tensor,
                         distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks,
                 train_ground_truth, _) = get_random_cached_bottlenecks(
                         sess, image_lists, FLAGS.train_batch_size, 'training',
                         FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                         bottleneck_tensor)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.

            
            train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={bottleneck_input: train_bottlenecks,
                                   ground_truth_input: train_ground_truth})
                validation_bottlenecks, validation_ground_truth, _ = (
                        get_random_cached_bottlenecks(
                                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                                bottleneck_tensor))
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={bottleneck_input: validation_bottlenecks,
                                   ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                print('Step: %d, Train accuracy: %.4f%%, Cross entropy: %f, Validation accuracy: %.1f%% (N=%d)' % (i,
                        train_accuracy * 100, cross_entropy_value, validation_accuracy * 100, len(validation_bottlenecks)))

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (
                get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
        test_accuracy, predictions = sess.run(
                [evaluation_step, prediction],
                feed_dict={bottleneck_input: test_bottlenecks,
                           ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (
                test_accuracy * 100, len(test_bottlenecks)))

        if FLAGS.print_misclassified_test_images:
            print('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    print('%70s  %s' % (test_filename,
                              list(image_lists.keys())[predictions[i]]))

        # Write out the trained graph and labels with the weights stored as
        # constants.
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
        with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
        )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='logs/output_graph.pb',
        help='Where to save the trained graph.'
        )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='logs/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
        )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='logs/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
        )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=5000,
        help='How many training steps to run before ending.'
        )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
        )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
        )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
        )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=100,
        help='How often to evaluate the training results.'
        )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
        )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
        How many images to test on. This test set is only used once, to evaluate
        the final accuracy of the model after training completes.
        A value of -1 causes the entire test set to be used, which leads to more
        stable results across runs.\
        """
        )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
        How many images to use in an evaluation batch. This validation set is
        used much more often than the test set, and is an early indicator of how
        accurate the model is during training.
        A value of -1 causes the entire validation set to be used, which leads to
        more stable results across training iterations, but may be slower on large
        training sets.\
        """
        )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
        Whether to print out a list of all misclassified test images.\
        """,
        action='store_true'
        )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='logs/imagenet',
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """
        )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
        )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
        The name of the output classification layer in the retrained graph.\
        """
        )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
        Whether to randomly flip half of the training images horizontally.\
        """,
        action='store_true'
        )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
        A percentage determining how much of a margin to randomly crop off the
        training images.\
        """
        )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly scale up the size of the
        training images by.\
        """
        )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
        A percentage determining how much to randomly multiply the training image
        input pixels up or down by.\
        """
        )
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
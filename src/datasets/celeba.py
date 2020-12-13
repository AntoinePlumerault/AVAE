# To use this dataset you must follow this procedure:
#
# 1) Download Align&Cropped images from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#    Direct link: https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
# 2) Extract `img_align_celeba_zip` into `../data/`.
# 4) Open a terminal and place yourself in the `src` directory with the `cd` command.
# 3) Execute the command python3 dataset/celeba.py to create the tfrecords files.
# 4) Then this dataset should be usable as the other ones.

import os 
import tensorflow as tf 

from absl import app, flags, logging
from random import choice


def load(batch_size=1, shuffle_and_repeat=True, mode='train'):
    if mode != 'train':
        shuffle_and_repeat = False
        
    if mode == 'train':
        tfrecords = [os.path.join(
            '..', 'data', 'celeba', 'tfrecords', 
            'celeba.tfrecord.{:05d}-of-{:05d}'.format(fold+1, 10)) 
            for fold in range(9)]
    
    if mode == 'test': 
        tfrecords = [os.path.join(
            '..', 'data', 'celeba', 'tfrecords', 
            'celeba.tfrecord.{:05d}-of-{:05d}'.format(10, 10)) 
        ]
    
    def _decode(example):
        features = tf.io.parse_single_example(example, features={
            'image': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
            'path' : tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        })
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.image.central_crop(image, 0.6)
        image = tf.image.resize([image], [64, 64])
        image = image[0] / 255. * 2 - 1
        path = tf.cast(features['path'], tf.string)

        return {'image': image, 'path': path}

    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=len(tfrecords)) 
    dataset = dataset.repeat() if shuffle_and_repeat else dataset
    dataset = dataset.shuffle(200000) if shuffle_and_repeat else dataset
    dataset = dataset.map(_decode, num_parallel_calls=8)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)
    
    return dataset, (64,64,3)


def main(argv):

    dataset_dir = os.path.join(FLAGS.data_dir, 'celeba')
    print(dataset_dir)
    def _int64list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    os.makedirs(os.path.join(dataset_dir, 'tfrecords'), exist_ok=True)
    writers = [tf.io.TFRecordWriter(os.path.join(
        dataset_dir, 'tfrecords', 'celeba.tfrecord.{:05d}-of-{:05d}'.format(fold+1, 10)
    )) for fold in range(10)]
    
    images = {}
    directory = os.path.join(dataset_dir, 'img_align_celeba')
    print(os.listdir(directory))
    for image in os.listdir(directory):
        if image[-4:] == '.jpg':
            images[image]  = {
                'path': os.path.join(directory, image),
            }

    # Graph for reading files
    @tf.function
    def read_image(path):
        image_string = tf.io.read_file(path)
        return image_string
    
    for image_name in list(images.keys()):
        writer = choice(writers)
        image = images[image_name]
        image_data = open(image['path'], 'rb').read()
        writer.write(tf.train.Example(features=tf.train.Features(feature = {
            'image': _bytes_feature(tf.compat.as_bytes(image_data)),
        })).SerializeToString())

    for writer in writers: writer.close()

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('data_dir', os.path.join('..', 'data'), 'the directory where the images are stored')
    
    app.run(main)
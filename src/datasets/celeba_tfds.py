# This dataset may not work as there is a problem with the checksum verification
# when downloading the data with tensorflow-datasets.

import os
import tensorflow as tf 
import tensorflow_datasets as tfds

from absl import flags, app



FLAGS = flags.FLAGS

def load(batch_size=1, shuffle_and_repeat=True, mode='train'):  

    name = 'celeb_a'
    data_dir = os.path.join(FLAGS.data_dir, name)

    if mode != 'train':
        shuffle_and_repeat = False
    
    dataset = tfds.load(
        name=name+':2.*.*',
        split=mode,
        data_dir=data_dir,
        batch_size=None,
        shuffle_files=True,
        download=True,
        as_supervised=False)

    def prepare(features):
        image = tf.cast(features['image'], tf.float32) / 255. * 2 - 1 
        image = tf.image.central_crop(image, 0.6)
        image = tf.image.resize(image, (64, 64))
        return {'image': image}
    
    dataset = dataset.map(prepare, num_parallel_calls=8)
    dataset = dataset.repeat() if shuffle_and_repeat else dataset
    dataset = dataset.shuffle(30000) if shuffle_and_repeat else dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)
    
    return dataset, (64,64,3)
    
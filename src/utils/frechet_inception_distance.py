# (FID) Gans trained by a two time-scale update rule converge to a local nash equilibrium.
import tensorflow as tf
import numpy as np
from scipy import linalg
import tensorflow_hub as hub

# model.build([None, 299, 299, 3])
from absl import flags

FLAGS = flags.FLAGS

class FrechetInceptionDistance:
    def __init__(self, dataset, N):
        # take images between [-1, 1]
        self.model = m = tf.keras.Sequential([hub.KerasLayer(
            "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", 
            output_shape=[2048],
            trainable=False)]) # images in [0,1]
        self.mu, self.sigma = 0, 0 
        feature_vectors = []
        N_batch = (N+FLAGS.batch_size-1)//FLAGS.batch_size
        N = N_batch*FLAGS.batch_size
        for step, features in dataset.take(N_batch).enumerate():
            images = (tf.image.resize(features['image'], (299, 299)) + 1) / 2
            feature_vectors.append(self.model(images))
            if step % 1000 == 0 or step == (N_batch-1):
                feature_vectors = tf.concat(feature_vectors, axis=0).numpy()
                self.mu = np.sum(feature_vectors, axis=0) + self.mu
                self.sigma = feature_vectors.transpose().dot(feature_vectors) + self.sigma
                feature_vectors = []
        self.mu /= N
        self.sigma /= (N - 1)
        self.sigma -= np.reshape(self.mu, [-1,1]).dot(np.reshape(self.mu, [1,-1]))


    def __call__(self, images):
        # take images between [-1, 1]
        mu, sigma = 0, 0
        feature_vectors = []
        N = len(images)*FLAGS.batch_size
        for step, image in enumerate(images):
            image = (tf.image.resize(image, (299, 299)) + 1) / 2
            feature_vectors.append(self.model(image))
            if step % 1000 == 0 or step == (len(images)-1):
                feature_vectors = tf.concat(feature_vectors, axis=0).numpy()
                mu = np.sum(feature_vectors, axis=0) + mu 
                sigma = feature_vectors.transpose().dot(feature_vectors) + sigma
                feature_vectors = []
        mu /= N
        sigma /= (N - 1) 
        sigma -= np.reshape(mu, [-1,1]).dot(np.reshape(mu, [1,-1]))
        
        trace1 = np.trace(sigma)
        trace2 = np.trace(self.sigma)
        trace12 = np.trace(linalg.sqrtm(self.sigma.dot(sigma))).real
        return np.sum(np.square(mu - self.mu)) + trace1 + trace2 - 2*trace12
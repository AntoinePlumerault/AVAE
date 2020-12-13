import tensorflow as tf



def generator_adversarial_loss_bigan(s_e, s_g):
    s_g_x, s_g_z, s_g_xz = s_g
    s_e_x, s_e_z, s_e_xz = s_e
    return (
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(s_e_xz), s_e_xz) + 
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(s_g_xz), s_g_xz))

def discriminator_adversarial_loss_bigan(s_e, s_g):
    s_g_x, s_g_z, s_g_xz = s_g
    s_e_x, s_e_z, s_e_xz = s_e
    return (
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(s_g_xz), s_g_xz) + 
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(s_e_xz), s_e_xz))

def generator_adversarial_loss(d_fake):
    return - d_fake

def discriminator_adversarial_loss(d_real, d_fake):
    return (
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.ones_like(d_real), d_real) + 
        tf.nn.sigmoid_cross_entropy_with_logits(
            tf.zeros_like(d_fake), d_fake))

def kl_divergence(z_mu, z_log_sigma):
    return tf.reduce_sum(
        (z_mu**2 + tf.exp(2*z_log_sigma))/2 - 1/2 - z_log_sigma, 
        axis=1)

def negative_log_likelyhood(x, x_mu, sigma=1):
    return tf.reduce_sum(
        1/2 * tf.math.square((x - x_mu) / sigma) ,
        axis=[1,2,3])

def latent_loss(z, z_mu, z_log_sigma):
    w = tf.stop_gradient(tf.nn.softmax(-2*z_log_sigma))
    return tf.reduce_sum(1/2 * tf.math.square(z - z_mu) * w)
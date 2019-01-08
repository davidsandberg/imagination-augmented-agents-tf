import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from tensorflow.python.data import Dataset


def conv_stack(X, k1, c1, k2, c2, k3, c3):
    """Implements the conv_stack module as described in figure 6 in the paper
    """
    conv1 = tf.contrib.layers.conv2d(X, num_outputs=c1, kernel_size=k1,
            stride=1, padding='same', activation_fn=None)
    conv1_relu = tf.nn.relu(conv1)
    conv2 = tf.contrib.layers.conv2d(conv1_relu, num_outputs=c2, kernel_size=k2,
            stride=1, padding='same', activation_fn=None)
    conv2_relu = tf.nn.relu(conv1_relu + conv2)
    conv3 = tf.contrib.layers.conv2d(conv2_relu, num_outputs=c3, kernel_size=k3,
            stride=1, padding='same', activation_fn=None)
    return conv3

def res_conv(X, use_extra_convolution=True):
    """Implements the res_conv module as described in figure 7 in the paper
    """
    if use_extra_convolution:
        c = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=1,
                stride=1, padding='same', activation_fn=None)  #  shape=(?, 10, 10, 64)
    else:
        c = X
    rc1_relu = tf.contrib.layers.conv2d(c, num_outputs=32, kernel_size=3,
            stride=1, padding='same', activation_fn=tf.nn.relu)
    rc2_relu = tf.contrib.layers.conv2d(rc1_relu, num_outputs=32, kernel_size=5,
            stride=1, padding='same', activation_fn=tf.nn.relu)
    rc3 = tf.contrib.layers.conv2d(rc2_relu, num_outputs=64, kernel_size=3,
            stride=1, padding='same', activation_fn=None)
    rc = c + rc3
    return rc

def pool_inject(X):
    """Implements the pool & inject module as described in figure 8 in the paper
    """
    width, height = X.get_shape()[1:3]
    m = tf.layers.max_pooling2d(X, pool_size=(width, height), strides=(1, 1))
    tiled = tf.tile(m, (1, width, height, 1))
    pi = tf.concat([tiled, X], axis=-1)
    return pi

def state_transition_module(a, s, z):
    """Implements the state transition function g(s,z,a) as described in figure 9 in the paper
    An action, a state and a latent variable at time t-1 is transitioned to 
    a state at time t.
    """
    with tf.variable_scope('state_transition_module'):
        if z is None:
            c = tf.concat([a, s], axis=-1)
        else:
            c = tf.concat([a, s, z], axis=-1)
        rc1_relu = tf.nn.relu(res_conv(c))  #  shape=(?, 10, 10, 64)
        pi = pool_inject(rc1_relu)  #  shape=(?, 10, 10, 64)
        s_next = res_conv(pi)
    return s_next

def observation_encoder(o):
    """
    """
    with tf.variable_scope('observation_encoder_module'):
        std1 = tf.nn.space_to_depth(o, 4)
        cs1 = conv_stack(std1, 3, 16, 5, 16, 3, 64)
        std2 = tf.nn.space_to_depth(cs1, 2)
        cs2 = conv_stack(std2, 3, 32, 5, 32, 3, 64)
        e = tf.nn.relu(cs2)
    return e

def prior_module(s, a):
    """
    """
    with tf.variable_scope('prior_module'):
        c = tf.concat([s, a], axis=-1)
        mu = conv_stack(c, 1, 32, 3, 32, 3, 64)
        sigma = tf.nn.softplus(conv_stack(c, 1, 32, 3, 32, 3, 64))
    return mu, sigma

def posterior_module(mu, sigma, s, e, a):
    """
    """
    with tf.variable_scope('posterior_module'):
        c = tf.concat([mu, sigma, s, e, a], axis=-1)
        mu_hat = conv_stack(c, 1, 32, 3, 32, 3, 64)
        sigma_hat = tf.nn.softplus(conv_stack(c, 1, 32, 3, 32, 3, 64))
    return mu_hat, sigma_hat

def initial_state_module(ex):
    """Computes the initial state from the feature maps of a
    number of consequtive initial observations. 
    These feature maps are given in the batch dimension, meaning
    that the first dimension should be of size batch_size x nrof_observations.
    """
    with tf.variable_scope('initial_state_module'):
        e = tf.unstack(ex, axis=1)
        c = tf.concat(e, axis=-1)
        s = conv_stack(c, 1, 64, 3, 64, 3, 64)
    return s

def observation_decoder(s, z):
    with tf.variable_scope('observation_decoder_module'):
        if z is None:
            c = s
        else:
            c = tf.concat([s, z], axis=-1)
        cs1 = conv_stack(c, 1, 32, 5, 32, 3, 64)
        dts1 = tf.nn.depth_to_space(cs1, 2)
        cs2 = conv_stack(dts1, 3, 64, 3, 64, 1, 48)
        dts2 = tf.nn.depth_to_space(cs2, 4)
    return dts2

# def kl_divergence_gaussians(mu0, log_sigma0, mu1, log_sigma1):
#     '''Implements KL divergence between two Gaussians with diagonal covariance matrices based on
#     https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
#     The prior evolves over time and will hence not be restricted to N(0,1)
#     '''
#     sigma0 = tf.exp(log_sigma0)
#     sigma1 = tf.exp(log_sigma1)
#     k = 0.0
#     tf1 = tf.reduce_sum(1.0/sigma1 * sigma0, axis=[2,3,4])
#     tf2 = tf.reduce_sum((mu1-mu0)*1.0/sigma1*(mu1-mu0), axis=[2,3,4])
#     tf3 = tf.cast(tf.log(tf.reduce_prod(sigma1/sigma0, axis=[2,3,4])), tf.float32)
#     Dkl = 0.5 * (tf1 + tf2 - k + tf3)
#     return Dkl
  
# def kl_divergence_gaussians(p_mu, p_sigma_log, q_mu, q_sigma_log):
#     zz = tf.distributions.kl_divergence(
#     tf.distributions.Normal(loc=q_mu, scale=tf.exp(q_sigma_log)+1e-4),
#     tf.distributions.Normal(loc=p_mu, scale=tf.exp(p_sigma_log)+1e-4))
#     return tf.reduce_sum(zz, axis=[2,3,4]), zz
  
def kl_divergence_gaussians(p_mu, p_sigma, q_mu, q_sigma):
    eps = 1e-1
    zz = tf.distributions.kl_divergence(
    tf.distributions.Normal(loc=q_mu, scale=q_sigma+eps),
    tf.distributions.Normal(loc=p_mu, scale=p_sigma+eps))
    return tf.reduce_sum(zz, axis=[2,3,4]), zz
  
def kl_divergence_gaussians2(p_mu, p_sigma_log, q_mu, q_sigma_log):
    #https://hk.saowen.com/a/7404b78cd5b980e16e08423192e35ec18ecb3cb243d310a9d9a194747d9ee1ba
    eps = 1e-3
    r = q_mu - p_mu
    p_sigma, q_sigma = tf.exp(p_sigma_log), tf.exp(q_sigma_log)
    #zz = tf.log(p_sigma) - tf.log(q_sigma) - .5 * (1. - (q_sigma**2 + r**2) / (p_sigma**2+eps))
    zz = p_sigma_log - q_sigma_log - .5 * (1. - (q_sigma**2 + r**2) / (p_sigma**2+eps))
    return tf.reduce_sum(zz, axis=[2,3,4]), zz

def cross_entropy(labels, logits):
    #xent = -tf.reduce_sum(labels * tf.log(logits), axis=[2,3,4])
    xent = -tf.reduce_sum(labels * logits, axis=[2,3,4])
    return xent
  
def get_onehot_actions(actions, nrof_actions, state_shape):
    length = actions.get_shape()[1]
    _, height, width, _ = state_shape
    qq = tf.one_hot(tf.reshape(actions, [-1, length, 1, 1]), nrof_actions, axis=-1)
    onehot_actions = tf.tile(qq, multiples=(1, 1, height, width, 1))
    return onehot_actions

def format_mmm(x):
    min_str, max_str, mean_str = [], [], []
    for i in range(x.shape[1]):
        min_str += [ '%.3g' % np.min(x[:,i,:,:,:]) ]
        max_str += [ '%.3g' % np.max(x[:,i,:,:,:]) ]
        mean_str += [ '%.3g' % np.mean(x[:,i,:,:,:]) ]
    return ', '.join(min_str), ', '.join(max_str), ', '.join(mean_str)
  
class EnvModel():
    
    def __init__(self, obs_shape, nrof_actions=None, nrof_time_steps=None):
        length, width, height, depth = obs_shape
        nrof_init_time_steps = 3
    
        self.obs = tf.placeholder(tf.float32, [None, length, width, height, depth])
        self.actions = tf.placeholder(tf.int32, [None, length])
        
        # Encode observations
        obs_reshaped = tf.reshape(self.obs, [-1, width, height, depth])
        self.encoded_obs_reshaped = observation_encoder(obs_reshaped)
        
        shape = [-1,length]+self.encoded_obs_reshaped.get_shape().as_list()[1:]
        self.encoded_obs = tf.reshape(self.encoded_obs_reshaped, shape)
        
        # Initialize state
        self.encoded_obs_init = self.encoded_obs[:,:nrof_init_time_steps,:,:,:]
        self.initial_state = initial_state_module(self.encoded_obs_init)
        state = self.initial_state
        
        # Convert actions to one-hot
        onehot_actions = get_onehot_actions(self.actions, nrof_actions, state.get_shape().as_list())

        obs_hat_list = []
        next_state_list = []
        mu_list = []
        sigma_list = []
        mu_hat_list = []
        sigma_hat_list = []
        z_list = []
        for t in range(nrof_time_steps):
          
            if t > 0:
                # Variables are reused for time step 1 and onwards
                tf.get_variable_scope().reuse_variables()
          
            # Compute prior statistics
            mu, sigma = prior_module(state, onehot_actions[:,t,:,:,:])
            mu_list += [ mu ]
            sigma_list += [ sigma ]
            
            # Compute posterior statistics
            mu_hat, sigma_hat = posterior_module(mu, sigma, state, self.encoded_obs[:,t,:,:,:], onehot_actions[:,t,:,:,:])
            mu_hat_list += [ mu_hat ]
            sigma_hat_list += [ sigma_hat ]
            
            # Sample from z using the reparametrization trick
            eps = tf.random_normal(tf.shape(sigma_hat), 0.0, 1.0, dtype=tf.float32)
            z = mu_hat + tf.multiply(sigma_hat, eps)
            z_list += [ z ]
            
            # Calculate next state
            next_state = state_transition_module(onehot_actions[:,t,:,:,:], state, z)
            next_state_list += [ next_state ]
            
            # Calculate observation
            obs_hat = observation_decoder(next_state, z)
            obs_hat_list += [ obs_hat ]
            
            
            state = next_state
            
        # Stack lists of tensors
        self.mu = tf.stack(mu_list, axis=1)
        self.sigma = tf.stack(sigma_list, axis=1)
        self.mu_hat = tf.stack(mu_hat_list, axis=1)
        self.sigma_hat = tf.stack(sigma_hat_list, axis=1)
        self.z = tf.stack(z_list, axis=1)
        self.next_state = tf.stack(next_state_list, axis=1)
        self.obs_hat = tf.stack(obs_hat_list, axis=1)

        # Calculate loss
        self.regularization_loss, self.zz = kl_divergence_gaussians(self.mu, self.sigma, self.mu_hat, self.sigma_hat)
        self.reconstruction_loss = cross_entropy(self.obs[:,nrof_init_time_steps:,:,:,:], self.obs_hat)  # Should be KL divergence according to paper
        
def create_dataset(filelist, path, nrof_epochs=1, buffer_size=25, batch_size=10):
    def gen(filelist, path):
        for fn in filelist:
            data = np.float32(load_pickle(os.path.join(path, fn)))
            for i in range(data.shape[0]):
                yield data[i,:13,:,:,:]
          
    ds = Dataset.from_generator(lambda: gen(filelist, path), tf.float32, tf.TensorShape([13, 80, 80, 3]))
    ds = ds.repeat(nrof_epochs)
    ds = ds.prefetch(buffer_size)
    ds = ds.batch(batch_size)
    return ds    
    
def load_pickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr

if __name__ == '__main__':

    with tf.Session() as sess:
      
        filelist = [ 'bouncing_balls_training_data_%03d.pkl' % i for i in range(20) ]
        ds = create_dataset(filelist, 'data', nrof_epochs=1, buffer_size=25, batch_size=10)
      
        with tf.variable_scope('env_model'):
            env_model = EnvModel((13, 80, 80, 3), 1, 10)

        reg_loss = tf.reduce_sum(env_model.regularization_loss)
        rec_loss = tf.reduce_sum(env_model.reconstruction_loss)
        loss = reg_loss + rec_loss
        train_op = tf.train.AdamOptimizer().minimize(loss)

        sess.run(tf.global_variables_initializer())

        train = load_pickle('data/bouncing_balls_testing_data.pkl')
        train = np.expand_dims(train, 4)
        train = np.repeat(train, 3, axis=4)
        obs = train[0:10,:13,:,:,:]
        actions = np.zeros((10,13), dtype=np.float32)
        m = env_model
        
        print('Training')
        feed_dict = {env_model.obs:obs, env_model.actions:actions}
        for batch in range(20):
            #_, obs_hat_, next_state_, rec_loss_, reg_loss_, loss_ = sess.run([train_op, env_model.obs_hat, env_model.next_state, rec_loss, reg_loss, loss], feed_dict=feed_dict)
            obs_hat_, initial_state_, next_state_, reg_loss_, rec_loss_, eoi_, mu_, sigma_, mu_hat_, sigma_hat_, zz_, encoded_obs_, z_, _ = sess.run(
              [m.obs_hat, m.initial_state, m.next_state, m.regularization_loss, m.reconstruction_loss, m.encoded_obs_init, m.mu, m.sigma, m.mu_hat, m.sigma_hat, m.zz, m.encoded_obs, m.z, train_op], feed_dict=feed_dict)
            
            print('obs:              (%s), (%s), (%s)' % format_mmm(obs))
            print('encoded obs:      (%s), (%s), (%s)' % format_mmm(encoded_obs_))
            print('mu:               (%s), (%s), (%s)' % format_mmm(mu_))
            print('sigma:            (%s), (%s), (%s)' % format_mmm(sigma_))
            print('mu_hat:           (%s), (%s), (%s)' % format_mmm(mu_hat_))
            print('sigma_hat:        (%s), (%s), (%s)' % format_mmm(sigma_hat_))
            print('z:                (%s), (%s), (%s)' % format_mmm(z_))
            print('next_state:       (%s), (%s), (%s)' % format_mmm(next_state_))
            print('obs_hat:          (%s), (%s), (%s)' % format_mmm(obs_hat_))
            
            print('Reconstruction loss: ', rec_loss_[0,:])
            print('Regularization loss: ', reg_loss_[0,:])
            
            print('\n\n\n')
        #print('Total loss: ', loss_)
        
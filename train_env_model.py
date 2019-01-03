import tensorflow as tf
import numpy as np
import pickle

def conv_stack(X, k1, c1, k2, c2, k3, c3):
    """Implements the conv_stack module as described in figure 6 in the paper
    """
    conv1 = tf.layers.conv2d(X, c1, kernel_size=k1,
            strides=1, padding='same', activation=None)
    conv1_relu = tf.nn.relu(conv1)
    conv2 = tf.layers.conv2d(conv1_relu, c2, kernel_size=k2,
            strides=1, padding='same', activation=None)
    conv2_relu = tf.nn.relu(conv1_relu + conv2)
    conv3 = tf.layers.conv2d(conv2_relu, c3, kernel_size=k3,
            strides=1, padding='same', activation=None)
    return conv3

def res_conv(X, use_extra_convolution=True):
    """Implements the res_conv module as described in figure 7 in the paper
    """
    if use_extra_convolution:
        c = tf.layers.conv2d(X, 64, kernel_size=1,
                strides=1, padding='same', activation=None)  #  shape=(?, 10, 10, 64)
    else:
        c = X
    rc1_relu = tf.layers.conv2d(c, 32, kernel_size=3,
            strides=1, padding='same', activation=tf.nn.relu)
    rc2_relu = tf.layers.conv2d(rc1_relu, 32, kernel_size=5,
            strides=1, padding='same', activation=tf.nn.relu)
    rc3 = tf.layers.conv2d(rc2_relu, 64, kernel_size=3,
            strides=1, padding='same', activation=None)
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

def initial_state_module(ex, nrof_observations=3):
    """Computes the initial state from the feature maps of a
    number of consequtive initial observations. 
    These feature maps are given in the batch dimension, meaning
    that the first dimension should be of size batch_size x nrof_observations.
    """
    with tf.variable_scope('initial_state_module'):
        shape = [-1,nrof_observations]+ex.get_shape().as_list()[1:]
        e = tf.unstack(tf.reshape(ex, shape), axis=1)
        c = tf.concat(e, axis=-1)
        s = conv_stack(c, 1, 64, 3, 64, 3, 64)
    return s

def observation_decoder(s, z):
    if z is None:
        c = s
    else:
        c = tf.concat([s, z], axis=-1)
    cs1 = conv_stack(c, 1, 32, 5, 32, 3, 64)
    dts1 = tf.nn.depth_to_space(cs1, 2)
    cs2 = conv_stack(dts1, 3, 64, 3, 64, 1, 48)
    dts2 = tf.nn.depth_to_space(cs2, 4)
    return dts2

def kl_divergence_gaussians(mu0, log_sigma0, mu1, log_sigma1):
    '''Implements KL divergence between two Gaussians with diagonal covariance matrices based on
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    The prior evolves over time and will hence not be restricted to N(0,1)
    '''
    sigma0 = tf.exp(log_sigma0)
    sigma1 = tf.exp(log_sigma1)
    k = 0.0
    tf1 = tf.reduce_sum(1.0/sigma1 * sigma0)
    tf2 = tf.reduce_sum((mu1-mu0)*1.0/sigma1*(mu1-mu0))
    tf3 = tf.cast(tf.log(tf.reduce_prod(sigma1/sigma0)), tf.float32)
    Dkl = 0.5 * (tf1 + tf2 - k + tf3)
    return Dkl

def cross_entropy(labels, logits):
    xent = -tf.reduce_sum(labels * tf.log(logits), [0, 1, 2, 3])
    return xent

class EnvModel():
    
    def __init__(self, obs_shape, num_actions):
        width, height, depth = obs_shape
    
        self.obs_init = tf.placeholder(tf.float32, [None, width, height, depth])  # TODO: Fix dimensions with batches
        self.obs = tf.placeholder(tf.float32, [None, width, height, depth])
        self.actions = tf.placeholder(tf.int32, [None,])
        
        assert_op = tf.Assert(tf.equal(tf.shape(self.obs_init)[0], tf.shape(self.obs)[0]*3), [self.obs_init, self.obs])
        with tf.control_dependencies([assert_op]):
            # Make sure that parameters for the observation encoder are shared
            with tf.variable_scope('observation_encoder', reuse=False):
                encoded_obs = observation_encoder(self.obs)
            with tf.variable_scope('observation_encoder', reuse=True):
                encoded_init_obs = observation_encoder(self.obs_init)
            
        # Initialize state
        self.state = initial_state_module(encoded_init_obs)
        
        # Convert actions to one-hot
        qq = tf.one_hot(tf.reshape(self.actions, [-1, 1, 1]), num_actions, axis=-1)
        onehot_actions = tf.tile(qq, multiples=(1, self.state.get_shape()[1], self.state.get_shape()[2], 1))
        
        # Compute statistics for prior and posterior
        mu, sigma = prior_module(self.state, onehot_actions)
        mu_hat, sigma_hat = posterior_module(mu, sigma, self.state, encoded_obs, onehot_actions)
        
        # Sample from z using the reparametrization trick
        eps = tf.random_normal(tf.shape(sigma_hat), 0.0, 1.0, dtype=tf.float32)
        z = mu_hat + tf.multiply(tf.exp(sigma_hat), eps)  # Check how standard deviation is represented in the paper
        
        self.next_state = state_transition_module(onehot_actions, self.state, z)
        self.obs_hat = observation_decoder(self.next_state, z)
        
        self.regularization_loss = kl_divergence_gaussians(mu, sigma, mu_hat, sigma_hat)
        self.reconstruction_loss = cross_entropy(self.obs, self.obs_hat)  # Should be KL divergence according to paper
        self.loss = self.regularization_loss + self.reconstruction_loss
        
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
    
def load_pickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr

if __name__ == '__main__':

    with tf.Session() as sess:
        with tf.variable_scope('env_model'):
            env_model = EnvModel((80, 80, 3), 1)

        sess.run(tf.global_variables_initializer())

        train = load_pickle('bouncing_balls_testing_data.pkl')
        train = np.expand_dims(train, 4)
        train = np.repeat(train, 3, axis=4)
        #print(train.shape)
        obs_init = train[0:5,:3,:,:,:].reshape(15,80,80,3)
        obs = train[0:5,3,:,:,:]
        actions = np.zeros((5,), dtype=np.float32)
        
        feed_dict = {env_model.obs_init:obs_init, env_model.obs:obs, env_model.actions:actions}
        _, obs_hat_, next_state_ = sess.run([env_model.train_op, env_model.obs_hat, env_model.next_state], feed_dict=feed_dict)
        
        print('Next state shape: ', next_state_.shape)
        print('Generated observation shape: ', obs_hat_.shape)
        
import tensorflow as tf
import numpy as np
import pickle
import os
import time
import datetime
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

def kl_div_gauss(action_dist1, action_dist2, action_size):
    # https://github.com/openai/baselines/blob/f2729693253c0ef4d4086231d36e0a4307ec1cb3/baselines/acktr/utils.py
    mean1, std1 = action_dist1[:, :action_size], action_dist1[:, action_size:]
    mean2, std2 = action_dist2[:, :action_size], action_dist2[:, action_size:]

    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)
  
def kl_divergence_gaussians(p_mu, p_sigma, q_mu, q_sigma):
    eps = 1e-1
    zz = tf.distributions.kl_divergence(
    tf.distributions.Normal(loc=q_mu, scale=q_sigma+eps),
    tf.distributions.Normal(loc=p_mu, scale=p_sigma+eps))
    return tf.reduce_sum(zz, axis=[2,3,4])
  
def kl_divergence_gaussians2(p_mu, p_sigma_log, q_mu, q_sigma_log):
    #https://hk.saowen.com/a/7404b78cd5b980e16e08423192e35ec18ecb3cb243d310a9d9a194747d9ee1ba
    eps = 1e-3
    r = q_mu - p_mu
    p_sigma, q_sigma = tf.exp(p_sigma_log), tf.exp(q_sigma_log)
    #zz = tf.log(p_sigma) - tf.log(q_sigma) - .5 * (1. - (q_sigma**2 + r**2) / (p_sigma**2+eps))
    zz = p_sigma_log - q_sigma_log - .5 * (1. - (q_sigma**2 + r**2) / (p_sigma**2+eps))
    return tf.reduce_sum(zz, axis=[2,3,4])

def kl_div_bernoulli(p, q):
    eps = 1e-6
    #kl = tf.reduce_sum(p*tf.log((p+eps)/(q+eps)) + (1-p)*tf.log((1-p+eps)/(1-q+eps)+eps), axis=[2,3,4])
    pc = tf.clip_by_value(p, eps, 1-eps)
    qc = tf.clip_by_value(q, eps, 1-eps)
    kl = tf.reduce_sum(pc*tf.log(pc/qc) + (1-pc)*tf.log((1-pc)/(1-qc)), axis=[2,3,4])
    return kl
  
def get_onehot_actions(actions, nrof_actions, state_shape):
    length = actions.get_shape()[1]
    _, height, width, _ = state_shape
    qq = tf.one_hot(tf.reshape(actions, [-1, length, 1, 1]), nrof_actions, axis=-1)
    onehot_actions = tf.tile(qq, multiples=(1, 1, height, width, 1))
    return onehot_actions
  
def softmax(x, axis):
    z = x - tf.reduce_max(x, axis, keepdims=True)
    y = tf.exp(z) / tf.reduce_sum(tf.exp(z), axis, keepdims=True)
    return y
  
class EnvModel():
    
    def __init__(self, is_pdt, obs, actions, nrof_actions=None, nrof_time_steps=None):
        _, length, width, height, depth = obs.get_shape().as_list()
        nrof_init_time_steps = 3
    
        self.obs = obs
        self.actions = actions
        
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
            eps = tf.random_normal(tf.shape(sigma), 0.0, 1.0, dtype=tf.float32)
            mu_x = tf.where(is_pdt[:,t], mu, mu_hat)
            sigma_x = tf.where(is_pdt[:,t], sigma, sigma_hat)
            z = mu_x + tf.multiply(sigma_x, eps)
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
        self.obs_hat = tf.nn.sigmoid(tf.stack(obs_hat_list, axis=1))

        # Calculate loss
        lmbd = 0.05  # nats per dimension
        f = lmbd * np.prod(self.mu.get_shape().as_list()[2:])
        print('Reg loss limit: %.3f' % f)
        self.regularization_loss = tf.maximum(tf.constant(f, tf.float32), kl_divergence_gaussians(self.mu, self.sigma, self.mu_hat, self.sigma_hat))
        self.reconstruction_loss = kl_div_bernoulli(self.obs[:,nrof_init_time_steps:,:,:,:], self.obs_hat)
        
def create_dataset(filelist, path, buffer_size=25, batch_size=10):
    def gen(filelist, path):
        for fn in filelist:
            data = np.float32(load_pickle(os.path.join(path, fn)))
            data = np.expand_dims(data, 4)
            data = np.repeat(data, 3, axis=4)
            for i in range(data.shape[0]):
                yield data[i,:13,:,:,:], np.zeros((13,), dtype=np.int32)
          
    ds = Dataset.from_generator(lambda: gen(filelist, path), (tf.float32, tf.int32), (tf.TensorShape([13, 80, 80, 3]), tf.TensorShape([13,])))
    ds = ds.repeat(count=None)
    ds = ds.prefetch(buffer_size)
    ds = ds.batch(batch_size)
    return ds
    
def load_pickle(filename):
    with open(filename, 'rb') as f:
        arr = pickle.load(f)
    return arr
  
def save_pickle(filename, var_list):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

def gettime():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')

if __name__ == '__main__':
  
    res_name = gettime()
    res_dir = os.path.join('/home/david/git/imagination-augmented-agents-tf/results/', res_name) 
    os.makedirs(res_dir, exist_ok=True)
    log_filename = os.path.join(res_dir, 'log.pkl')
    model_filename = os.path.join(res_dir, res_name)

    with tf.Session() as sess:
      
        batch_size = 16
        length = 10
        max_nrof_steps = 10000
      
        filelist = [ 'bouncing_balls_training_data_%03d.pkl' % i for i in range(20) ]
        dataset = create_dataset(filelist, 'data', buffer_size=20000, batch_size=batch_size)

        # Create an iterator over the dataset
        iterator = dataset.make_one_shot_iterator()
        obs, action = iterator.get_next()
        
        
        is_pdt_ph = tf.placeholder(tf.bool, [None, length])
        is_pdt = np.ones((batch_size, length), np.bool)
        is_pdt[:,0::4] = False
      
        with tf.variable_scope('env_model'):
            env_model = EnvModel(is_pdt_ph, obs, action, 1, length)

        reg_loss = tf.reduce_mean(env_model.regularization_loss)
        rec_loss = tf.reduce_mean(env_model.reconstruction_loss)
        loss = reg_loss + rec_loss

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        
        loss_log = np.zeros((max_nrof_steps,), np.float32)
        rec_loss_log = np.zeros((max_nrof_steps,), np.float32)
        reg_loss_log = np.zeros((max_nrof_steps,), np.float32)

        try:
            print('Started training')
            rec_loss_tot, reg_loss_tot, loss_tot = (0.0, 0.0, 0.0)
            t = time.time()
            for i in range(max_nrof_steps+1):
                _, rec_loss_, reg_loss_, loss_ = sess.run([train_op, rec_loss, reg_loss, loss], feed_dict={is_pdt_ph: is_pdt})
                loss_log[i], rec_loss_log[i], reg_loss_log[i] = loss_, rec_loss_, reg_loss_
                rec_loss_tot += rec_loss_
                reg_loss_tot += reg_loss_
                loss_tot += loss_
                if i % 10 == 0:
                    print('step: %-5d  rec_loss: %-12.1f reg_loss: %-12.1f loss: %-12.1f' % (i, rec_loss_tot/10, reg_loss_tot/10, loss_tot/10))
                    rec_loss_tot, reg_loss_tot, loss_tot = (0.0, 0.0, 0.0) 
                    t = time.time()
                if i % 5000 == 0 and i>0:
                    saver.save(sess, model_filename, i)
                if i % 100 == 0 and i>0:
                    save_pickle(log_filename, [loss_log, rec_loss_log, reg_loss_log])

                
        except tf.errors.OutOfRangeError:
            pass
          
        print("Saving model...")
        saver.save(sess, model_filename, i)

        print('Done!')
        
        
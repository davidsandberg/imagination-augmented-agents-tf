import os
import gym
import time
import logging
import numpy as np
import tensorflow as tf
from common.minipacman import MiniPacman
from common.multiprocessing_env import SubprocVecEnv
import tensorflow.contrib.slim as slim
from tqdm import tqdm

from env_model import create_env_model
from a2c import get_actor_critic, train, CnnPolicy
from pacman_util import num_pixels, mode_rewards, pix_to_target, rewards_to_target, mode_rewards, target_to_pix


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p



class ImaginationCore(object):
    def __init__(self, num_rollouts, num_actions, num_rewards,
            ob_space, actor_critic, env_model):

        self.num_rollouts = num_rollouts
        self.num_actions  = num_actions
        self.num_rewards  = num_rewards
        self.ob_space     = ob_space
        self.actor_critic = actor_critic
        self.env_model    = env_model


    def imagine(self, state, sess):
        nw, nh, nc = self.ob_space

        batch_size = state.shape[0]

        state = np.tile(state, [self.num_actions, 1, 1, 1, 1])
        state = state.reshape(-1, nw, nh, nc)

        action = np.array([[[i] for i in range(self.num_actions)] for j in
            range(batch_size)])

        action = action.reshape((-1,))

        rollout_batch_size = batch_size * self.num_actions

        rollout_states = []
        rollout_rewards = []

        for step in range(self.num_rollouts):
            onehot_action = np.zeros((rollout_batch_size, self.num_actions, nw, nh))
            onehot_action[range(rollout_batch_size), action] = 1
            onehot_action = np.transpose(onehot_action, (0, 2, 3, 1))

            imagined_state, imagined_reward = sess.run(
                    [self.env_model.imag_state, self.env_model.imag_reward],
                    feed_dict={
                        self.env_model.input_states: state,
                        self.env_model.input_actions: onehot_action,
                })

            imagined_state = np.argmax(softmax(imagined_state, axis=1), axis=1)
            imagined_reward = np.argmax(softmax(imagined_reward, axis=1), axis=1)

            imagined_state = target_to_pix(imagined_state)
            imagined_state = imagined_state.reshape((rollout_batch_size, nw, nh,
                nc))

            onehot_reward = np.zeros((rollout_batch_size, self.num_rewards))
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state)
            rollout_rewards.append(onehot_reward)

            state = imagined_state
            action, _, _ = self.actor_critic.act(state)

        return np.array(rollout_states), np.array(rollout_rewards)


g_actor_critic = None
def get_cache_loaded_a2c(sess, nenvs, nsteps, ob_space, ac_space):
    global g_actor_critic
    if g_actor_critic is None:
        with tf.variable_scope('actor'):
            g_actor_critic = get_actor_critic(sess, nenvs, nsteps, ob_space,
                    ac_space, CnnPolicy, should_summary=False)
        g_actor_critic.load('weights/model_100000.ckpt')
        print('Actor restored!')
    return g_actor_critic


g_env_model = None
def get_cache_loaded_env_model(sess, nenvs, ob_space, num_actions):
    global g_env_model
    if g_env_model is None:
        with tf.variable_scope('env_model'):
            g_env_model = create_env_model(ob_space, num_actions, num_pixels,
                    len(mode_rewards['regular']))

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        loader = tf.train.Saver(var_list=save_vars)
        loader.restore(sess, 'weights/env_model.ckpt')

        print('Env model restored!')

    return g_env_model



class I2aPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        hidden_size = 256
        nsteps = 5

        num_rewards = len(mode_rewards['regular'])
        num_actions = ac_space.n
        width, height, depth = ob_space

        actor_critic = get_cache_loaded_a2c(sess, nbatch, nsteps, ob_space, ac_space)
        env_model = get_cache_loaded_env_model(sess, nbatch, ob_space, num_actions)

        num_rollouts = 1

        self.imagination = ImaginationCore(num_rollouts, num_actions, num_rewards,
                ob_space, actor_critic, env_model)

        with tf.variable_scope('model', reuse=reuse):
            # Model based path.
            self.imagined_state = tf.placeholder(tf.float32, [None, None, width, height, depth])
            self.imagined_reward = tf.placeholder(tf.float32, [None, None, num_rewards])

            num_steps = tf.shape(self.imagined_state)[0]
            batch_size = tf.shape(self.imagined_state)[1]

            hidden_state = self.get_encoder(self.imagined_state, self.imagined_reward,
                    num_steps, batch_size, width, height, depth, hidden_size)

            # Model free path.
            self.state = tf.placeholder(tf.float32, [None, width, height,
                depth])

            state_batch_size = tf.shape(self.state)[0]

            c1 = slim.conv2d(self.state, 16, kernel_size=3,
                    stride=1, padding='valid', activation_fn=tf.nn.relu)
            c2 = slim.conv2d(c1, 16, kernel_size=3,
                    stride=2, padding='valid', activation_fn=tf.nn.relu)

            features = tf.reshape(c2, [state_batch_size, 6 * 8 * 16])

            self.features = features

            hidden_state = tf.reshape(hidden_state, [state_batch_size, 80 * 256
                // 16])

            x = tf.concat([features, hidden_state], axis=1)
            x = slim.fully_connected(x, 256, activation_fn=tf.nn.relu)

            self.pi = slim.fully_connected(x, num_actions)
            self.vf = slim.fully_connected(x, 1)[:, 0]

        # Sample action. `pi` is like the logits
        u = tf.random_uniform(tf.shape(self.pi))
        self.a0 = tf.argmax(self.pi - tf.log(-tf.log(u)), axis=-1)

        # Get the negative log likelihood
        one_hot_actions = tf.one_hot(self.a0, self.pi.get_shape().as_list()[-1])
        self.neglogp0 = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.pi,
            labels=one_hot_actions)


    def get_encoder(self, state, reward, num_steps, batch_size, width, height, depth, hidden_size):
        state = tf.reshape(state, [num_steps * batch_size, width, height,
            depth])

        c1 = slim.conv2d(state, 16, kernel_size=3, stride=1,
                padding='valid', activation_fn=tf.nn.relu)
        features = slim.conv2d(c1, 16, kernel_size=3, stride=2,
                padding='valid', activation_fn=tf.nn.relu)

        features = tf.reshape(features, [num_steps, batch_size, 6 * 8 * 16])

        rnn_input = tf.concat([features, reward], 2)

        cell = tf.contrib.rnn.GRUCell(hidden_size)
        _, internal_state = tf.nn.dynamic_rnn(cell, rnn_input, time_major=True, dtype=tf.float32)

        return internal_state


    def step(self, sess, ob):
        imagined_state, imagined_reward, ob = self.transform_input(ob, sess)

        a, v, neglogp = sess.run([
                self.a0,
                self.vf,
                self.neglogp0
            ],
            {
                self.imagined_state: imagined_state,
                self.imagined_reward: imagined_reward,
                self.state: ob
        })
        return a, v, neglogp


    def value(self, sess, ob):
        imagined_state, imagined_reward, ob = self.transform_input(ob, sess)

        v = sess.run(self.vf, {
            self.imagined_state: imagined_state,
            self.imagined_reward: imagined_reward,
            self.state: ob
        })
        return v

    def get_inputs(self):
        return [self.imagined_state, self.imagined_reward, self.state]

    def transform_input(self, X, sess):
        imagined_state, imagined_reward = self.imagination.imagine(X, sess)
        return [imagined_state, imagined_reward, X]



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    train(I2aPolicy, 'i2a', summarize=False)



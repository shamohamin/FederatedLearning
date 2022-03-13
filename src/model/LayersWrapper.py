from tensorflow import keras
import tensorflow as tf
from functools import partial
import numpy as np

ReluDenseLayer = partial(
    keras.layers.Dense, kernel_initializer="he_normal",
    activation="relu", kernel_regularizer=keras.regularizers.l2())

ReluConvLayer = partial(
    keras.layers.Conv2D, activation="relu", kernel_initializer="he_normal",
    kernel_regularizer=keras.regularizers.l2())


def load_default_model(input_shape: list, output_units: int):
    inputs = keras.layers.Input(shape=(*input_shape,))

    layers = [
        ReluConvLayer(filters=64, strides=4, kernel_size=8),
        ReluConvLayer(filters=32, strides=2, kernel_size=4),
        keras.layers.BatchNormalization(),
        ReluConvLayer(filters=16, strides=1, kernel_size=2),
        keras.layers.Flatten(),
        ReluDenseLayer(units=512),
        ReluDenseLayer(units=256),
        ReluDenseLayer(units=output_units, activation="softmax")
    ]
    
    inp = layers[0](inputs)
    
    for layer in layers[1:]:
        inp = layer(inp)
    
    model = keras.models.Model(inputs=inputs, outputs=inp)
    model.summary()
    
    return model

from keras.layers import Conv2D, Dense, Flatten
class DQNAgent:
    def __init__(self, name, state_shape, n_actions, epsilon=0, reuse=False):
        """A simple DQN agent"""
        # with tf.variable_scope(name, reuse=reuse):
            
        self.network = keras.models.Sequential()
    
            # Keras ignores the first dimension in the input_shape, which is the batch size. 
            # So just use state_shape for the input shape
        self.network.add(Conv2D(32, (8, 8), strides=4, activation='relu',use_bias=False, input_shape=state_shape,kernel_initializer=keras.initializers.variance_scaling(scale=2)))
        self.network.add(Conv2D(64, (4, 4), strides=2, activation='relu',use_bias=False,kernel_initializer=keras.initializers.variance_scaling(scale=2)))
        self.network.add(Conv2D(64, (3, 3), strides=1, activation='relu',use_bias=False,kernel_initializer=keras.initializers.variance_scaling(scale=2)))
        self.network.add(Conv2D(1024, (7, 7), strides=1, activation='relu',use_bias=False,kernel_initializer=keras.initializers.variance_scaling(scale=2)))
        self.network.add(Flatten())
        self.network.add(Dense(n_actions, activation='linear',kernel_initializer=keras.initializers.variance_scaling(scale=2)))
            
            # prepare a graph for agent step
            # self.state_t = tf.placeholder('float32', [None,] + list(state_shape))
            # self.qvalues_t = self.get_symbolic_qvalues(self.state_t)
            
        # self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.epsilon = epsilon

    def get_symbolic_qvalues(self, state_t):
        """takes agent's observation, returns qvalues. Both are tf Tensors"""
        qvalues = self.network(state_t)
        
        
        assert tf.is_numeric_tensor(qvalues) and qvalues.shape.ndims == 2, \
            "please return 2d tf tensor of qvalues [you got %s]" % repr(qvalues)
        assert int(qvalues.shape[1]) == n_actions
        
        return qvalues
    
    def get_qvalues(self, state_t):
        """Same as symbolic step except it operates on numpy arrays"""
        sess = tf.get_default_session()
        return sess.run(self.qvalues_t, {self.state_t: state_t})
    
    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p = [1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

from gym.core import ObservationWrapper
from gym.spaces import Box

# from scipy.misc import imresize
import cv2

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        ObservationWrapper.__init__(self,env)
        
        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        """what happens to each observation"""
        
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = img[34:-16, :, :]
        
        # resize image
        img = cv2.resize(img, self.img_size)
        
        img = img.mean(-1,keepdims=True)
        
        img = img.astype('float32') / 255.
              
        return img
    
from gym.spaces.box import Box
from gym.core import Wrapper
import gym

class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4, dim_order='tensorflow'):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(FrameBuffer, self).__init__(env)
        self.dim_order = dim_order
        if dim_order == 'tensorflow':
            height, width, n_channels = env.observation_space.shape
            """Multiply channels dimension by number of frames"""
            obs_shape = [height, width, n_channels * n_frames] 
        else:
            raise ValueError('dim_order should be "tensorflow" or "pytorch", got {}'.format(dim_order))
        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')
        
    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer
    
    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, done, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, reward, done, info
    
    def update_buffer(self, img):
        if self.dim_order == 'tensorflow':
            offset = self.env.observation_space.shape[-1]
            axis = -1
            cropped_framebuffer = self.framebuffer[:,:,:-offset]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis = axis)
        

def make_env():
    # env = gym.make("BreakoutDeterministic-v4")
    env = gym.make("BreakoutDeterministic-v4")
    env = PreprocessAtari(env)
    env = FrameBuffer(env, n_frames=4, dim_order='tensorflow')
    return env

def load_model():
    inputs = keras.layers.Input(shape=(84, 84, 4,))

    # Convolutions on the frames on the screen
    layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = keras.layers.Flatten()(layer3)

    layer5 = keras.layers.Dense(512, activation="relu")(layer4)
    action = keras.layers.Dense(4, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

from collections import deque
import random
import cv2
import gym
import numpy as np
from ..config import BATCH_SIZE, DISCOUNT_FACTOR, ENV_NAME, EPSILON, EPISODES, LRATE, MINIMUM_EXPERIENCE_MEMORY
from .LayersWrapper import load_default_model
from tensorflow import keras


class Agent:
    def __init__(self, env, episodes, policy):
        self.policy = policy
        self.episodes = episodes
        self.env = env
        self.input_shape = self.env.observation_space.shape  # input frame size
        self.output_shape = self.env.action_space.n  # actions count
    
    def updateParameters(self):
        raise NotImplementedError
    
    def train(self):
        raise NotImplementedError


class DoubleQNModel(Agent):
    def __init__(self, env, episodes, model, policy):
        super().__init__(env, episodes, policy)
        self.targetModel = keras.models.clone_model(model)
        self.workerModel = keras.models.clone_model(model)
    
    def updateParameters(self):
        pass
    
    def train(self):
        pass
        
    
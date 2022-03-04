import gym
import numpy as np
from ..config import ENV_NAME
from .LayersWrapper import load_default_model
from tensorflow import keras
import cv2


class DoubleQNModel:
    def __init__(self):
        self.targetModel = None
        self.workerModel = None
        self.env = gym.make(ENV_NAME)
        self.input_shape  = self.env.observation_space.shape # input frame size
        self.output_shape = self.env.action_space.n # actions count
        print(self.input_shape, self.output_shape)
        
    
    def createModel(self):
        assert self.targetModel is not None
        self.workerModel = keras.models.clone_model(self.targetModel)
        
    
    
class EpsilonGreedyModel(DoubleQNModel):
    def __init__(self):
        super().__init__()
        self.createModel()
    
    def createModel(self):
        self.targetModel = load_default_model(self.input_shape, self.output_shape)
        super().createModel()
        
    
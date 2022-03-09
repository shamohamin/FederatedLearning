from ..config import (
    EPSILON,
    EPSILON_GREEDY_FRAMES,
    MIN_EPSILON
)
import numpy as np
import tensorflow as tf


class Policy:
    def __init__(self, name, *args, **kwargs):
        self.policyName = name
        self.args = args
        self.kwargs = kwargs

    def getAction(self, currState, *args, **kwargs):
        raise NotImplementedError

    def updateDefaultParameters(self, *args, **kwargs):
        raise NotImplementedError


class EpsilonGreedyPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__("epsilon_greedy", *args, **kwargs)
        if len(args) <= 1:
            raise Exception(
                "Please Pass ActionShapes And ObserveFrameCount."
            )
        
        self.actionCount = self.args[0]
        self.observeFramesCount = self.args[1]
        
        self.epsilon_greedy_frames = self.kwargs.get(
            "epsilon_greedy_frames", EPSILON_GREEDY_FRAMES)
        self.epsilon = self.kwargs.get("epsilon", EPSILON)
        self.epsilon_min = self.kwargs.get("epsilon_min", MIN_EPSILON)
        self.epsilon_max = self.kwargs.get("epsilon_max", MIN_EPSILON)

        self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
        )

    def getAction(self, currState, *args, **kwargs):
        if len(args) <= 1:
            raise Exception("please pass model and frame count")
        model = args[0]
        frameCount = args[1]

        if frameCount < self.observeFramesCount or np.random.rand(1)[0] < self.epsilon:
            # make random move to observe the environment
            action = np.random.choice(self.actionCount)
        else:
            # use prediction
            expandedState = np.expand_dims(currState, axis=0)
            tensorState = tf.convert_to_tensor(expandedState)
            preds = model.predict(tensorState, training=False)
            action = tf.argmax(preds).numpy()

        return action

    def updateDefaultParameters(self, *args, **kwargs):
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)

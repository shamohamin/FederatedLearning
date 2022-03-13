import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import bz2
import pickle
import random

from .policy import Policy

from ..config import (
    BATCH_SIZE,
    DISCOUNT_FACTOR,
    EPISODES,
    LRATE,
    MAX_STEPS_PER_EPISODE,
    MINIMUM_EXPERIENCE_MEMORY,
    UPDATE_AFTER_ACTIONS,
    UPDATE_TARGET_NETWOTK
)


class Agent:
    def __init__(self, env, episodes, policy, maxMemorySize, updateModelAfter=None):
        self.policy = policy
        self.episodes = episodes
        self.env = env
        self.input_shape = self.env.observation_space.shape  # input frame size
        self.output_shape = self.env.action_space.n  # actions count
        self.memorySize = maxMemorySize
        self.updateModelAfter = 2 * \
            UPDATE_TARGET_NETWOTK if updateModelAfter is None else updateModelAfter

    def updateParameters(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class DoubleQNModel(Agent):
    def __init__(self, env, model, policy: Policy, episodes=EPISODES,
                 maxMemorySize=MINIMUM_EXPERIENCE_MEMORY,
                 lossFunction=keras.losses.Huber(),
                 optimizer=keras.optimizers.Adam(
                     learning_rate=LRATE, clipnorm=1.0),
                 updateModelAfter=None,
                 senderFunction=None, waitFunction=None,
                 federatedLearning=True, terminateSignal=None,
                 procName=None
                 ):
        super().__init__(env, episodes, policy, maxMemorySize, updateModelAfter)
        self.targetModel = keras.models.clone_model(model)
        self.workerModel = keras.models.clone_model(model)

        self.lossFunction = lossFunction
        self.optimizer = optimizer
        self.startTime = datetime.now()

        self.data = {
            "frame_count": 0,
            "running_reward": 0,
            "episode_count": 0,
            "reward_history": [],
            "history": []
        }

        self.federatedLearning = federatedLearning
        self.senderFunction = senderFunction
        self.terminateSignal = terminateSignal
        self.waitAndSaveModel = waitFunction
        self.procName = procName

    def updateParameters(self):
        cond = self.data["frame_count"] % UPDATE_AFTER_ACTIONS and len(
            self.data["history"]) > BATCH_SIZE

        if cond:
            batch = random.sample(self.data["history"], k=BATCH_SIZE)
            # batch = np.random.choice(np.array(self.data["history"]), size=BATCH_SIZE)

            sampleState = np.array([sample[0] for sample in batch])
            nextSampleState = np.array([sample[2] for sample in batch])
            sampleRewards = np.array([sample[-1] for sample in batch])
            sampleActions = np.array([sample[1] for sample in batch])
            sampleDone = tf.convert_to_tensor(
                [float(sample[-2]) for sample in batch]
            )

            futureRewards = self.targetModel.predict(nextSampleState)
            updatedQValues = sampleRewards + DISCOUNT_FACTOR * tf.reduce_max(
                futureRewards, axis=1
            )

            updatedQValues = updatedQValues * (1 - sampleDone) - sampleDone

            masks = tf.one_hot(sampleActions, self.output_shape)

            with tf.GradientTape() as tape:
                q_values = self.workerModel(sampleState)
                q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = self.lossFunction(updatedQValues, q_actions)

            gradians = tape.gradient(
                loss, self.workerModel.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradians, self.workerModel.trainable_variables))

    def train(self):

        for episode in range(self.episodes):
            state = np.array(self.env.reset())
            episodeReward = 0
            self.data["episode_count"] = episode + 1

            for _ in range(1, MAX_STEPS_PER_EPISODE):
                self.data["frame_count"] = self.data["frame_count"] + 1

                action = self.policy.getAction(
                    state, self.workerModel, self.data["frame_count"])
                self.policy.updateDefaultParameters()

                # perform action
                nextState, reward, done, _ = self.env.step(action)
                nextState = np.array(nextState)

                episodeReward = episodeReward + reward
                # save state in the replyMemory
                self.data["history"].append(
                    (state, action, nextState, done, reward))

                state = nextState

                # train model after 4 frames and update target model
                # and
                # update target model after 100000 frames
                self.updateParameters()

                if self.data["frame_count"] % UPDATE_TARGET_NETWOTK == 0:
                    self.targetModel.set_weights(
                        self.workerModel.get_weights())
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(
                        self.data["running_reward"], episode, self.data["frame_count"]))
                    
                    if self.data["frame_count"] % self.updateModelAfter == 0:
                        self.senderFunction(self.targetModel)
                        self.waitAndSaveModel()
                        
                        self.saveStates()
                    #exit(1)

                if len(self.data["history"]) > self.memorySize:
                    # make room for new experience
                    del self.data["history"][:1]

                if done:
                    break

            self.data["reward_history"].append(episodeReward)
            if len(self.data["reward_history"]) > 100:
                del self.data["reward_history"][:1]

            self.data["running_reward"] = np.mean(
                np.array(self.data["reward_history"]))
            print("episode: {} -> reward: {}".format(episode, episodeReward))

            if self.data["running_reward"] > 30:
                break

    def saveStates(self):
        now = datetime.now()
        currTime = now.strftime("%y-%m-%d-%H-%M-%S")

        self.targetModel.save_weights(f"{currTime}_model.h5")

        with bz2.BZ2File(f"{currTime}_data.pbz2", "w") as file:
            pickle.dump(self.data, file)

    def loadStates(self):
        import glob

        self.workerModel.load_weights(sorted(list(glob.glob("*.h5")))[-1])
        self.targetModel.set_weights(self.targetModel.get_weights())

        filename = sorted(list(glob.glob("*.pbz2")))[-1]
        data = bz2.BZ2File(filename, "rb")
        data = pickle.load(data)
        self.data = data

from ..model.DoubleQNModel import DoubleQNModel
from ..config import EPISODES, HOST, MINIMUM_EXPERIENCE_MEMORY, LRATE, PORT

from multiprocessing import Process, Event
from tensorflow import keras
from io import BytesIO
import socket
import pickle
import requests as re


class RLClinet(Process):
    def __init__(self, procName,
                 env, model, policy,
                 episodes=EPISODES,
                 maxMemorySize=MINIMUM_EXPERIENCE_MEMORY,
                 sendModelAfter=None,
                 optimizer=keras.optimizers.Adam(
                     learning_rate=LRATE, clipnorm=1),
                 lossFunction=keras.losses.Huber(),
                 host=HOST, port=PORT):

        self.terminateSignal = Event()

        self.agent = DoubleQNModel(
            env, model, policy,
            episodes, maxMemorySize,
            optimizer=optimizer, lossFunction=lossFunction,
            updateModelAfter=sendModelAfter, federatedLearning=True,
            senderFunction=self.senderFunction,
            waitFunction=self.recieveAndSave,
            terminateSignal=self.terminateSignal,
            procName=procName
        )

        self.serverHost = host
        self.serverPort = port
        self.sock = None

    def senderFunction(self, model, score):
        print("start sending model.")
        print(score)
        modelBytes = pickle.dumps(
            {
                "proc_name": self.agent.procName,
                "weights": model.get_weights(),
                "score": score
            }
        )
        res = re.post("http://127.0.0.1:5000/get_weights", modelBytes)
        print(res.json())
        print("sending model completed.")

    def recieveAndSave(self):
        print("start getting model")
        exit(1)

    def run(self):

        self.agent.train()

from ..model.DoubleQNModel import DoubleQNModel
from ..config import EPISODES, HOST, MINIMUM_EXPERIENCE_MEMORY, LRATE

from multiprocessing import Process, Event
from tensorflow import keras
import pickle
import requests as re
import time


class RLClinet(Process):
    def __init__(self, procName,
                 env, model, policy,
                 episodes=EPISODES,
                 maxMemorySize=MINIMUM_EXPERIENCE_MEMORY,
                 sendModelAfter=None,
                 optimizer=keras.optimizers.Adam(
                     learning_rate=LRATE, clipnorm=1),
                 lossFunction=keras.losses.Huber(),
                 host=HOST):

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
        self.sock = None

    def senderFunction(self, model, score):
        print("start sending model.")
        modelBytes = pickle.dumps(
            {
                "proc_name": self.agent.procName,
                "weights": model.get_weights(),
                "score": score
            }
        )
        res = re.post(HOST + "get_weights", modelBytes)
        print(res.json())
        print("sending model completed.")

    def recieveAndSave(self):
        print("start getting model")
        while True:
            res = re.get(HOST + "get_glob_model", params={"proc_name": self.agent.procName})
            if res.status_code == 400:
                time.sleep(1)
            elif res.status_code == 200:
                print(len(res.content))
                data = pickle.loads(res.content)
                print("weights successfully received.")
                return data["weights"]                

    def run(self):
        self.agent.train()

from ..model.DoubleQNModel import DoubleQNModel
from ..config import EPISODES, MINIMUM_EXPERIENCE_MEMORY, LRATE

from multiprocessing import Process, Event
from tensorflow import keras



class RLClinet(Process):
    def __init__(self, procName, 
                 env, model, policy,
                 episodes = EPISODES,
                 maxMemorySize = MINIMUM_EXPERIENCE_MEMORY, 
                 sendModelAfter=None,
                 optimizer=keras.optimizers.Adam(learning_rate=LRATE, clipnorm=1),
                 lossFunction=keras.losses.Huber()):
        
        self.terminateSignal = Event()
        
        self.agent = DoubleQNModel(
            env, model, policy,
            episodes, maxMemorySize,
            optimizer=optimizer, lossFunction=lossFunction,
            updateModelAfter=sendModelAfter, federatedLearning=True,
            senderFunction=self.senderFunction,
            waitFunction=self.recieveAndSave,
            terminateSignal = self.terminateSignal,
            procName = procName
        )
        
        
    
    def senderFunction(self, model):
        print("start sending model.")
        
        print("sending model completed.")
    
    def recieveAndSave(self):
        print("start getting model")
        
        print("end getting model")

    def run(self):
        self.agent.train()
    
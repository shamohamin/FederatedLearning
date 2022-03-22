import os
import socket
import threading
import pickle
import copy

from tensorflow import keras
from pyngrok import ngrok
from ..config import CLIENT_NUMBERS, HOST, PORT
import io
import sys

class TCPServer:
    def __init__(self, host=HOST, port=PORT):
        self.__clients = []
        self.sock = None
        self.host = host
        self.port = port
        self.__updateModelThread = None
        self.terminate = False
        self.setupTheConnection()

    def ListenForClients(self):
        raise NotImplementedError

    def sendUpdatedModel(self):
        raise NotImplementedError

    def setupTheConnection(self):
        try:

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind(("", self.port))
            self.sock.listen()
            print("binding completed.")
            self.ngrokTunnel = ngrok.connect(self.port, "tcp")
            print("connected to the tunnel {}".format(
                self.ngrokTunnel.public_url))

            self.__updateModelThread = threading.Thread(
                target=self.sendUpdatedModel, args=(), daemon=True)
            self.__updateModelThread.start()

        except BaseException as ex:
            print("error while binding the address err -> ", ex)

    def terminateTheServer(self):
        self.terminate = True


class FederatedServer(TCPServer):
    def __init__(self, host=HOST, port=PORT, waitUntilModelAppeared=CLIENT_NUMBERS):
        super().__init__(host, port)
        self.waitUntilModelAppeared = waitUntilModelAppeared
        self.clientCounter = 0
        self.modelAppearedSignal = threading.Event()
        self.listenSignal = threading.Event()
        self.lock = threading.Lock()
        self.helperLock = threading.Lock()
        self.listenSignal.set()
        self.bufSize = 1024
        self.listeners = []

    def getModel(self, conn: socket.socket, clientAddr):
        try:
            buff = b''
            while True:
                data = conn.recv(self.bufSize)
                buff = buff + data
                if not data:
                    break

            print("buff len", len(buff))

            model = pickle.load(io.BytesIO(buff))
            conn.close()
            sys.stdout.flush()
            
            with self.helperLock:  # lock other threads from messing the queuing
                with self.lock:
                    self.clientCounter += 1
                    self.__clients.append((model, clientAddr))
                    self.modelAppearedSignal.clear()
                    if self.clientCounter == self.modelAppearedSignal:
                        self.clientCounter = 0
                        self.modelAppearedSignal.set()
                        self.listenSignal.clear()

                self.listenSignal.wait()
        except Exception as ex:
            print("error while getting model in server ", ex)

    def sendUpdatedModel(self):
        while self.terminate:
            self.modelAppearedSignal.wait()
            clients = []
            with self.lock:
                clients.append(copy.deepcopy(self.__clients[0]))
                clients.append(copy.deepcopy(self.__clients[1]))
                del self.__clients[:2]  # delete

            # integerate the models.

            self.modelAppearedSignal.clear()
            self.listenSignal.set()

    def ListenForClients(self):
        try:
            while not self.terminate:
                conn = None
                try:
                    (conn, clientAddr) = self.sock.accept()
                    print(f"{clientAddr} connected.")
                    # non blocking accepts
                    client = threading.Thread(target=self.getModel, args=(
                        conn, clientAddr), daemon=True)
                    client.start()
                    self.listeners.append(client)
                except InterruptedError:
                    break
                except BaseException as ex:
                    print('error in getting connection err => ', ex)
                    break

        finally:
            if self.sock is not None:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()

            for listener in self.listeners:
                try:
                    listener.join()
                except:
                    pass

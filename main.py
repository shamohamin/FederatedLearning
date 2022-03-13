# from src.model.DoubleQNModel import EpsilonGreedyModel
# from tensorflow import keras
# import random
import cv2
from src.model.LayersWrapper import make_env, DQNAgent, load_model
import numpy as np
from baselines.common.atari_wrappers  import make_atari, wrap_deepmind




env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
# env = make_env()
# env.reset()
n_actions = env.action_space.n
state_dim = env.observation_space.shape
agent = DQNAgent("ad", state_dim, n_actions)
agent.network.load_weights("./assets/dqn_model_atari_weights.h5")

model = load_model()
model.load_weights("./_model.h5")

model2 = load_model()
model2.load_weights("./assets/model_1.h5")



def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(11037)
    for i in range(total_episodes):
        state = env.reset()
        # agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            # action = np.argmax(agent.network.predict(np.expand_dims(state, axis=0)))
            # else:
            action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
            
            
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            cv2.imshow("frames", np.array(new_state))
            cv2.waitKey(10)
            # print(reward)
            state = new_state
        
        # env.close()
        print("episode reward ", episode_reward)
        rewards.append(episode_reward)
    
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))

# # epsilonGreedy = EpsilonGreedyModel()
# # epsilonGreedy.train(keras.optimizers.Adam(), keras.losses.Huber(), ["accuracy"])

test(agent, env)
from src.model.policy import EpsilonGreedyPolicy
from src.model.LayersWrapper import load_model
from src.client.RLClient import RLClinet
from baselines.common.atari_wrappers  import make_atari, wrap_deepmind
import sys
import numpy as np
import cv2

model = load_model()

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
policy = EpsilonGreedyPolicy(env.action_space.n)
client1 = RLClinet(policy=policy, model=model, env=env, procName="proc_1")
# client2 = RLClinet(policy=policy, model=model, env=env, procName="proc_2")
# client1.agent.loadStates()
client1.run()
# client2.agent.loadStates()

# client1.agent.procName = "proc_3"
# model1_weights = client1.agent.targetModel.get_weights()
# model2_weights = client2.agent.targetModel.get_weights()

# for i in range(len(model1_weights)):
#     print(len(model1_weights[i]))
#     model1_weights[i] = (model1_weights[i] + model2_weights[i]) / 2.0
#     print(len(model1_weights[i]), model1_weights[i])
#     print("*************************************************")
    
# client1.agent.targetModel.set_weights(model1_weights)
# client1.agent.workerModel.set_weights(model1_weights)

# client1.agent.saver.step.assign_add(1)
# client1.agent.manager.save()

#client.run()
#agent.train()


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(41)
    for i in range(total_episodes):
        state = env.reset()
        # agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            # action = np.argmax(agent.network.predict(np.expand_dims(state, axis=0)))
            # else:
            action = np.argmax(agent.targetModel.predict(np.expand_dims(state, axis=0)))
            
            
            new_state, reward, done, info = env.step(action)
            episode_reward += reward
            new_state_tmp = np.array(new_state)
            new_state_tmp = cv2.resize(new_state_tmp, (new_state_tmp.shape[0]*4, new_state_tmp.shape[1]*4)) 
            cv2.imshow("frames", np.array(new_state_tmp))
            q = cv2.waitKey(1)
            if q == ord('q'):
                break
            # print(reward)
            state = new_state
        
        # env.close()
        print("episode reward ", episode_reward)
        rewards.append(episode_reward)
    
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


# test(client1.agent, env)



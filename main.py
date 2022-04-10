from src.model.policy import EpsilonGreedyPolicy
from src.model.LayersWrapper import load_model
from src.client.RLClient import RLClinet
from baselines.common.atari_wrappers  import make_atari, wrap_deepmind

model = load_model()

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
policy = EpsilonGreedyPolicy(env.action_space.n)
client = RLClinet(policy=policy, model=model, env=env, procName="proc_3")
client.run()
#agent.train()


# def test(agent, env, total_episodes=30):
#     rewards = []
#     env.seed(11037)
#     for i in range(total_episodes):
#         state = env.reset()
#         # agent.init_game_setting()
#         done = False
#         episode_reward = 0.0

#         #playing one game
#         while(not done):
#             # action = np.argmax(agent.network.predict(np.expand_dims(state, axis=0)))
#             # else:
#             action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
            
            
#             new_state, reward, done, info = env.step(action)
#             episode_reward += reward
#             cv2.imshow("frames", np.array(new_state))
#             cv2.waitKey(10)
#             # print(reward)
#             state = new_state
        
#         # env.close()
#         print("episode reward ", episode_reward)
#         rewards.append(episode_reward)
    
#     print('Run %d episodes'%(total_episodes))
#     print('Mean:', np.mean(rewards))


# test(agent, env)
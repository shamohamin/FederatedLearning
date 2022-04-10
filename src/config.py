# General Options
ENV_NAME = "BreakoutNoFrameskip-v4"
EPISODES = 30000
DISCOUNT_FACTOR = 0.99
MINIMUM_EXPERIENCE_MEMORY = 10000
BATCH_SIZE = 32
LRATE = 0.00025
MAX_STEPS_PER_EPISODE = 10000

# Number of frames to take random action and observe output
EPSILON_RANDOM_FRAMES = 50000

# it will be used for epsilon greedy policy
EPSILON = 1
MIN_EPSILON = 0.1
MAX_EPSILON = 1.0

# Number of frames for exploration
EPSILON_GREEDY_FRAMES = 1000000.0

# update target model after 
UPDATE_TARGET_NETWOTK = 50

# train worker model after 
UPDATE_AFTER_ACTIONS = 4

# HOST configuration
HOST = "http://127.0.0.1:5000/"
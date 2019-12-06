import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

#requiered prodcut
GOAL = {'colour': [127.5,127.5,0],
        'volume': 10
        }


DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 10_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'Mix_2x256'
MIN_REWARD = 50  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = True

def colour_convert(list1):
    
    list1 = list1[::-1]
    return list1

class env:
    def __init__(self):
        self.size = 150
        self.resize = self.size 
        self.colour = [[255,0,0], [0,255,0], [255,255,255] ]  #source tank 1 and 2, then the destination tank colour
        self.level = [100,100,0]                              #source tank 1 and 2, then the destination tank levels
        self.tank_info = {'st1': [120,20,0,[255,255,255]],
                          'st2': [120,45,1, [255,255,255]],
                          'dt1': [120,70,2, [255,255,255]] 
                         } 
        
        self.episode_step = 0
        self.OBSERVATION_SPACE_VALUES = (self.resize, self.resize, 3)  # 4
        self.Consumable_usage_cost = 1
        self.Tank_overfill_penalty = 300
        self.Complete_mixture = 100
        self.Action_space = 2
        

    def action(self, choice):
        '''
        Gives us 2 total options. add red (0) or blue (1)
        '''
        #take from source tank
        self.level[choice] = self.level[choice]-1
        self.level[2] = self.level[2]+1
        #deliver to destination tank
        for j in range(3):
            self.colour[2][j] = self.colour[2][j]*((self.level[2]-1)/self.level[2]) + self.colour[choice][j] * (1/self.level[2]) 
        
    def step(self, action):
        self.episode_step +=1
        self.action(action)
        done = False
        reward = 0
        new_obs = np.array(self.get_image())
        good = False  

        c=0
        for i in range(2):
            c += abs(GOAL['colour'][i] - self.colour[2][i])


        if self.level[2] > 100:
            reward = -self.Tank_overfill_penalty
            done = True
        if self.level[2] > GOAL['volume']:     #might have to put something in about colour too...
            reward = -self.Tank_overfill_penalty
            done = True
        
        if self.colour[2] == GOAL['colour']:
            if self.level[2] == GOAL['volume']:
                reward = self.Complete_mixture
                done = True
                good = True
        else:
            reward = -self.Consumable_usage_cost - c/20

        return new_obs, reward, done, good



    def reset(self):
        self.colour = [[255,0,0], [0,255,0], [255,255,255] ]  #source tank 1 and 2, then the destination tank colour
        self.level = [100,100,0]                              #source tank 1 and 2, then the destination tank levels
        
        self.episode_step = 0
        env_plot = np.array(self.get_image())
        return env_plot
    
    def get_image(self):
        # starts an rbg of our size
        env_plot = np.zeros((self.size, self.size, 3), dtype=np.uint8)  
        
        #draw the env
        for tank in self.tank_info:
            y = self.tank_info[tank][0]
            x = self.tank_info[tank][1]
            seq = self.tank_info[tank][2]
            tank_wall_colour = self.tank_info[tank][3]
            #Contense of tank
            for l in range(self.level[seq],0,-1):
                env_plot[y-l][x:x+10] = colour_convert(self.colour[seq])
            
            #Side of tank
            for i in range(110):
                env_plot[y-110+i][x-2:x] = colour_convert(tank_wall_colour)
            for i in range(110):
                env_plot[y-110+i][x+10:x+12] = colour_convert(tank_wall_colour)
            for i in range(2):
                env_plot[y+i][x-2:x+12] = colour_convert(tank_wall_colour)

            
            if tank == 'dt1':
                #Goal example
                for l in range(GOAL['volume'],0,-1):
                    env_plot[y-l][x+25:x+35] = colour_convert(GOAL['colour'])
                #Side of tank
                for i in range(110):
                    env_plot[y-110+i][x+25-2:x+25] = colour_convert(tank_wall_colour)
                for i in range(110):
                    env_plot[y-110+i][x+35:x+2+35] = colour_convert(tank_wall_colour)
                for i in range(2):
                    env_plot[y+i][x+25-2:x+35+2] = colour_convert(tank_wall_colour)

        img = Image.fromarray(env_plot, mode='RGB')  # reading to rgb.
        
        img = img.resize((self.resize, self.resize)) #resize to decrease vram usage
        return img

        



    def render(self,done,good):
        img = self.get_image()
        img = img.resize((600, 600))  # resizing so we can see
        cv2.imshow("Tank Level", np.array(img))  # show it!
        cv2.waitKey(50)

        if good:
            cv2.imshow("Tank Level", np.array(img))  # show it!
            img2 = cv2.imread('gg.jpg',0)
            cv2.imshow("DONE", np.array(img2))  # show done
            cv2.waitKey(200)
            cv2.destroyAllWindows()





    
env = env()

#testing environment
'''
for i in range(4):
    new_obs, reward, done = env.step(0)
    print(reward, done)
    env.render(done)
for i in range(8):
    new_obs, reward, done = env.step(1)
    env.render(done)
    print(reward, done)
    if done:
        break

'''

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), input_shape=((env.OBSERVATION_SPACE_VALUES))))  # OBSERVATION_SPACE_VALUES = (150, 150, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(32))

        model.add(Dense(env.Action_space, activation='linear'))  # action space = 2
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]


agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.Action_space)

        new_state, reward, done, good = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW: #and not episode % AGGREGATE_STATS_EVERY:
            env.render(done, good)

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    
    print (' min', 'av',  'max')
    print (min_reward, average_reward,  max_reward)
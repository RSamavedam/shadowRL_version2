# -*- coding: utf-8 -*-
import random
#import gym
from TestingEnvironment import Environment
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPool2D
from keras.optimizers import Adam
from tensorflow import keras

EPISODES = 1001

class DQNAgent:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.action_size = input_dim[0] * input_dim[1]
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Convolution2D(32, 3, 1, activation="relu", input_shape=self.input_dim))
        model.add(Flatten())
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.reshape(state, (1, 30, 30, 3))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        sumValue = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            better_next_state = np.reshape(next_state, (1, 30, 30, 3))
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(better_next_state)[0]))
            better_state = np.reshape(state, (1, 30, 30, 3))
            target_f = self.model.predict(better_state)
            original = target_f[0][action]
            sumValue += (original - target) * (original - target)
            target_f[0][action] = target
            states.append(state[:])
            targets.append(target_f[:])
            #better_state = state.reshape(1, 30, 30, 3)
            #print(better_state.shape)
            self.model.fit(better_state, target_f, epochs=1, verbose=0)
        print("Fitting a batch...")
        sumValue /= batch_size
        outputFile = open("best_model_mse.txt", "a")
        outputFile.write(str(sumValue) + "\n")
        outputFile.close()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model = keras.models.load_model(name)

    def save(self, name):
        self.model.save(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    env = Environment(30, 30, 30)
    #state_size = env.observation_space.shape[0]
    #action_size = env.action_space.n
    agent = DQNAgent((env.dim1, env.dim2, 3))
    #agent.load("./save/heat_island_learner.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        totalScore = 0
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done= env.step(action)
            #reward = reward if not done else -10
            totalScore += reward
            #next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, totalScore, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        #if e % 100 == 0:
            #agent.save("./save/heat_island_learner_architecture_main_model_.001_" + str(e) + "_.h5")

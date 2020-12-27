# Wrapper of environment
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines import bench, logger

import numpy as np
import heapq
import random

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from meta_aad.utils import read_data_as_matrix, run_iforest

class EnvBase(gym.Env):
    def __init__(self, datapath="data/toy.csv", budget=100):
        # Read dataset
        X_train, labels, anomalies = read_data_as_matrix(datapath)
        self.X_train = X_train
        self.labels = labels
        self.size = len(self.labels)
        self.budget = budget
        self.dim = X_train.shape[1]
        self.state_dim = 6

        # Unsupervised scores
        self.scores = np.expand_dims(run_iforest(self.X_train), axis=1)
        self.scores = (self.scores-np.mean(self.scores, axis=0)[None,:]) / np.std(self.scores, axis=0)[None,:]

        # Exatract distances features
        self.X_train = StandardScaler().fit_transform(self.X_train)
        self.distances = euclidean_distances(self.X_train, self.X_train)
        self.distances = (self.distances - np.mean(self.distances, axis=1)[:,None]) / np.std(self.distances, axis=1)[:,None]
        self.nearest_neighbors = np.argpartition(self.distances, 10)[:,:10]

        print("Data loaded: {} Total instances: {} Anomalies: {}".format(datapath, self.size, len(anomalies)))

        # Gym settings
        self.action_space = spaces.Discrete(2)
        high = np.ones(self.state_dim) * 10.0
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

class TrainEnv(EnvBase):

    def __init__(self, datapath):
        super().__init__(datapath=datapath)

    def step(self, action):
        """ Proceed to the next state given the curernt action
            1 for check the instance, 0 for not
            return next state, reward and done
        """
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action == 0:
            r = 0
        elif action == 1:
            if self._Y[self.pointer] == 0:
                r = 1
                self.anomalies.append(self.pointer)

            else:
                r = -0.1
                self.normalies.append(self.pointer)
            self.count += 1
        self.pointer += 1

        # Set maximum lenths to 2000
        if self.pointer >= self.size or self.pointer >= 2000:
            self.done = True
        else:
            self.done = False

        return self._obs(), r, self.done, {}

    def reset(self):
        """ Reset the environment, for streaming evaluation
        """
        self._process_data()

        # Some stats
        self.pointer = 0
        self.count = 0
        self.done = False
        self.anomalies = []
        self.normalies = []
        self.labeled = []

        return self._obs()


    def _process_data(self):
        # Shuffle data
        self.indices = indices = np.random.choice(self.size, self.size, replace=False)
        self._X = self.X_train[indices]
        self._Y = self.labels[indices]
        self._scores = self.scores[indices]

    def _obs(self):
        """ Return the observation of the current state
        """
        if self.done:
            return np.zeros(self.state_dim)

        features = []
        ori_pointer = self.indices[self.pointer]
        ori_anomalies = self.indices[self.anomalies]
        ori_nomalies = self.indices[self.normalies]
        near_anomalies = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_anomalies), 1, 0)
        near_normalies = np.where(np.isin(self.nearest_neighbors[ori_pointer], ori_nomalies), 1, 0)
        features.append(1) if np.count_nonzero(near_anomalies[:5])>0 else features.append(0)

        # Avg distance to abnormal instances
        a = np.mean(self.distances[ori_pointer,ori_anomalies]) if len(self.anomalies) > 0 else 0
        a_min = np.min(self.distances[ori_pointer,ori_anomalies]) if len(self.anomalies) > 0 else 0
        # Avg distance to normal instances
        n = np.mean(self.distances[ori_pointer,ori_nomalies]) if len(self.normalies) > 0 else 0
        n_min = np.min(self.distances[ori_pointer,ori_nomalies]) if len(self.normalies) > 0 else 0

        features.extend([a, a_min, n, n_min])
        c = self._scores[self.pointer]
        features.extend(c)

        return features
        
class EnsembleTrainEnv(gym.Env):
    def __init__(self, datapaths):
        self.envs = []
        for datapath in datapaths:
            self.envs.append(TrainEnv(datapath))

        # Gym settings
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.seed()
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        for env in self.envs:
            env.seed(seed)
        return [seed]

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env = random.choice(self.envs)
        return self.env.reset()

    def render(self):
        pass

class EvalEnv(EnvBase):

    def step(self, action):
        """ Evaluation step
        """
        assert action in self.legal
        if self._Y[action] == 0:
            r = 1
            self.anomalies.append(action)
        else:
            r = 0
            self.normalies.append(action)
        self.count += 1
        self.labeled.append(action)
        self.legal.remove(action)
        if self.count >= self.budget:
            self.done = True
        else:
            self.done = False
        s = self._obs()
        return s, self.legal, r, self.done, {}
    
    def reset(self):
        """ Evaluation reset
        """
        self._X = self.X_train
        self._Y = self.labels
        self._scores = self.scores

        # Some stats
        self.count = 0
        self.done = False
        self.anomalies = []
        self.normalies = []
        self.labeled = []
        self.legal = [i for i in range(self.size)]

        return self._obs(), self.legal
        
    def _obs(self):
        """ Return the observation of the current state
        """
        if self.done:
            return np.zeros(self.state_dim)

        near_anomalies = np.where(np.isin(self.nearest_neighbors, self.anomalies), 1, 0)
        near_normalies = np.where(np.isin(self.nearest_neighbors, self.normalies), 1, 0)
        a_top_5 = np.where(np.count_nonzero(near_anomalies[:,:5], axis=1)>0, 1, 0)

        # Avg distance to abnormal instances
        a = np.mean(self.distances[:,self.anomalies], axis=1) if len(self.anomalies) > 0 else np.zeros(self.size) 
        a_min = np.min(self.distances[:,self.anomalies], axis=1) if len(self.anomalies) > 0 else np.zeros(self.size)
        # Avg distance to normal instances
        n = np.mean(self.distances[:,self.normalies], axis=1) if len(self.normalies) > 0 else np.zeros(self.size)
        n_min = np.min(self.distances[:,self.normalies], axis=1) if len(self.normalies) > 0 else np.zeros(self.size)

        c = self._scores
        features = np.concatenate((
                                   np.expand_dims(a_top_5, axis=1),
                                   np.expand_dims(a, axis=1),
                                   np.expand_dims(a_min, axis=1),
                                   np.expand_dims(n, axis=1),
                                   np.expand_dims(n_min, axis=1),
                                   c),
                                   axis=1)

        return features

def make_train_env(datapaths):
    if len(datapaths) > 1:
        env = EnsembleTrainEnv(datapaths)
    else:
        env = TrainEnv(datapaths[0])
    env = bench.Monitor(env, logger.get_dir())
    return env

def make_eval_env(datapath, budget):
    env = EvalEnv(datapath, budget)
    return env



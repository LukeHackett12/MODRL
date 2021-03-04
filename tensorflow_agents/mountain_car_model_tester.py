from custom_envs.mountain_car.engine import MountainCar
from tensorflow import keras
import numpy as np
import tensorflow as tf


class DQNAgentTester:
    def __init__(self, model):
        self.model = keras.models.load_model(model)

    def selectAction(self, state):
        preds = np.squeeze(self.model(
            state, training=False).numpy(), axis=0)
        action = np.argmax(preds)
        return action


class MountainCarModelTester(object):
    def __init__(self, model):
        self.model = model
        self.env = MountainCar(speed=1e8, graphical_state=False,
                               render=True, is_debug=False, random_starts=True)
        self.agent = DQNAgentTester(self.model)

    def test(self):
        scores = np.array([])
        i = 0
        while True:
            done = False

            state = self.env.reset().reshape(1, 2)
            maxHeight = -10000

            while not done:
                action = self.agent.selectAction(state)
                obs, reward, done, totalScore = self.env.step_all(action)
                state = obs.reshape(1, 2)

            if totalScore > 200:
                print("DNF")

            scores = np.append(scores, totalScore)
            print("Average Score: {}".format(scores.mean()))

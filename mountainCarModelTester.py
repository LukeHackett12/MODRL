import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras
from custom_envs.mountain_car.engine import MountainCar


class DQNAgentTester:
    def __init__(self, model):
        self.model = keras.models.load_model(model)

    def selectAction(self, state):
        preds = np.squeeze(self.model(
            state, training=False).numpy(), axis=0)
        action = np.argmax(preds)
        return action

model = "./mountain_car_tfngmo_132323.chkpt"

env = MountainCar(speed=1e8, graphical_state=False,
                  render=True, is_debug=False, random_starts=True)
agent = DQNAgentTester(model)

scores = np.array([])
i = 0
while True:
    done = False

    state = env.reset().reshape(1, 2)
    maxHeight = -10000

    while not done:
        action = agent.selectAction(state)
        obs, reward, done, totalScore = env.step_all(action)
        state = obs.reshape(1, 2)

    scores = np.append(scores, totalScore)
    print("Average Score: {}".format(scores.mean()))
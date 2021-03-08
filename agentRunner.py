import argparse


from tensorflow_agents.mountain_car_model_tester import MountainCarModelTester
from tensorflow_agents.mountain_car_mo_baseline import MultiObjectiveMountainCar
from tensorflow_agents.mountain_car_mo_wdqn import MultiObjectiveWMountainCar
from tensorflow_agents.mountain_car_mog import MultiObjectiveMountainCarGraphical
from tensorflow_agents.mountain_car_open_ai import OpenAIMountainCar
'''
from tensorflow_agents.deep_sea_mo_baseline import DeepSeaTreasureBaseline
'''
from tensorflow_agents.mario_baseline import MarioBaseline

parser = argparse.ArgumentParser(description='Run agentArg model for game')
parser.add_argument("-a", "--agentArg", required=True)

args = parser.parse_args()
agentArg = args.agentArg

if agentArg == 'mountain_car_mo_baseline':
  agent = MultiObjectiveMountainCar(2000)
elif agentArg == 'mountain_car_mo_wdqn':
  agent = MultiObjectiveWMountainCar(2000)
elif agentArg == 'mountain_car_mog':
  agent = MultiObjectiveMountainCarGraphical(2000)
elif agentArg == 'mountain_car_open_ai':
  agent = OpenAIMountainCar(2000)
elif agentArg == 'deep_sea_mo_baseline':
  agent = DeepSeaTreasureBaseline(2000)
elif agentArg == 'mario_baseline':
  agent = MarioBaseline(2000)

agent.train()

'''
agentArg = MountainCarModelTester("./mountain_car_wnet_54540.chkpt")
agentArg.test()
'''
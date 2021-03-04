from tensorflow_agents.mountain_car_model_tester import MountainCarModelTester
from tensorflow_agents.mountain_car_mo_baseline import MultiObjectiveMountainCar
from tensorflow_agents.mountain_car_mo_wdqn import MultiObjectiveWMountainCar
from tensorflow_agents.mountain_car_mog import MultiObjectiveMountainCarGraphical
from tensorflow_agents.mountain_car_open_ai import OpenAIMountainCar

from tensorflow_agents.deep_sea_mo_baseline import DeepSeaTreasureBaseline

if __name__ == '__main__':

  agent = MultiObjectiveWMountainCar(2000)
  agent.train()

  agent = MountainCarModelTester("./mountain_car_wnet_54540.chkpt")
  agent.test()

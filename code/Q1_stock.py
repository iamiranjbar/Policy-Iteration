import numpy as np

from amalearn.environment import MDPEnvironment
from amalearn.agent import StockAgent

STATES_COUNT = 20000

def run():
    environment = MDPEnvironment(STATES_COUNT, '1')
    agent = StockAgent('1', environment, 0.9)
    agent.iterate_policy()
    initial_wealth = 20
    initial_b_value = 5
    initial_c_value = 15
    initial_d_value = 10
    initial_state_index = ((initial_wealth//5)-1) * 1000 + ((initial_b_value//5)-1) * 100 + ((initial_c_value//5)-1) * 10 + ((initial_d_value//5)-1)
    agent.print_optimal_policy(initial_state_index)

if __name__ == "__main__":
    run()

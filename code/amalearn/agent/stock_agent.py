import numpy as np
import copy
from amalearn.agent import AgentBase

STATES_COUNT = 20000
NOTHING = 3
B = 0
C = 1
D = 2
ACTIONS = [NOTHING, B, C, D, (B, C), (B, D), (C, D)]

class StockAgent(AgentBase):
    def __init__(self, id, environment, discount_factor):
        super(StockAgent, self).__init__(id, environment)
        self.discount_factor = discount_factor
        self.tetha = 0.01
        self.state_value = [0 for _ in range(STATES_COUNT)]
        # Because we now that optimal policy is greedy
        self.policy = [0 for _ in range(STATES_COUNT)]

    def evaluate_policy(self):
        stock_values = [index * 5 for index in range(1, 11)]
        wealth_values = [index * 5 for index in range(1, 21)]
        while True:
            delta = 0
            for wealth_index, wealth_value in enumerate(wealth_values):
                for b_index, b_value in enumerate(stock_values):
                    for c_index, c_value in enumerate(stock_values):
                        for d_index, d_value in enumerate(stock_values):
                            if self.environment.is_terminal(wealth_value):
                                continue
                            state_index = wealth_index * 1000 + b_index * 100 + c_index * 10 + d_index
                            old_state_value = self.state_value[state_index]
                            optimal_action = self.policy[state_index]
                            next_state_probability_reward = self.environment.get_probability_and_reward(wealth_value, b_value, c_value, d_value, optimal_action)
                            new_state_value = 0
                            for state_probability_reward in next_state_probability_reward:
                                next_state_index, reward, probability = state_probability_reward
                                new_state_value += (probability * (reward + self.discount_factor * self.state_value[next_state_index]))
                            new_diff = abs(new_state_value - old_state_value)
                            if new_diff > delta:
                                delta = new_diff
                            self.state_value[state_index] = new_state_value
            print("Delta: {}".format(delta))
            if delta < self.tetha:
                break

    def improve_policy(self):
        policy_stable = True
        stock_values = [index * 5 for index in range(1, 11)]
        wealth_values = [index * 5 for index in range(1, 21)]
        for wealth_index, wealth_value in enumerate(wealth_values):
            for b_index, b_value in enumerate(stock_values):
                for c_index, c_value in enumerate(stock_values):
                    for d_index, d_value in enumerate(stock_values):
                        state_index = wealth_index * 1000 + b_index * 100 + c_index * 10 + d_index
                        old_action = self.policy[state_index]
                        action_values = []
                        for action_index in range(len(ACTIONS)):
                            next_state_probability_reward = self.environment.get_probability_and_reward(wealth_value, b_value, c_value, d_value, action_index)
                            new_action_value = 0
                            for state_probability_reward in next_state_probability_reward:
                                next_state_index, reward, probability = state_probability_reward
                                new_action_value += (probability * (reward + self.discount_factor * self.state_value[next_state_index]))
                            action_values.append(new_action_value)
                        if state_index == 3021:
                            print("Action values:\n {}".format(action_values))
                        new_optimal_action = np.argmax(np.array(action_values))
                        self.policy[state_index] = new_optimal_action
                        if new_optimal_action != old_action:
                            policy_stable = False
        return policy_stable
        

    def iterate_policy(self):
        iteration_count = 0
        while True:
            iteration_count += 1
            print("Iteration #: {}".format(iteration_count))
            self.evaluate_policy()
            policy_stable = self.improve_policy()
            print("__________________________________________")
            if policy_stable:
                break
    
    def take_action(self) -> (object, float, bool, object):
        return None, 0, False, None

    def print_optimal_policy(self, initial_state_index):
        optimal_action_index = self.policy[initial_state_index]
        optimal_action = ACTIONS[optimal_action_index]
        print("Optimal Action:")
        if optimal_action == NOTHING:
            print("Nothing")
        if optimal_action == B or (type(optimal_action) is tuple and B in optimal_action):
            print("B")
        if optimal_action == C or (type(optimal_action) is tuple and C in optimal_action):
            print("C")
        if optimal_action == D or (type(optimal_action) is tuple and D in optimal_action):
            print("D")

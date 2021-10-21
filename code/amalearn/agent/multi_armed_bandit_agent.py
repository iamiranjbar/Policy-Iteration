import math
import numpy as np

from amalearn.agent import AgentBase


class MultiArmedBanditAgent(AgentBase):
    def __init__(self, id, environment):
        super(MultiArmedBanditAgent, self).__init__(id, environment)
        available_actions = self.environment.available_actions()
        self.expected_means = [0 for _ in range(available_actions)]
        self.expected_pis = [1 for _ in range(available_actions)]

    def calculate_std_from_pi(self, pi):
        return math.sqrt(1 / pi)
    
    def choose_temporal_best_action(self):
        samples = [np.random.normal(loc=mean, scale=self.calculate_std_from_pi(pi)) 
                   for mean, pi in zip(self.expected_means, self.expected_pis)]
        chosen_action = samples.index(max(samples))
        return chosen_action
    
    def update_statistics(self, action, reward):
        old_pi = self.expected_pis[action]
        old_mean = self.expected_means[action]
        reward_pi = 1
        new_pi = old_pi + reward_pi
        learning_rate = reward_pi / new_pi
        new_mean = old_mean + (learning_rate * (reward - old_mean))
        self.expected_pis[action] = new_pi
        self.expected_means[action] = new_mean
    
    def take_action(self) -> (object, float, bool, object):
        chosen_action = self.choose_temporal_best_action()
        _, reward, _, _ = self.environment.step(chosen_action)
        self.update_statistics(chosen_action, reward)
        return chosen_action, reward

    def get_best_action(self):
        return self.expected_means.index(max(self.expected_means))

    def calculate_regret(self, results):
        optimal_action_reward = max(self.expected_means)
        regret = 0
        for result in results:
            chosen_action, _ = result
            regret += optimal_action_reward - self.expected_means[chosen_action]
        return regret

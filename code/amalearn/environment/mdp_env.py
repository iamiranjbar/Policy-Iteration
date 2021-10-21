import gym
import random
from amalearn.environment import EnvironmentBase

INCREASE = 1
NO_CHANGE = 0
DECREASE = -1
CHANGES = [INCREASE, NO_CHANGE, DECREASE]
STOCK_MIN_VALUE = 5
STOCK_MAX_VALUE = 50
STOCK_DAILY_CHANGE = 5
NOTHING = 3
B = 0
C = 1
D = 2
ACTIONS = [NOTHING, B, C, D, (B, C), (B, D), (C, D)]

class MDPEnvironment(EnvironmentBase):
    def __init__(self, states_count, id, container=None):
        state_space = gym.spaces.Discrete(states_count)
        action_space = gym.spaces.Discrete(7)
        super(MDPEnvironment, self).__init__(action_space, state_space, id, container)
        self.target_wealth = 100
        self.wealth = 20
        self.b_value = 5
        self.c_value = 15
        self.d_value = 10
        self.values = [self.b_value, self.c_value, self.d_value]
        self.b_probabilities = {INCREASE: 0.4, NO_CHANGE: 0.3, DECREASE: 0.3}
        self.c_probabilities = {INCREASE: 0.1, NO_CHANGE: 0.8, DECREASE: 0.1}
        self.d_probabilities = {INCREASE: 0.2, NO_CHANGE: 0.1, DECREASE: 0.7}
        self.probabilities = [self.b_probabilities, self.c_probabilities, self.d_probabilities]
        self.optimum_stay_chance = 0.25
        self.prev_b_value = 5
        self.prev_c_value = 15
        self.prev_d_value = 10
        self.prev_values = [self.prev_b_value, self.prev_c_value, self.prev_d_value]
        self.days_count = 0
        self.next_states = dict()
        self.generate_state_reward_probabilities()

    def calculate_reward(self, action):
        reward = 0
        if action != NOTHING:
            if action in [B, C, D]:
                reward = self.values[action] - self.prev_values[action]
            else:
                for decision in action:
                    reward += (self.values[decision] - self.prev_values[decision])
        return reward - self.days_count

    def terminated(self):
        return self.wealth == self.target_wealth

    def is_terminal(self, wealth):
        return wealth == self.target_wealth

    def observe(self):
        return {}

    def change_stock_value(self, probababilities, prev_value, value):
        random_value = random.random()
        prev_value = value
        if value == STOCK_MIN_VALUE:
            if random_value > self.optimum_stay_chance:
                value += STOCK_DAILY_CHANGE
        elif value == STOCK_MAX_VALUE:
            if random_value > self.optimum_stay_chance:
                value -= STOCK_DAILY_CHANGE
        else:
            increase_chance = probababilities[INCREASE]
            no_change_chance = probababilities[NO_CHANGE]
            if random_value < increase_chance:
                value += STOCK_DAILY_CHANGE
            elif random_value > increase_chance + no_change_chance:
                value -= STOCK_DAILY_CHANGE

    def next_state(self, action):
        for stock in [B, C, D]:
            self.change_stock_value(self.probabilities[stock], self.prev_b_value[stock], 
                                    self.b_value[stock])
        self.days_count += 1
        if action != NOTHING:
            if action in [B, C, D]:
                self.wealth += (self.values[action] - self.prev_values[action])
            else:
                for decision in action:
                    self.wealth += (self.values[decision] - self.prev_values[decision])

    def reset(self):
        self.wealth = 20
        self.b_value = 5
        self.c_value = 15
        self.d_value = 10
        self.days_count = 0

    def render(self, mode='human'):
        return

    def close(self):
        return

    def step(self, action):
        self.next_state(action)
        reward = self.calculate_reward(action)
        done = self.terminated()
        return reward, done

    def get_limit_probabilities(self, value, change, prob):
        if value == STOCK_MIN_VALUE:
            if change == INCREASE:
                prob = 1 - self.optimum_stay_chance
            elif change == NOTHING:
                prob = self.optimum_stay_chance
            else:
                prob = 0
        elif value == STOCK_MAX_VALUE:
            if change == DECREASE:
                prob = 1 - self.optimum_stay_chance
            elif change == NOTHING:
                prob = self.optimum_stay_chance
            else:
                prob = 0
        return prob


    def generate_state_reward_probabilities(self):
        stock_values = [index * 5 for index in range(1, 11)]
        wealth_values = [index * 5 for index in range(1, 21)]
        for wealth_value in wealth_values:
            for b_value in stock_values:
                for c_value in stock_values:
                    for d_value in stock_values:
                        for action_index, action in enumerate(ACTIONS):
                            dictionary_key = (wealth_value, b_value, c_value, d_value, action_index)
                            next_state_plus_rewards = []
                            for b_change in CHANGES:
                                for c_change in CHANGES:
                                    for d_change in CHANGES:
                                        b_prob = self.b_probabilities[b_change]
                                        c_prob = self.c_probabilities[c_change]
                                        d_prob = self.d_probabilities[d_change]
                                        b_prob = self.get_limit_probabilities(b_value, b_change, b_prob)
                                        c_prob = self.get_limit_probabilities(c_value, c_change, c_prob)
                                        d_prob = self.get_limit_probabilities(d_value, d_change, d_prob)
                                        probability = 1
                                        new_wealth_value = wealth_value
                                        if action != NOTHING:
                                            if action == B or (type(action) is tuple and B in action):
                                                new_wealth_value += (b_change*5)
                                            if action == C or (type(action) is tuple and C in action):
                                                new_wealth_value += (c_change*5)
                                            if action == D or (type(action) is tuple and D in action):
                                                new_wealth_value += (d_change*5)
                                        new_b_value = b_value + (b_change * 5)
                                        new_c_value = c_value + (c_change * 5)
                                        new_d_value = d_value + (d_change * 5)
                                        if (new_b_value < STOCK_MIN_VALUE or new_b_value > STOCK_MAX_VALUE) or \
                                            (new_c_value < STOCK_MIN_VALUE or new_c_value > STOCK_MAX_VALUE) or \
                                            (new_d_value < STOCK_MIN_VALUE or new_d_value > STOCK_MAX_VALUE) or \
                                            (new_wealth_value > 100 or new_wealth_value < 5):
                                            continue
                                        next_state_index = (new_wealth_value//5-1) * 1000 + (new_b_value//5-1) * 100 + (new_c_value//5-1) * 10 + (new_d_value//5-1)
                                        probability = b_prob * c_prob * d_prob
                                        state_plus_reward = (next_state_index, new_wealth_value - wealth_value, probability)
                                        next_state_plus_rewards.append(state_plus_reward)
                            self.next_states[dictionary_key] = next_state_plus_rewards

    def get_probability_and_reward(self, wealth_value, b_value, c_value, d_value, action):
        dictionary_key = (wealth_value, b_value, c_value, d_value, action)
        return self.next_states[dictionary_key]

import numpy as np
import random

from core.models import NeuralNetwork


class Policy:

    def __init__(self, env):
        """Initialize Policy by saving the environment as an instance variable and assigning it a turn"""
        self.environment = env
        self.turn = None
        self.model = None

    def sample(self, state):
        """Samples an action given a state"""

    def collect(self, state, action, reward):
        """Stores in memory a state, an action and a reward"""
        pass

    def update(self):
        """Performs any updates needed"""
        pass


class NeuralNetPolicy(Policy):

    def __init__(self, env, hidden_layers, alpha=.01, gamma=.95, alpha_decay_factor=None, eps_decay_factor=None,
                 trainable=True):
        """Initializes the Policy
        :param env: Environment the policy acts on
        :type env: TicTacToe
        :param hidden_layers: List of the number of neurons per hidden layer
        :type hidden_layers: list[int]
        :param alpha: Learning rate of the neural net
        :type alpha: float
        :param gamma: Discount factor the policy uses
        :type gamma: float
        :param alpha_decay_factor: Multiplicative decay factor to apply to alpha after a gradient descent update. No
            decay applied by default
        :type alpha_decay_factor: float or None
        :param eps_decay_factor: Multiplicative decay factor to apply to epsilon after a policy update. No eps-greedy
            applied by default
        :type eps_decay_factor: float or None
        :param trainable: Weather we want to train this net or not
        :type trainable: bool
        """
        super().__init__(env)
        self.afterstate_values = {}
        self.afterstate_visits = {}
        self.states_actions_rewards = []
        self.gamma = gamma
        self.model = NeuralNetwork(self.environment.state_dimension, *hidden_layers, 1, alpha=alpha,
                                   decay_factor=alpha_decay_factor)
        if eps_decay_factor:
            self.epsilon = .9
            self.eps_decay_factor = eps_decay_factor
        else:
            self.epsilon = None
        self.trainable = trainable

    def sample(self, state):
        """Samples an action given a state
        :param state: Tuple of 9 integers representing the state of the game
        :type state: tuple[int]
        :return: The desired action to perform, ranging from 0 to 8
        :rtype: int
        """
        if self.epsilon:
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.environment.get_action_space(state))

        actions = self.environment.get_action_space(state)
        random.shuffle(actions)
        best_afterstate_value = float('-inf')

        for action in actions:
            afterstate = self.environment.get_afterstate(state, action, self.turn)
            afterstate_value = self.model.predict(np.array(afterstate, dtype=np.float64).reshape([self.environment.state_dimension, 1]))
            if afterstate_value > best_afterstate_value:
                best_action = action
                best_afterstate_value = afterstate_value

        return best_action

    def collect(self, state, action, reward):
        """Stores in memory a state, an action and a reward obtained in one timestep"""
        self.states_actions_rewards.append((state, action, reward))

    def update(self):
        """Performs a Monte Carlo update of the afterstate estimates once the episode has finished"""
        G = 0
        for state, action, reward in reversed(self.states_actions_rewards):
            G = reward + self.gamma * G
            afterstate = self.environment.get_afterstate(state, action, self.turn)
            old_value = self.afterstate_values.get(afterstate, 0)
            no_visits = self.afterstate_visits.get(afterstate, 0) + 1
            self.afterstate_visits[afterstate] = no_visits
            new_value = old_value + (1/no_visits) * (G - old_value)
            self.afterstate_values[afterstate] = new_value

        self.states_actions_rewards = []

    def update_gradient(self):
        """Updates the gradient using the encountered afterstates so far and their corresponding Monte Carlo estimates
            of their values. It then clears these estimates, as the policy has changed and so has the value function. It
            also decays epsilon if using eps-greedy"""
        X = np.array(list(self.afterstate_values.keys()), dtype=np.float64)
        Y = np.array(list(self.afterstate_values.values()))

        if self.trainable:
            self.model.gd_step(X.T, Y)
        self.afterstate_visits = {}
        self.afterstate_values = {}
        if self.epsilon:
            self.epsilon *= self.eps_decay_factor


class RandomPolicy(Policy):

    def sample(self, state):
        """Samples a random action from a given state"""
        return np.random.choice(self.environment.get_action_space(state))


class OneStepPolicy(Policy):

    def sample(self, state):
        """Samples the action that yields the best immediate reward from a given state. It only considers afterstates,
            not the moves the opponent might make afterwards"""
        actions = self.environment.get_action_space(state)
        random.shuffle(actions)
        best_afterstate_value = float('-inf')

        for action in actions:
            afterstate = self.environment.get_afterstate(state, action, self.turn)
            afterstate_value = (self.environment.get_reward_player1(
                grid=np.array(afterstate)) or 0) * self.environment.turn
            if afterstate_value > best_afterstate_value:
                best_action = action
                best_afterstate_value = afterstate_value

        return best_action


class HumanPolicy(Policy):

    def sample(self, state):
        """Samples an action from a state with User input"""
        actions = self.environment.get_action_space(state)
        print('Possible actions: ', actions)
        return int(input('Input your action'))
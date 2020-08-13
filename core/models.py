import numpy as np
import pandas as pd
import os

from core.fastmath import backprop, forward_prop


class TicTacToe:

    def __init__(self):
        """"Initializes the grid to an array of 9 zeros and sets the turn to 1, where X=1 and O=-1"""
        self.grid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.turn = 1

    @property
    def current_state(self):
        """"Returns the current state of the environment
        :returns: 9 integers as a tuple representing the state
        :rtype: tuple[int]
        """
        return tuple(self.grid)

    @property
    def state_dimension(self):
        """"Returns the number of dimensions the state vector has"""
        return 9

    def get_action_space(self, state):
        """"Returns a list of possible actions for a given state
        :param state: 9 integers as a tuple representing the state
        :type state: tuple[int]
        :returns: List of integers from 0 to 8 corresponding to the available actions
        :rtype: list[int]
        """
        return list(np.where(np.array(state) == 0)[0])

    def get_afterstate(self, state, action, turn):
        """"Returns an afterstate given a state, an action and a turn
        :param state: 9 integers as a tuple representing the state
        :type state: tuple[int]
        :param action: Integer representing the action that is taken
        :type action: int
        :param turn: 1 corresponds to Player 1 (X) and -1 corresponds to Player 2 (O)
        :type turn: int
        :returns: 9 integers as a tuple representing the afterstate
        :rtype: tuple[int]
        """
        grid = np.array(state)
        grid[action] = turn
        return tuple(grid)

    def get_reward_player1(self, grid = None):
        """Checks the reward obtained by Player 1 (X)
        :param grid: Numpy array with shape (9,) representing a state. If not specified, checks will be made on the
            current position of the grid
        :type grid: np.array
        :returns: 1 if Player 1 (X) has won, -1 if it has lost, 0 if there's a tie and None otherwise
        :rtype: int or None
        """
        if grid is None:
            grid = self.grid

        for i in range(0, 7, 3):
            if grid[i] == grid[i + 1] == grid[i + 2] and grid[i] != 0:
                return grid[i]
        for i in range(3):
            if grid[i] == grid[i + 3] == grid[i + 6] and grid[i] != 0:
                return grid[i]
        if (grid[0] == grid[4] == grid[8] or grid[2] == grid[4] == grid[6]) and grid[
            4] != 0:
            return grid[4]

        if len(list(np.where(np.array(grid) == 0)[0])) == 0:
            return 0

    def is_terminal(self):
        """Returns True if the current state of the environment is terminal"""
        return self.get_reward_player1() is not None

    def reset(self):
        """Resets the grid to its initial state"""
        self.grid = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.turn = 1

    def play(self, action):
        """Plays an action one the environment and flips the turn"""
        self.grid[action] = self.turn
        self.turn *= -1

    def print_grid(self, grid=None):
        """Pretty-prints the grid"""
        if grid is None:
            grid = self.grid
        for i in range(3):
            s = ""
            for j in range(3):
                s += 'X' if grid[i * 3 + j] == 1 else ('O' if grid[i * 3 + j] == -1 else '-')
            print(s)

    def __repr__(self):
        return 'TicTacToe'


class NeuralNetwork:

    def __init__(self, *layers, alpha=.01, decay_factor=None):
        """Initializes the Neural Network
        :param `*layers`: The number of neurons on each layer. Must include the input and the output layers
        :type layers: int
        :param alpha: Learning rate of the network
        :type alpha: float
        :param decay_factor: Multiplicative decay factor to apply to alpha after a gradient descent update. No decay
            applied by default
        :type decay_factor: float
        """
        self.n_layers = len(layers) - 1
        self.W = [np.random.randn(layers[i+1], layers[i]) / np.sqrt(layers[i] + layers[i+1]) for i in range(self.n_layers)]
        self.b = [np.random.randn(layers[i+1], 1) / np.sqrt(layers[i] + layers[i+1]) for i in range(self.n_layers)]
        self.alpha = alpha
        self.decay_factor = decay_factor

    def predict(self, x):
        """Makes a prediction on a given input
        :param x: Numpy array to make the prediction on. Shape must be (9, None)
        :type x: np.array
        :returns: Neural Net's prediction
        :rtype: float
        """
        return np.asscalar(forward_prop(self.W, self.b, x))

    def gd_step(self, x, v_target):
        """Makes one step of gradient descent and decreases alpha by decay_factor
        :param x: Numpy array with encountered afterstates. Shape must be (9, None)
        :type x: np.array
        :param v_target: Numpy array with afterstate values obtained my Monte Carlo. Shape must be (None,)
        """
        W_gradients, b_gradients = backprop(self.W, self.b, x, v_target)
        for i in range(self.n_layers):
            self.W[i] = self.W[i] - self.alpha * W_gradients[i]
            self.b[i] = self.b[i] - self.alpha * b_gradients[i]

        if self.decay_factor:
            self.alpha *= self.decay_factor

    def save_weights(self, name, turn):
        """Saves the weights of the model to an external file
        :param name: Name of the file
        :type name: str
        """
        for i in range(len(self.W)):
            df_W = pd.DataFrame(self.W[i])
            df_W.to_csv(f'data/{name}/weights_W{i+1}_p{turn if turn == 1 else 2}.csv')
            df_b = pd.DataFrame(self.b[i])
            df_b.to_csv(f'data/{name}/weights_b{i + 1}_p{turn if turn == 1 else 2}.csv')

    def load_weights(self, name, turn=1):
        """Loads the weights of the model from an external file
        :param name: Name of the file
        :type name: str
        :param name: 1 for player 1 and -1 for player 2
        :type name: int
        """
        turn = 2 if turn == -1 else turn
        for file_name in os.listdir(f'data/{name}/'):
            if 'weights_' in file_name and file_name[-5] == str(turn):
                if 'W' in file_name and int(file_name.replace('weights_W', '')[:-7]) - 1 < len(self.W):
                    self.W[int(file_name.replace('weights_W', '')[:-7]) - 1] = pd.read_csv(f'data/{name}/{file_name}',
                                                                                           index_col=0).to_numpy()
                elif 'b' in file_name and int(file_name.replace('weights_b', '')[:-7]) - 1 < len(self.b):
                    self.b[int(file_name.replace('weights_b', '')[:-7]) - 1] = pd.read_csv(f'data/{name}/{file_name}',
                                                                                           index_col=0).to_numpy()

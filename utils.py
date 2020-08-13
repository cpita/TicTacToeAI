import pandas as pd
from matplotlib import pyplot as plt
import time


def perform_analytics_on_sweep(NAMESPACE, NUM_LAYERS, NEURONS_PER_LAYER, ALPHA, ALPHA_DECAY, EPS_DECAY, EPS_FOR_UPDATE, TOTAL_ITERS):

    for num_layers in NUM_LAYERS:
        for neurons_per_layer in NEURONS_PER_LAYER:
            for alpha in ALPHA:
                for alpha_decay_factor in ALPHA_DECAY:
                    for epsilon_decay_factor in EPS_DECAY:
                        for eps_for_update in EPS_FOR_UPDATE:
                            for total_iters in TOTAL_ITERS:
                                name = f'{NAMESPACE}_l{num_layers}_n{neurons_per_layer}_a{alpha}_d{alpha_decay_factor}_ed{epsilon_decay_factor}_e{eps_for_update}_i{total_iters}'
                                df = pd.read_csv(f'data/{name}/win_ratio_{name}.csv', index_col=0)
                                plt.plot(df.index, df['Win Ratio 1'], label=name)
            plt.xlabel('Episodes')
            plt.ylabel('Win Rate Player 1')
            plt.legend()
            plt.savefig(f'data/{NAMESPACE}_l{num_layers}_n{neurons_per_layer}.png')
            plt.clf()

    print('Saved analytics')


def perform_analytics(name):

    df = pd.read_csv(f'data/{name}/win_ratio_{name}.csv', index_col=0)
    plt.plot(df.index, df['Win Ratio 1'], label=name)
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate Player 1')
    plt.legend()
    plt.savefig(f'data/analytics_{name}.png')
    plt.show()

class WinRatioMetrics:

    def __init__(self):
        self.p1_wins = 0
        self.p2_wins = 0
        self.ties = 0
        self.win_ratio = pd.DataFrame()

    def register_win(self, winner):
        if winner == 1:
            self.p1_wins += 1
        elif winner == -1:
            self.p2_wins += 1
        else:
            self.ties += 1

    def get_results(self):
        return self.p1_wins, self.p2_wins, self.ties

    def print_results(self):
        print('------------WIN RATIO METRICS------------')
        print(f'Player 1 has won {self.p1_wins} times')
        print(f'Player 2 has won {self.p2_wins} times')
        print(f'There have been {self.ties} ties')
        print('-----------------------------------------')

    def reset(self):
        self.p1_wins = 0
        self.p2_wins = 0
        self.ties = 0

    def update_win_ratio(self, win_ratio1, win_ratio2, iters):
        self.win_ratio = self.win_ratio.append(pd.DataFrame([[iters, win_ratio1, win_ratio2]], columns=['Episodes', 'Win Ratio 1', 'Win Ratio 2']))


class TimePerformanceMetrics:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def check_timer(self):
        print('---------TIME PERFORMANCE METRICS--------')
        print(f'{time.time() - self.start_time} seconds running')
        print('-----------------------------------------')

    def get_timer(self):
        return time.time() - self.start_time
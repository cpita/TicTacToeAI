from core.models import TicTacToe
from core.policies import NeuralNetPolicy, HumanPolicy

if __name__ == '__main__':

    env = TicTacToe()

    player1 = NeuralNetPolicy(env, [16, 16, 16, 16, 16])
    '''To load the weights you should use the name your model has inside the data folder'''
    player1.model.load_weights('player1', turn=1)
    player2 = HumanPolicy(env)

    player1.turn = 1
    player2.turn = -1
    while True:

        print('Game is starting...')
        while not env.is_terminal():
            env.print_grid()
            print('||||||||||||||||||||||||')
            p1_state = env.current_state
            p1_action = player1.sample(p1_state)
            env.play(p1_action)
            if env.is_terminal():
                break
            env.print_grid()
            print('||||||||||||||||||||||||')
            p2_state = env.current_state
            p2_action = player2.sample(p2_state)
            env.play(p2_action)

        print('Game Over!')
        winner = env.get_reward_player1()
        print('Tie!' if winner == 0.1 else f'{"X" if winner == 1 else "O"} has won!')
        env.print_grid()
        env.reset()

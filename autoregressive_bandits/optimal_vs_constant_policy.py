import os
import sys
import warnings
import json
import numpy as np
from src.agents import AutoregressiveClairvoyant
from src.environment import AutoregressiveEnvironment
from src.core import Core

os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'
warnings.filterwarnings('ignore')

sys.path.append('.')


def populate_dict(algs, dims):
    dct = dict()
    for alg in algs:
        dct[alg] = np.zeros(dims)
    return dct


if __name__ == '__main__':

    testcase_id = 6
    print(f'### Testcase {testcase_id} ###')

    f = open(f'fidelization_bandits/input/testcase_{testcase_id}.json')
    param_dict = json.load(f)

    print(f'Parameters: {param_dict}')

    param_dict['gamma'] = np.array(param_dict['gamma'])

    T = param_dict['T']
    k = param_dict['gamma'].shape[1] - 1
    n_arms = param_dict['gamma'].shape[0]

    noise_std_vect = [0, 0.1, 0.5, 1.0, 2.0]

    print('Ignoring provided noise_std')
    print('Simulation is performed using noise_std in: ' + str(noise_std_vect))

    clrv = 'Clairvoyant'
    clrvconstant = 'Clairvoyant Constant'
    algs = [clrv, clrvconstant]
    total_rewards_mean = populate_dict(algs, (len(noise_std_vect)))
    total_rewards_std = populate_dict(algs, (len(noise_std_vect)))
    sqrtn = param_dict['n_epochs']

    for noise_i in range(len(noise_std_vect)):

        noise = noise_std_vect[noise_i]

        env = AutoregressiveEnvironment(
            n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=noise,
            X0=param_dict['X0'], random_state=param_dict['seed']
        )
        agent = AutoregressiveClairvoyant(
            n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k
        )
        core = Core(env, agent)
        clairvoyant_logs = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=T
        )[:, :-len(param_dict['X0'])]

        env = AutoregressiveEnvironment(
            n_rounds=T, gamma=param_dict['gamma'], k=k,noise_std=noise,
            X0=param_dict['X0'], random_state=param_dict['seed']
        )
        agent = AutoregressiveClairvoyant(
            n_arms=n_arms, gamma=param_dict['gamma'],
            X0=param_dict['X0'], k=k, constant=True
        )
        core = Core(env, agent)
        clairvoyantconstant_logs = core.simulation(
            n_epochs=param_dict['n_epochs'], n_rounds=T
        )[:, :-len(param_dict['X0'])]

        total_rewards = dict()

        total_rewards[clrv] = np.sum(clairvoyant_logs, axis=1)
        total_rewards[clrvconstant] = np.sum(clairvoyantconstant_logs, axis=1)

        for alg in algs:

            total_rewards[alg] /= T
            total_rewards_mean[alg][noise_i] = np.mean(total_rewards[alg])
            total_rewards_std[alg][noise_i] = np.std(total_rewards[alg]) / sqrtn

    scientific_notation = lambda x: "{:.2e}".format(x)

    print('Policy ', end='')

    for noise in noise_std_vect:
        print('& ' + str(noise) + ' ', end='')

    print(' \\\\')

    for alg in algs:

        print(alg + ' ', end='')

        for noise_i, noise in enumerate(noise_std_vect):

            print(' & {:.4f}'.format(total_rewards_mean[alg][noise_i]), end='')
            print(' (', end='')
            print(scientific_notation(total_rewards_std[alg][noise_i]), end='')

            print(')', end='')

        print(' \\\\')

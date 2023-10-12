import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

if __name__ == '__main__':
    from src.agents import AutoregressiveRidgeAgent, AutoregressiveClairvoyant
    from src.environment import AutoregressiveEnvironment
    from src.core import Core
    import matplotlib.pyplot as plt
    import numpy as np
    import tikzplotlib as tikz
    import warnings
    import json
    import sys

    warnings.filterwarnings('ignore')
    out_folder = 'autoregressive_bandits/output/sigma_analysis2/'
    try:
        os.mkdir(out_folder)
        os.mkdir(out_folder+'png/')
        os.mkdir(out_folder+'tex/')
    except:
        pass

    f = open(f'autoregressive_bandits/input/testcase_sigma_analysis2.json')
    param_dict = json.load(f)

    print(f'Parameters: {param_dict}')

    param_dict['gamma'] = np.array(param_dict['gamma'])

    sigma_values = [0, 1e-3, 1e-2, 1e-1, 1, 15, 30]
    T = param_dict['T']+len(param_dict['X0'])
    n_arms = param_dict['gamma'].shape[0]
    param_dict['k'] = len(param_dict['gamma'][0])-1
    a_hists = {}
    # Clairvoyant
    print('Training Clairvoyant algorithm')
    clrv = 'Clairvoyant'
    env = AutoregressiveEnvironment(
        n_rounds=T, gamma=param_dict['gamma'], k=param_dict['k'], noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
    agent = AutoregressiveClairvoyant(
        n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=param_dict['k'])
    core = Core(env, agent)
    clairvoyant_logs, a_hists['Clairvoyant'] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
    clairvoyant_logs = clairvoyant_logs[:, len(param_dict['X0']):]

    arb_logs = {}
    arb = {}
    regret = {}
    for sigma in sigma_values:
        # ARB
        print(f'Training ARB Algorithm with $\sigma$={sigma}')
        arb[sigma] = f'ARB_{sigma}'
        env = AutoregressiveEnvironment(
            n_rounds=T, gamma=param_dict['gamma'], k=param_dict['k'], noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], param_dict['k'],  m=param_dict['m'],
                                         sigma_=sigma, delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        arb_logs[sigma], a_hists[sigma] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        arb_logs[sigma] = arb_logs[sigma][
            :, len(param_dict['X0']):]
        regret[arb[sigma]] = np.inf * np.ones((param_dict['n_epochs'], T))
        for i in range(param_dict['n_epochs']):
            regret[arb[sigma]][i, :] = clairvoyant_logs[i, :] - arb_logs[sigma][i, :]

    sqrtn = np.sqrt(param_dict['n_epochs'])
    f, ax = plt.subplots(1, figsize=(20, 30))
    x = np.arange(len(param_dict['X0']),T, step=50)
    for sigma in sigma_values:
        ax.plot(x, np.mean(np.cumsum(regret[arb[sigma]].T, axis=0), axis=1)[x],
                label=arb[sigma])
        ax.fill_between(x, np.mean(np.cumsum(regret[arb[sigma]].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[arb[sigma]].T, axis=0), axis=1)[x]/sqrtn,
                        np.mean(np.cumsum(regret[arb[sigma]].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[arb[sigma]].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3)

        ax.set_xlim(left=0)
        ax.set_title('Cumulative Regret')
        ax.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    tikz.save(out_folder+f"tex/testcase_sigma_analysis.tex")
    plt.savefig(out_folder+f"png/testcase_sigma_analysis.png")

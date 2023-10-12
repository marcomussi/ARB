# Autoregressive Bandits
This repository contains the code executed to perform the experimental evaluation presented in the paper [Autoregressive Bandits](https://arxiv.org/abs/2212.06251).
## Setup dependencies
To run the experiments you'll first need to clone the repository in your system and then install the dependencies running
```$ pip install -r requirements.txt```
## Baselines comparison
Section 6.1 of the paper compares AR-UCB to bandit baselines from the literature. The algorithms are compared in different settings: to create a scenario, you'll need to generate a `autoregressive_bandits/input/testcase_{#id}.json` file following the template provided in `autoregressive_bandits/input/testcase_template.json`.
The command 
```$ python autoregressive_bandits/baselines_comparison.py {folder_name} {#testcase_id_1} ... {#testcase_id_n}```
will let you execute simulations using scenarios from 1 to n and store the results in the folder `autoregressive_bandits/output/simulation_{folder_name}/`.
The three scenarios that are presented in the paper are contained in `autoregressive_bandits/input/testcase_6.json`, `autoregressive_bandits/input/testcase_8.json` and `autoregressive_bandits/input/testcase_11.json`, to execute them you'll need to run
```$ python autoregressive_bandits/baselines_comparison.py baselines 6 8 11```.
Pre-computed output is already available `autoregressive_bandits/output/simulation_baselines/`. 
## Effect of stochasticity
Section 6.2 shows that the AR-UCB algorithm relies on the noise contribution to rewards to compute its policy. Experimental evaluation shows that when the environment is noisy, AR-UCB significantly outperforms its deterministic counterpart that plays maximizing the steady-state.
To execute the paper's simulation, run
```$ python autoregressive_bandits/optimal_vs_constant_policy.py```
## Effect of parameter *m* misspecification
Section 6.3 assess the effect of misspecifying the AR-UCB's parameter *m*. Results show that providing AR-UCB with a higher value than the true one slows the learning but allows it to achieve convergence. Instead, providing a lower value may end up in the algorithm's non-convergence.
To execute the paper's simulation, run
```$ python autoregressive_bandits/m_bound_analysis.py```
## Effect of autoregressive order *k* misspecification
Section 6.4 assess the effect of not knowing the autoregressive process order *k*. Results show that providing AR-UCB with a higher value than the true one slows the learning but allows it to achieve convergence. Instead, providing a lower value may end up in the algorithm's non-convergence.
To execute the paper's simulation, run
```$ python autoregressive_bandits/k_analysis.py```
## Experiment with real-world data
Section 6.5 assess the performances of AR-UCB in scenarios computed from real-world data. Using data provided by an e-commerce that partnered with us, we extracted the parameters of the autoregressive processes linking prices (our set of actions) with the sales (reward). For the top four bestselling products, we collected the generated scenarios in `autoregressive_bandits/input/testcase_pricing{1-4}.json`.
To execute the paper's simulation, run
```$ python autoregressive_bandits/baselines_comparison.py baselines pricing0 pricing1 pricing2 pricing3```
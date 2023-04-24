# Hamiltonian Active Learning (HAL)
Code on Hamiltonian Active Learning algorithms accompanying "[Active Learning of Quantum System Hamiltonians yields Query Advantage](https://arxiv.org/abs/2112.14553)")

We consider the problem of learning quantum system Hamiltonians and assess performance of different algorithms in learning the cross-resonance (CR) Hamiltonian.
This work proposes an active learning strategy for Hamiltonian learning based on Fisher information. We call the resulting algorithm HAL-FI.

## Code design

The structure of the code in `hamiltonianlearner` is as follows
* `quantum_system_models` defines models of the CR Hamiltonians in presence and absence of noise
* `quantum_system_oracle` describes how to set up different simulators and experimental datasets that can be queried
* `estimators` defines different estimation methods that can be used by different learners (passive or active)
* `learners` defines different learners such as HAL-FI 
* `utils` contains different scripts and functions that help in running learning experiments

Additionally, the `cases` directory includes the following Jupyter notebooks that demonstrate usage of the code:
* `demo_simulator.ipynb` describes how to define different simulators considering the CR Hamiltonian in the presence and absence of different noise sources such as readout noise, control noise, decoherence, etc.
* `demo_estimation_simulated_data.ipynb` describes how the estimators of linear regression and MLE can be applied to simulated data
* `demo_PL_simulator.ipynb` describes how to run learning experiments on the CR Hamiltonian on a simulator with a passive learner
* `demo_HALFI_simulator.ipynb` describes how to run learning experiments on the CR Hamiltonian on a simulator with the active learner HAL-FI
* `HAL_error_scalings.ipynb` describes how to obtain the expected error scalings for the different learners

Finally, if you want to run larger jobs, there are scripts on learning experiments in `jobs`. Post-processing of the data generated through the numerical experiments is illustrated in `visualize_jobs.ipynb`. Data obtained in our numerical experiments on the simulator and used in generating results for the paper can also be found here.
Please contact the authors for access to the experimental data sets collected from an IBM Quantum device. Some examples of experimental datasets that were collected by the authors are included in `cases/Data`.

### Requirements

To run this package, we recommend installing the requirements specified in [environment.yml](https://github.com/arkopaldutt/HAL/blob/main/environment.yml).
It includes dependencies of mosek, cvxopt, and cvxpy, that are required by HAL-FI.

### Note
The code accompanying our [paper](https://arxiv.org/abs/2112.14553) is divided into two repos of [HAL](https://github.com/arkopaldutt/HAL) and [BayesianHL](https://github.com/arkopaldutt/BayesianHL). 
Please refer the latter package BayesianHL to use HAL-FI with a Bayesian estimator. It has different dependencies and requires an older Python version to use the methods from [Qinfer](https://github.com/QInfer/python-qinfer).

## Citing this repository

To cite this repository please include a reference to our [paper](https://arxiv.org/abs/2112.14553).


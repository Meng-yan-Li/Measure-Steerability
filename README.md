# Measure-Steerability
## Introduction

This repository contains the implementation code for the work:

Mengyan Li, Yanning Jia, Fenzhuo Guo, Haifeng Dong, Sujuan Qin, Fei Gao, Measuring network quantum steerability utilizing artificial neural networks, [arXiv:2502.17084](https://arxiv.org/abs/2502.17084).

---

We utilize artificial neural networks (ANNs) to model the causal information in quantum networks, directly using them to measure network quantum steerability. Additionally, we also utilize semidefinite programming (SDP) to compute the lower and upper bounds of quantum steerability in single-source scenarios.

## Prerequisites

- Python 3.8+
- TensorFlow 2.9.0
- NumPy 1.24+
- Matplotlib 3.5+
- SciPy 1.10+

## Repository Structure

```
.
├── Measure steerability utilizing ANNs/   # Using ANNs to measure the steerability
│   ├── Bilocal steering scenario/
│   │   ├── targets.py           # Generates the target assemblage
│   │   ├── config.py            # Configures parameters
│   │   ├── utils_nn.py          # Utilizes ANNs to model causal information
│   │   ├── train.py             # Main function for execution
│   │   └── Introduction.ipynb   # Introduction to this project
│   ├── Bipartite steering scenario/...
│   ├── Multipartite steering scenarios/
│   │   ├── 3-qubit_1-unt/...
│   │   └── 3-qubit_2-unt/...
│   └── requirements.txt
└── Bounds for steerability/     # Using SDP to calculate the bounds for steerability
    ├── 2_qudit_isotropic_states/
    │   ├── absolute_trace.m     # Computes the trace from Fq to Fp
    │   ├── MUBs.m               # Returns the mutually unbiased bases (MUBs) in d dimension
    │   ├── trace_distance.m     # Calculates the trace distance between two matrices p and q
    │   ├── targets.m            # Generates the target assemblages
    │   ├── lower_bound.m        # Determines the lower bound of bipartite steerability
    │   ├── upper_bound.m        # Determines the upper bound of bipartite steerability
    │   └── main.m               # Main function for execution
    ├── 3_qubit_GHZ_states_1unt/...
    └── 3_qubit_GHZ_states_2unt/...
```

## Acknowledgments

The ANN modeling part of this project refers to the following work ideas:

- Tamás Kriváchy, Yu Cai, Daniel Cavalcanti, Arash Tavakoli, Nicolas Gisin, Nicolas Brunner, A neural network oracle for quantum nonlocality problems in networks, npj Quantum Inf 6, 70 (2020). Open access available at https://doi.org/10.1038/s41534-020-00305-x

- Antoine Girardin, Nicolas Brunner, Tamás Kriváchy, Maximal Randomness Generation from Steering Inequality Violations Using Qudits, Phys. Rev. Research 4, 023238 (2022). Open access available at https://doi.org/10.1103/PhysRevResearch.4.023238



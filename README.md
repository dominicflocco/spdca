# Successive Proximal DCA (SPDCA) for Bilevel Optimization Problems
This repository contains the implementation of a successive proximal difference-of-convex function algorithm (SPDCA) for solving mathematical programs with complementarity constraints. The code in this repository supports ongoing research.



## Getting Started

The current implementation of the model and algorithms is in Python and uses the [Pyomo](http://www.pyomo.org/) optimization modeling language. To construct and solve the models, you will need to install Pyomo and a suitable solver (e.g., Gurobi). The environment can be set up using the provided `environment.yml` file or `requirements.txt` file:

```bash
conda env create -f environment.yml
``` 

or 

```bash
pip install -r requirements.txt
```

## Usage
The repository currently contains scripts to solve the bilevel optimization problems with varying upper-level and lower-level objective functions. It is designed to use the benchmark instances from 
[BOBILib](https://bobilib.org):

> Thürauf, J., Kleinert, T., Ljubić, I., Ralphs, T., & Schmidt, M. (2024). *BOBILib: Bilevel Optimization (Benchmark) Instance Library*. Available at: https://optimization-online.org/?p=27063

The implementation assumes that you have downloaded the BOBILib instances and placed them in a directory named `bobilib` in the root of this repository.

#### Running SPDCA

The implementation of the SPDCA is designed for a general difference-of-convex bilevel optimization problem. To run the SPDCA algorithm on a specific dataset, use the following command:


```bash
python -m scripts.benchmark_qp --stem <stem_name> --methods milp dca-l1-n --outpath './results/text.xlsx' 
```

For more details on the available options, run:
```bash
python -m scripts.benders_solve --help
```

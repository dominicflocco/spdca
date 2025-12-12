
"""Utility functions for SPDCA optimization workflows.
This module provides helpers for:
- File discovery by stem name.
- Matrix property checks (symmetry, positive definite/semidefinite).
- Lipschitz constant estimation for generic and quadratic functions.
- Pyomo subproblem solving with Gurobi.
- Handling Pyomo variables and parameters (stacking, value extraction, assignment).
- Iteration logging, summaries, and result export to DataFrame/CSV/Excel.
- Plotting convergence metrics.
- Model pretty-printing to file.
- Aggregating benchmark and tuning results.
- Gurobi callback for progress logging.
The utilities support workflows around bilevel optimization and DC algorithms,
particularly with Pyomo models and Gurobi solver integration.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from typing import Any, Callable
import logging, contextlib, os, time
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.sparse as sp
import gurobipy as gp

logging.getLogger('pyomo').setLevel(logging.CRITICAL)

def find_files_by_stem(root_dir: str, stem: str):
    """Find files under a directory that match a given stem (filename without extension).
    Args:
        root_dir (str): Root directory to search recursively.
        stem (str): Filename stem to match (with or without '.mps' suffix).
    Returns:
        list[str]: List of matching file paths as strings.
    """
    root = Path(root_dir)
    return [str(f) for f in root.rglob('*') if f.stem == stem or f.stem == stem + '.mps']

def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a dense matrix is positive definite via Cholesky decomposition.
    Args:
        matrix (np.ndarray): Dense square matrix to test.
    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def is_symmetric(matrix: sp.csr_matrix) -> bool:   
    """Check if a sparse CSR matrix is symmetric.
    Args:
        matrix (sp.csr_matrix): Sparse CSR matrix to test.
    Returns:
        bool: True if the matrix is symmetric, False otherwise.
    """ 
    return (matrix - matrix.T).nnz == 0
    
def is_positive_semidefinite(matrix: sp.csr_matrix) -> bool:
    """Check if a sparse CSR matrix is positive semidefinite.
    Uses the smallest algebraic eigenvalue to determine PSD status.
    Args:
        matrix (sp.csr_matrix): Sparse CSR matrix to test (assumed symmetric).
    Returns:
        bool: True if the matrix is PSD (λ_min ≥ -1e-6), False otherwise.
    """
    lam_min = sp.linalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
    return lam_min >= -1e-6
    
def is_in_row_space(vector: np.ndarray, matrix: np.ndarray) -> bool:
    """Check if a vector is in the row space of a matrix.
    Determines membership by comparing ranks of the matrix and the matrix augmented
    with the vector as an additional column.
    Args:
        vector (np.ndarray): Vector to check (1D or column-compatible).
        matrix (np.ndarray): Dense matrix whose row space is considered.
    Returns:
        bool: True if the vector lies in the row space of the matrix.
    Raises:
        False: Raised if the vector is not in the row space (note: unusual behavior).
    """
    if matrix.shape[0] == 0:
        return False
    vector = vector.reshape(-1,1)
    rank_D = np.linalg.matrix_rank(matrix)
    rank_aug = np.linalg.matrix_rank(np.hstack([matrix, vector]))

    if rank_D == rank_aug:
        return True
    else:
        raise False

def estimate_lipschitz(g: Callable, *X: list[np.ndarray], use_st_pt: bool=True) -> float:
    """Estimate the Lipschitz constant of a function g over input vectors.
    Computes a single-sample quotient of output and input differences using random
    perturbations. The estimate is lower-bounded by 1e-6.
    Args:
        g (Callable): Function that maps input arrays to an output array/scalar.
        *X (list[np.ndarray]): One or more input arrays serving as base points.
        use_st_pt (bool): If True, use X vs a random Y; otherwise use X_new vs Y.
    Returns:
        float: Estimated Lipschitz constant (≥ 1e-6).
    """

    Y = [np.abs(np.random.randn(x.shape[0] )) for x in X]
    if use_st_pt:
        lipschitz = np.linalg.norm(g(*X) - g(*Y)) / np.linalg.norm(np.hstack(X) - np.hstack(Y))
    else:
        X_new = [np.abs(np.random.randn(x.shape[0] )) for x in X]
        lipschitz = np.linalg.norm(g(*X_new) - g(*Y)) / np.linalg.norm(np.hstack(X_new) - np.hstack(Y))
    return max(lipschitz, 1e-6)

def estimate_lipschitz_quadratic(Q: np.ndarray) -> float:
    """Estimate the Lipschitz constant of a function g over input vectors.
    Computes a single-sample quotient of output and input differences using random
    perturbations. The estimate is lower-bounded by 1e-6.
    Args:
        g (Callable): Function that maps input arrays to an output array/scalar.
        *X (list[np.ndarray]): One or more input arrays serving as base points.
        use_st_pt (bool): If True, use X vs a random Y; otherwise use X_new vs Y.
    Returns:
        float: Estimated Lipschitz constant (≥ 1e-6).
    """
    if Q.shape[0] == 0:
        return 1.0
    if sp.issparse(Q):
        L = sp.linalg.eigsh(Q, k=1, which="LM", return_eigenvectors=False)[0]
    else:
        L = np.linalg.eigvalsh(Q).max()
    return max(L, 1e-6)

def solve_subproblem(
        subproblem_model: pyo.ConcreteModel, 
        optimizer: Any = None, 
        verbose: bool = True,
        gurobi_options: dict[str, Any] = None
        ) -> Any:
    """Solve a Pyomo subproblem using the provided optimizer (default: Gurobi).
    Args:
        subproblem_model (pyo.ConcreteModel): Pyomo model to solve.
        optimizer (Any, optional): Pyomo SolverFactory or solver instance. Defaults to Gurobi.
        verbose (bool, optional): If True, print solver output (tee). Defaults to True.
        gurobi_options (dict[str, Any], optional): Solver options for Gurobi (unused here).
    Returns:
        Any: Solver results object if successful; None on exception.
    """
    if optimizer is None:
        optimizer = pyo.SolverFactory('gurobi')
    try:
        results = optimizer.solve(
            subproblem_model, 
            tee=verbose
        )
    except BaseException as e:
        print(e)
        return None

    
    return results

def hstack_var_values(*vars: list[Any]) -> np.ndarray: 
    """Horizontally stack values of Pyomo variables (indexed or scalar) into a NumPy array.
    Args:
        *vars (list[Any]): Pyomo Var objects (indexed or scalar).
    Returns:
        np.ndarray: Concatenated array of variable values in index order.
    """
    all_vars = []
    for var in vars:
        if var.is_indexed():
            all_vars.extend([var[i].value for i in var.index_set()])
        else:
            all_vars.append(var.value)
    return np.array(all_vars)

def hstack_vars(*vars: list[Any]) -> list[pyo.Var]:
    """Flatten Pyomo variables (indexed or scalar) into a list of Var components.
    Args:
        *vars (list[Any]): Pyomo Var objects (indexed or scalar).
    Returns:
        list[pyo.Var]: List of Var components maintaining index order.
    """
    all_vars = []
    for var in vars:
        if var.is_indexed():
            all_vars.extend([var[i] for i in var.index_set()])
        else:
            all_vars.append(var)
    return all_vars

def set_values(pyo_param: pyo.Param, pyo_var: pyo.Var) -> pyo.Var:
    """Set a Pyomo Param's values from a Pyomo Var's current values.
    Args:
        pyo_param (pyo.Param): Parameter to assign values to (indexed or scalar).
        pyo_var (pyo.Var): Variable providing values (indexed or scalar).
    Returns:
        pyo.Param: The parameter with updated values.
    Raises:
        ValueError: If the cardinality of the Param and Var differ.
    """
    if len(pyo_var) != len(pyo_param):
        raise ValueError("Length of values does not match the size of the Pyomo variable.")
    
    if pyo_param.is_indexed():
        for i, value in pyo_var.get_values().items():
            pyo_param[i].set_value(value)
    else:
        pyo_param.set_value(pyo_var.value)  # Assuming values is a 1D array with one element

    return pyo_param

def values(pyo_var: pyo.Var) -> np.ndarray:
    """Extract values from a Pyomo Var (indexed or scalar) as a NumPy array.
    Args:
        pyo_var (pyo.Var): Variable to extract values from.
    Returns:
        np.ndarray: Array of variable values in index order.
    """
    all_vars = []
    if pyo_var.is_indexed():
        all_vars.extend([pyo_var[i].value for i in pyo_var.index_set()])
    else:
        all_vars.append(pyo_var.value)

    return np.array(all_vars)

def iter_summary(solver, model, iter, z_diff, st, results) -> dict[str, Any]:
    """Create an iteration summary and log key metrics.
    Args:
        solver: SPDCA solver instance providing f(model) and theta(model).
        model: Pyomo model with attributes like gk and tau_k.
        iter (int): Current iteration count.
        z_diff (float): Norm difference between successive iterates.
        st: Start time (epoch seconds) of the iteration or solve.
        results: Solver results object to extract termination condition.
    Returns:
        dict[str, Any]: Dictionary of iteration metrics for logging and analysis.
    """
    rt = time.time() - st

    status_str = results.solver.termination_condition
    theta = np.abs(pyo.value(solver.theta(model)))
    obj_val = pyo.value(solver.f(model))
    penalty_param = model.gk.value

    logging.info(f"{iter:5d} {z_diff:12.4e} {theta:12.4e} {obj_val:12.4e} {penalty_param:12.4e} {model.tau_k.value:12.4e} {round(time.time() - st):>12d}s {status_str:>12}")

    return {
        'iter': iter,
        'runtime': rt,
        'z_diff': z_diff,
        'penalty': theta, 
        'obj_val': obj_val,
        'penalty_param': penalty_param,
        'status_str': status_str
    }

    
def start_summary(solver, model):
    """Log initial solver and model statistics for SPDCA runs.
    Args:
        solver: SPDCA solver instance with configuration and data attributes.
        model: Pyomo model to be solved.
    """
    logging.info("Starting SPDCA Solve.")
    logging.info(f"Log file: {solver.log_file}\n")


    logging.info(f"Model Statistics:\n")
    logging.info(f"{'Number of Variables:':35}{model.nvariables():>10}")
    logging.info(f"{'  Upper level primal:':35}{solver.data.nr_ul_vars:>10}")
    logging.info(f"{'  Lower level primal:':35}{solver.data.nr_ll_vars:>10}")
    # logging.info(f"{'  Lower level dual:':35}{solver.data.nr_ll_constrs:>10}")
    logging.info(f"{'  Auxiliary:':35}{model.nvariables() - (solver.data.nr_ul_vars + solver.data.nr_ll_vars):>10}")

    logging.info(f"\n")
    logging.info(f"{'Number of Constraints:':35}{model.nconstraints():>10}")
    logging.info(f"{'  Upper level:':35}{solver.data.nr_ul_constrs:>10}")
    logging.info(f"{'  Lower level:':35}{solver.data.nr_ll_constrs:>10}")
    logging.info(f"{'  Bilinear:':35}{solver.data.nr_ll_constrs:>10}")
    logging.info(f"{'  Auxiliary:':35}{model.nconstraints() - (solver.data.nr_ul_constrs + solver.data.nr_ll_constrs + solver.data.nr_ll_constrs):>10}")

    logging.info("\n")
    logging.info("DCA Solver Statistics:")
    logging.info(f"{'  Iteration Limit:':35}{solver.options.max_iters:>10}")
    logging.info(f"{'  Initial penalty (g0):':35}{solver.gk:>10.4e}")
    logging.info(f"{'  Lipschitz of gradient:':35}{solver.L:>10.4e}")
    logging.info(f"{'  Convergence tolerance:':35}{solver.options.conv_tol:>10}")
    logging.info(f"{'  Feasibility tolerance:':35}{solver.options.feas_tol:>10}")
    # logging.info(f"{'  Adaptive Prox:':35}{solver.options.adaptive_prox:>10}")
    logging.info(f"{'  Accelerate:':35}{solver.options.accelerate:>10}")
    if solver.norm is not None:
        logging.info(f"{'  Penalty Norm:':35}{solver.norm:>10}")
    else:
        logging.info(f"{'  Penalty Norm:':35}{'l1':>10}")
    logging.info("\n")

def end_summary(solver, model, iters, z_diff, st, results, solve_time=None)-> None:
    """Log final solver statistics and convergence summary.
    Args:
        solver: SPDCA solver instance with objective and penalty functions.
        model: Pyomo model that was solved.
        iters (int): Total number of iterations performed.
        z_diff (float): Final iterate difference metric.
        st: Start time (epoch seconds) of the overall solve.
        results: Subproblem solver results (may be None on error).
        solve_time (float, optional): Total solver time excluding overhead.
    Returns:
        None: Logs results and returns None.
    """
    rt = time.time() - st

    logging.info(f"Converged in {iters} iterations.")
    logging.info("\n")
    
    if iters == solver.options.max_iters:
        status_str = 'max_iterations'
    else:
        status_str = 'converged'
    logging.info(f"{'Final Statistics':28}")
    logging.info(f"{'  SPDCA status:':28}{status_str:>10}")
    if results is not None:
        logging.info(f"{'  Subproblem status:':28}{str(results.solver.termination_condition):>10}")
    else:
        logging.info(f"{'  Subproblem status:':28}{'error':>10}")
    logging.info(f"{'  Number of iterations:':28}{iters:>10}")
    logging.info(f"{'  Total runtime:':28}{rt:>10.4f}")
    if solve_time is not None:
        logging.info(f"{'  Total solve time:':28}{solve_time:>10.4f}")
        logging.info(f"{'  Total overhead time:':28}{(rt-solve_time):>10.4f}")
    logging.info(f"{'  Final obj. value:':28}{pyo.value(solver.f(model)):>10.4e}")
    logging.info(f"{'  Final penalty value:':28}{pyo.value(solver.theta(model)):>10.4e}")
    logging.info(f"{'  Final iter diff:':28}{z_diff:>10.4e}")

    logging.info(f"\n")
    logging.info("SPDCA Solve Completed.")

    return None

def iter_log_to_df(iter_log: list[dict[str, Any]], outpath: str=None) -> pd.DataFrame:
    """Convert an iteration log (list of dicts) into a DataFrame and optionally save.
    Args:
        iter_log (list[dict[str, Any]]): List of iteration metric dictionaries.
        outpath (str, optional): Path to save the DataFrame as an Excel file.
    Returns:
        pd.DataFrame: DataFrame indexed by 'iter' with the provided metrics.
    """
    if len(iter_log) == 0:
        iter_df = pd.DataFrame()
    else:
        iter_df = pd.DataFrame(
            data=iter_log,
            columns=iter_log[0].keys()
        )
        iter_df.set_index(
            'iter',
            inplace=True
        )
    if outpath is not None:
        with pd.ExcelWriter(outpath, mode='w') as writer:
            iter_df.to_excel(excel_writer=writer)

    return iter_df

def iter_log_to_csv(iter_log: list[dict[str, Any]], outpath: str=None) -> None:
    """Export an iteration log to CSV via DataFrame conversion.
    Args:
        iter_log (list[dict[str, Any]]): List of iteration metric dictionaries.
        outpath (str, optional): Path to save the CSV file.
    Returns:
        pd.DataFrame: DataFrame representation of the iteration log.
    """
    iter_df = iter_log_to_df(iter_log)

    if outpath is not None:
        iter_df.to_csv(outpath)
    return iter_df
def plot_iter_log(iter_df: pd.DataFrame, outpath: str=None):
    """Plot convergence metrics from an iteration log DataFrame.
    Generates a semilog plot of z_diff and penalty over iterations, with optional
    saving to a file.
    Args:
        iter_df (pd.DataFrame): DataFrame containing 'z_diff' and 'penalty' columns.
        outpath (str, optional): Path to save the plot image; displays if None.
    """

    plt.rcParams['text.usetex'] = True


    plt.figure(figsize=(8, 5))
    plt.plot(iter_df.index, 
             iter_df['z_diff'], 
             label=r'$\|z^{k+1} - z^k\|$',
             color='b')
    plt.plot(iter_df.index, 
             iter_df['penalty'], 
             label=r'Penalty $\theta(z^{k})$',
             color='r')  # 'penalty' is your theta column

    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(f'SPDCA Convergence Plot ({Path(outpath).stem})')
    plt.legend()
    plt.yscale('log')
    
    plt.grid(True)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight')
    else:
        plt.show()

def pprint_model(model, outpath):
    """Pretty-print a Pyomo model to a file.
    Args:
        model: Pyomo model to pretty-print.
        outpath: Output file path for the printed model structure.
    """
    with open(outpath, 'w') as f:
        model.pprint(ostream=f)

def benchmark_results_to_df(results: dict[Any, dict[str, Any]], outpath: str=None) -> pd.DataFrame:
    """Aggregate benchmark results into a DataFrame and optionally save to Excel.
    Args:
        results (dict[Any, dict[str, Any]]): Mapping of parser to benchmark metrics.
            Each value contains named benchmark sections with metric dicts. The key
            'solved_model' is excluded from aggregation.
        outpath (str, optional): Path to save the DataFrame as an Excel file.
    Returns:
        pd.DataFrame: DataFrame indexed by instance name with aggregated metrics.
    """
    results_list = []
    for parser, benchmark_results in results.items():
        
        data = {
            "Instance": parser._instance_data.name,
            "Nr. UL Vars": parser._instance_data.nr_ul_vars,
            "Nr. LL Vars": parser._instance_data.nr_ll_vars,
            "Nr. UL Constrs": parser._instance_data.nr_ul_constrs,
            "Nr. LL Constrs": parser._instance_data.nr_ll_constrs,
        }

        for benchmark_name, metrics in benchmark_results.items():
            
            for metric, value in metrics.items():
                if metric != 'solved_model':
                    data.update({
                        f"{benchmark_name}_{metric}": value
                    })

        results_list.append(data)
    result_df = pd.DataFrame(results_list)
    result_df.set_index(
        keys='Instance', 
        inplace=True
    )

    if outpath is not None:
        with pd.ExcelWriter(outpath, mode='w') as writer:
            result_df.to_excel(excel_writer=writer)

    return result_df

def tuning_results_to_df(results: dict[Any, dict[str, Any]], outpath: str=None) -> pd.DataFrame:
    """Aggregate tuning results across solver configurations into a DataFrame.
    Includes instance statistics, Lipschitz constant, objective values for exact and
    relaxed models, penalty for relaxed model, solver options, DCA metrics, and an
    optimality gap relative to the exact objective.
    Args:
        results (dict[Any, dict[str, Any]]): Mapping from (solver, index) to result dicts.
        outpath (str, optional): Path to save the DataFrame as an Excel file.
    Returns:
        pd.DataFrame: DataFrame indexed by instance name with tuning metrics.
    """
    results_list = []

    for (solver, i), tuning_results in results.items():

        data = {
            "Instance": solver.data.name,
            "Tuning Index": i,
            "Nr. UL Vars": solver.data.nr_ul_vars,
            "Nr. LL Vars": solver.data.nr_ll_vars,
            "Nr. UL Constrs": solver.data.nr_ul_constrs,
            "Nr. LL Constrs": solver.data.nr_ll_constrs,
            "Lipschitz Gradient": solver.L,
            "Optimal Obj.": tuning_results['exact']['solved_model'].obj(),
            "Relaxed Obj.": tuning_results['relaxed']['solved_model'].obj(),
            "Relaxed Penalty": pyo.value(solver.penalty(tuning_results['relaxed']['solved_model'])),
            **vars(solver.options),
            **{k: v for k, v in tuning_results['dca'].items() if k != 'solved_model'},
            'optimality_gap': (tuning_results['exact']['solved_model'].obj() - pyo.value(solver.f(tuning_results['dca']['solved_model'])))/np.abs(tuning_results['exact']['solved_model'].obj())
        }
        results_list.append(data)
    result_df = pd.DataFrame(results_list)
    result_df.set_index(
        keys='Instance', 
        inplace=True
    )

    if outpath is not None:
        with pd.ExcelWriter(outpath, mode='w') as writer:
            result_df.to_excel(excel_writer=writer,merge_cells=False)
    return result_df


    
def log_progress_cb(model, solver, where):
    """Gurobi callback for logging MIP progress during solve.
    Captures best objective, best bound, and elapsed time, storing them in the
    solver's model state for external progress tracking.
    Args:
        model: Pyomo or solver-associated model (unused here, provided by Gurobi).
        solver: Solver wrapper containing _solver_model state attributes.
        where: Gurobi callback location code (e.g., gp.GRB.Callback.MIP).
    """
    if where == gp.GRB.Callback.MIP:
        cur_obj = round(solver.cbGet(gp.GRB.Callback.MIP_OBJBST), 6)
        cur_bd = round(solver.cbGet(gp.GRB.Callback.MIP_OBJBND), 6)
        

        if (solver._solver_model._obj is None 
            or solver._solver_model._obj != cur_obj 
            or solver._solver_model._bd != cur_bd):

            solver._solver_model._obj = cur_obj
            solver._solver_model._bd = cur_bd


            cur_time = time.time() - solver._solver_model._start_time
            
            
            solver._solver_model._progress_data.append([cur_time, cur_obj, cur_bd])


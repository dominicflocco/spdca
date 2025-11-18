
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
    root = Path(root_dir)
    return [str(f) for f in root.rglob('*') if f.stem == stem or f.stem == stem + '.mps']

def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def is_symmetric(matrix: sp.csr_matrix) -> bool:    
    return (matrix - matrix.T).nnz == 0
    
def is_positive_semidefinite(matrix: sp.csr_matrix) -> bool:
    lam_min = sp.linalg.eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0]
    return lam_min >= -1e-6
    
def is_in_row_space(vector: np.ndarray, matrix: np.ndarray) -> bool:
    """Check if a vector is in the row space of a matrix."""
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
    #TODO: think about BOUNDS on X

    Y = [np.abs(np.random.randn(x.shape[0] )) for x in X]
    if use_st_pt:
        lipschitz = np.linalg.norm(g(*X) - g(*Y)) / np.linalg.norm(np.hstack(X) - np.hstack(Y))
    else:
        X_new = [np.abs(np.random.randn(x.shape[0] )) for x in X]
        lipschitz = np.linalg.norm(g(*X_new) - g(*Y)) / np.linalg.norm(np.hstack(X_new) - np.hstack(Y))
    return max(lipschitz, 1e-6)

def estimate_lipschitz_quadratic(Q: np.ndarray) -> float:
    """Estimate the Lipschitz constant of the gradient of a quadratic function defined by matrix Q."""
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

    # if results.solver.termination_condition != pyo.TerminationCondition.optimal:
    #     raise ValueError("Subproblem did not solve to optimality.")
    return results

def hstack_var_values(*vars: list[Any]) -> np.ndarray:
    """Convert a list of Pyomo INDEXED variables to a NumPy array."""
    all_vars = []
    for var in vars:
        if var.is_indexed():
            all_vars.extend([var[i].value for i in var.index_set()])
        else:
            all_vars.append(var.value)
    return np.array(all_vars)

def hstack_vars(*vars: list[Any]) -> list[pyo.Var]:
    """Convert a list of Pyomo INDEXED variables to a NumPy array."""
    all_vars = []
    for var in vars:
        if var.is_indexed():
            all_vars.extend([var[i] for i in var.index_set()])
        else:
            all_vars.append(var)
    return all_vars

def set_values(pyo_param: pyo.Param, pyo_var: pyo.Var) -> pyo.Var:
    """Set values of a Pyomo variable from a NumPy array."""
    if len(pyo_var) != len(pyo_param):
        raise ValueError("Length of values does not match the size of the Pyomo variable.")
    
    if pyo_param.is_indexed():
        for i, value in pyo_var.get_values().items():
            pyo_param[i].set_value(value)
    else:
        pyo_param.set_value(pyo_var.value)  # Assuming values is a 1D array with one element

    return pyo_param

def values(pyo_var: pyo.Var) -> np.ndarray:
    all_vars = []
    if pyo_var.is_indexed():
        all_vars.extend([pyo_var[i].value for i in pyo_var.index_set()])
    else:
        all_vars.append(pyo_var.value)

    return np.array(all_vars)

def iter_summary(solver, model, iter, z_diff, st, results) -> dict[int, Any]:
    
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
    """Prints a summary of the solver and model."""
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

    iter_df = iter_log_to_df(iter_log)

    if outpath is not None:
        iter_df.to_csv(outpath)
    return iter_df
def plot_iter_log(iter_df: pd.DataFrame, outpath: str=None):


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
    # if optimal_obj is not None:
    #     plt.plot(iter_df.index,
    #              (optimal_obj - iter_df['obj_val'] )/np.abs(optimal_obj),
    #              label=rf'Optimality Gap',
    #              color='green'
    #             )
        # plt.ylim([-12,0])

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

    with open(outpath, 'w') as f:
        model.pprint(ostream=f)

def benchmark_results_to_df(results: dict[Any, dict[str, Any]], outpath: str=None) -> pd.DataFrame:

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


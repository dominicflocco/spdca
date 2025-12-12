from src.spdca_quadratic import SPDCAQuadraticSolver
from mps_aux_file_pair_parser.mps_aux_parser import MPS_AUX_Parser
from utils.utils import *
import argparse, csv

bobilib_dir = os.path.join(os.getcwd(), 'bobilib')

def solve_exact(
        spdca: SPDCAQuadraticSolver, 
        solver_name: str='gurobi_persistent', 
        milp: bool=False,
        log_file: str=None
    ) -> dict[str, Any]:
    """
    Solve the exact, nonconvex formulation using a MILP solver.

    Args:
        spdca (SPDCALinearSolver): The SPDCA solver instance.
        solver_name (str): Name of the MILP solver to use (default: 'gurobi_persistent').
        milp (bool): Whether to solve as a MILP (default
            False). If True, the model is solved as a MILP.
        log_file (str): Path to the log file for solver output (default: None).
    Returns:
        dict[str, Any]: Dictionary containing the objective value, error, runtime,
            solver status, and the solved model instance.
    Raises:
        ValueError: If the solver name is not recognized.
    """
    
    optimizer = pyo.SolverFactory(solver_name)
    optimizer.options['NonConvex'] = 2
    optimizer.options['TimeLimit'] = 1800
    optimizer.options['Threads'] = 12
    optimizer.options['IntFeasTol'] = 1e-9
    optimizer.options['FeasibilityTol'] = 1e-9
    if log_file is not None:
        optimizer.options['LogFile'] = log_file
        optimizer.options['OutputFlag'] = 1
        # optimizer.options['Callback'] = log_progress_cb
    

    instance = spdca.create_exact_instance(relax=False, milp=milp)
    optimizer.set_instance(instance)
    optimizer.set_callback(log_progress_cb)
    log_data = []

    st = time.time()
    optimizer._solver_model._obj = None
    optimizer._solver_model._bd = None
    optimizer._solver_model._start_time = st
    optimizer._solver_model._progress_data = log_data
    

    res = solve_subproblem(
        subproblem_model=instance,
        optimizer=optimizer,
        verbose=False
    )
    rt = time.time() - st

    obj_val = pyo.value(spdca.f(instance))
    error = pyo.value(spdca.penalty(instance))
    if log_file is not None:
        with open(log_file.replace('.log', '-progress.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['time', 'obj', 'best_bnd'])
            csv_writer.writerows(log_data)
        print(f"Wrote progress log to {log_file.replace('.log', '-progress.csv')}")

    if res is None:
        status = 'error'
    else:
        status = res.solver.termination_condition
    return {
        'obj_val': obj_val,
        'best_bnd': optimizer._solver_model._bd,
        'error': error,
        'runtime': rt,
        'status': status,
        'solved_model': instance
    }


def solve_relaxed(spdca: SPDCAQuadraticSolver, solver_name: str='gurobi', obj_sense=pyo.minimize):
    """Solve the relaxed formulation using a solver without complementarity constraints

    Args:
        spdca (SPDCALinearSolver): The SPDCA solver instance.
        solver_name (str): Name of the solver to use (default: 'gurobi').
        obj_sense (pyo.ObjectiveSense): Objective sense for the relaxed problem (default: pyo.minimize).
    Returns:
        dict[str, Any]: Dictionary containing the objective value, error, runtime,
            solver status, and the solved model instance.
    Raises:
        ValueError: If the solver name is not recognized.
    """
    optimizer = pyo.SolverFactory(solver_name)
    optimizer.options['NonConvex'] = 2

    instance = spdca.create_exact_instance(relax=True, obj_sense=obj_sense)

    st = time.time()

    res = solve_subproblem(
        subproblem_model=instance,
        optimizer=optimizer,
        verbose=False
    )
    rt = time.time() - st

    obj_val = pyo.value(spdca.f(instance))
    error = pyo.value(spdca.penalty(instance))


    return {
        'obj_val': obj_val,
        'error': error,
        'runtime': rt,
        'status': res.solver.termination_condition,
        'solved_model': instance
    }


def solve_dca(
        spdca: SPDCAQuadraticSolver, 
        dca_options: dict[str, Any], 
        solver_name: str='gurobi_persistent',
        norm: str=None, 
        starting_pt: pyo.ConcreteModel=None, 
        opt_cut_value: float=None, 
        output_stem: str="test",
        save_plot: bool=False
    ) -> dict[str, Any]:
    """
    Solve the DCA formulation using a specified solver.

    Args:
        spdca (SPDCALinearSolver): The SPDCA solver instance.
        dca_options (dict[str, Any]): Options for the DCA solver, including:
            - max_iters: Maximum number of iterations.
            - conv_tol: Convergence tolerance.
            - feas_tol: Feasibility tolerance.
            - delta: Initial penalty parameter.
            - delta2: Second penalty parameter.
            - gamma0: Initial gamma value.
            - beta: Beta parameter for the DCA algorithm.
            - verbose: Whether to print progress.
            - use_prox: Whether to use proximal updates.
            - adaptive_prox: Whether to use adaptive proximal updates.
            - accelerate: Whether to use acceleration in the DCA algorithm.
        solver_name (str): Name of the solver to use (default: 'gurobi_persistent').
        norm (str): Norm to use for the DCA problem (default: None, which means no norm).
        starting_pt (pyo.ConcreteModel): Starting point for the DCA algorithm (default: None).
        opt_cut_value (float): Optional cut value for the optimization problem (default: None).
        output_stem (str): Stem for the output files (default: "test").
        save_plot (bool): Whether to save a plot of the iteration log (default: False). 
    Returns:
        dict[str, Any]: Dictionary containing the objective value, error, runtime,
            solver status, number of iterations, initial and final penalty parameters,
            and the solved model instance.
    Raises:
        ValueError: If the solver name is not recognized.
    """
    optimizer = pyo.SolverFactory(solver_name)
    if solver_name in ['gurobi', 'gurobi_persistent']:
        optimizer.options['NonConvex'] = 0
        optimizer.options['ScaleFlag'] = 3

    instance = spdca.create_dca_instance(
        N=np.array([[1,1],[1,-1]]),
        starting_pt=starting_pt,
        reduce_vars=True,
        norm=norm,
        opt_cut_value=opt_cut_value
    )

    solved_model, iter_log = spdca.solve(
        instance=instance,
        optimizer=optimizer,
        **dca_options
    )

    iter_df = iter_log_to_csv(
        iter_log=iter_log,
        outpath=spdca.log_file.replace('.log', "-progress.csv") if spdca.log_file else None
    )
    if save_plot:

        plot_iter_log(
            iter_df=iter_df,
            outpath=os.path.join(os.getcwd(), 'results', 'iter_plots', f'{output_stem}.png')
        )

    obj_val = pyo.value(spdca.f(instance))
    error = pyo.value(spdca.theta(instance))

    rt = iter_df['runtime'].iloc[-1]

    status_str = iter_df['status_str'].iloc[-1]
    iters = len(iter_df)
    if iters == dca_options['max_iters']:
        status_str = 'max_iters'
    elif rt >= dca_options.get('time_limit', 1800):
        status_str = 'time_limit'
    else:
        status_str = 'converged'

    return {
        'obj_val': obj_val,
        'error': error,
        'runtime': rt,
        'status': status_str,
        'iters': len(iter_df),
        'gamma_init': iter_df.loc[0, 'penalty_param'],
        'gamma_final': iter_df.loc[len(iter_df)-1, 'penalty_param'],
        'solved_model': instance
    }

def solve_benchmark_instances(mps_parsers: dict[str,MPS_AUX_Parser], methods: list[str], outpath: str=None, ll_quad: bool=True):
    """
    Solve benchmark instances using the specified methods.

    Args:
        mps_parsers (dict[str, MPS_AUX_Parser]): Dictionary mapping instance names to MPS_AUX_Parser instances.
        methods (list[str]): List of methods to use for benchmarking. Options include:
            - 'relaxed': Solve the relaxed formulation.
            - 'ub': Solve the upper bound formulation.
            - 'milp': Solve the exact MILP formulation.
            - 'dca-linf-n': Solve DCA with linf norm, starting from zero.
            - 'dca-linf-r': Solve DCA with linf norm, starting from relaxed solution.
            - 'dca-linf-e': Solve DCA with linf norm, starting from 1.0.
            - 'dca-l1-n': Solve DCA with l1 norm, starting from zero.
            - 'dca-l1-r': Solve DCA with l1 norm, starting from relaxed solution.
            - 'dca-l1-e': Solve DCA with l1 norm, starting from 1.0.
        outpath (str): Path to save the benchmark results as an Excel file. If None, results are not saved. 
    Returns:
        dict: Dictionary containing the results for each instance and method.
    """

    adaptive_prox = False

    if adaptive_prox: suff = 'adapt_prox'
    else: suff = ''
    
    outpath_root = os.path.dirname(outpath) if outpath else None

    results = {}
    for inst_name, reader in mps_parsers.items():

        results[reader] = {}
        spdca = SPDCAQuadraticSolver(
            instance_data=reader._instance_data,
            name=f"SPDCA Instance {inst_name}",
            log_file=os.path.join(outpath_root, 'spdca_logs', f'{inst_name}_spdca.log') if outpath_root else None,
            init_val=0,
            ll_quad=ll_quad
        )   
        if 'relaxed' in methods:

            relaxed_results = solve_relaxed(spdca)

            results[reader].update({f'relaxed': relaxed_results})
        if 'ub' in methods:
            ub_results = solve_relaxed(spdca, obj_sense=pyo.maximize)
            results[reader].update({f'ub': ub_results})

        # bilinear_results = solve_exact(spdca, milp=False)

        if 'milp' in methods:

            milp_results = solve_exact(
                spdca, 
                milp=True,
                log_file=os.path.join(outpath_root, 'mip_logs', f'{inst_name}_mip.log') if outpath_root else None,
            )
            results[reader].update({'milp': milp_results})
    
        
        if 'dca-linf-n' in methods:
        
            dca_results_linf_n = solve_dca(
                spdca=spdca,
                norm='linfty',
                solver_name='mosek',
                # starting_pt=relaxed_results['solved_model'],
                starting_pt=0.0,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 1,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":True,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                output_stem=f"{spdca.data.name}-dca-linf-N-{suff}"
            )
            results[reader].update({f'dca_linf_N_{suff}': dca_results_linf_n})

        if 'dca-linf-r' in methods:
            dca_results_linf_r = solve_dca(
                spdca=spdca,
                norm='linfty',
                solver_name='mosek',
                starting_pt=relaxed_results['solved_model'],
                # starting_pt=None,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 1,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":False,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                output_stem=f"{spdca.data.name}-dca-linf-R-{suff}"
            )
            results[reader].update({f'dca_linf_R_{suff}': dca_results_linf_r})
        
        if 'dca-linf-e' in methods:
            dca_results_linf_e = solve_dca(
                spdca=spdca,
                norm='linfty',
                solver_name='mosek',
                starting_pt=1.0,
                # starting_pt=None,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 1,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":False,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                output_stem=f"{spdca.data.name}-dca-linf-E-{suff}"
            )
            results[reader].update({f'dca_linf_E_{suff}': dca_results_linf_e})

        if 'dca-l1-n' in methods:
            dca_results_l1_n = solve_dca(
                spdca=spdca,
                norm=None,
                solver_name='gurobi_persistent',
                # starting_pt=relaxed_results['solved_model'],
                starting_pt=0.0,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 10,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":False,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                opt_cut_value=None,
                output_stem=f"{spdca.data.name}-dca-l1-N-{suff}"
            )
            results[reader].update({f'dca_l1_N_{suff}': dca_results_l1_n})

        if 'dca-l1-r' in methods:
            dca_results_l1_r = solve_dca(
                spdca=spdca,
                norm=None,
                solver_name='gurobi_persistent',
                starting_pt=relaxed_results['solved_model'],
                # starting_pt=None,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 1,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":False,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                opt_cut_value=None,
                output_stem=f"{spdca.data.name}-dca-l1-R-{suff}"
            )
            results[reader].update({f'dca_l1_R_{suff}': dca_results_l1_r})

        if 'dca-l1-e' in methods:
            dca_results_l1_e = solve_dca(
                spdca=spdca,
                norm=None,
                solver_name='gurobi_persistent',
                starting_pt=1.0,
                # starting_pt=None,
                dca_options = {
                    "max_iters": 1000,
                    "conv_tol": 1e-4,
                    "feas_tol": 1e-6,
                    "delta": 10, 
                    "delta2": 1,
                    "gamma0": 1,
                    "beta": 0.9,
                    "verbose":False,
                    "use_prox": True,
                    'adaptive_prox':adaptive_prox,
                    'accelerate': False
                },
                opt_cut_value=None,
                output_stem=f"{spdca.data.name}-dca-l1-E-{suff}"
            )
            results[reader].update({f'dca_l1_E_{suff}': dca_results_l1_e})
        
    

        result_df = benchmark_results_to_df(results, outpath)

    return result_df

def batch_benchmark_from_stem(
        mps_stem: str, 
        methods: list[str], 
        outpath: str=None, 
        sign: int=1,
        ll_quad: bool=True
    ):
    """
    Batch benchmark a single instance by its stem name.
    Args:
        mps_stem (str): The stem name of the MPS instance to benchmark.
        methods (list[str]): List of methods to use for benchmarking. Options include:
            - 'relaxed': Solve the relaxed formulation.
            - 'ub': Solve the upper bound formulation.
            - 'milp': Solve the exact MILP formulation.
            - 'dca-linf-n': Solve DCA with linf norm, starting from zero.
            - 'dca-linf-r': Solve DCA with linf norm, starting from relaxed solution.
            - 'dca-linf-e': Solve DCA with linf norm, starting from 1.0.
            - 'dca-l1-n': Solve DCA with l1 norm, starting from zero.
            - 'dca-l1-r': Solve DCA with l1 norm, starting from relaxed solution.
            - 'dca-l1-e': Solve DCA with l1 norm, starting from 1.0.
        outpath (str): Path to save the benchmark results as an Excel file. If None, results are not saved.     
        Returns:
            pd.DataFrame: DataFrame containing the benchmark results for the specified instance.
        Raises:
            ValueError: If the MPS instance files cannot be found or if there are multiple pairs
                of MPS/AUX files with the same stem name.
    """

    files = find_files_by_stem(root_dir=bobilib_dir, stem=mps_stem)

    if len(files) < 2:
        raise ValueError(f"Could not find mps/aux file pair with stem {mps_stem}")
    elif len(files) > 2:
        raise ValueError(f"Found more than one mps/aux file pair with stem {mps_stem}")
    
    mps_file_name = [f for f in files if '.mps' in f]
    aux_file_name = [f for f in files if '.aux' in f]

    reader = MPS_AUX_Parser(
        mps_file_name=mps_file_name[0],
        aux_file_name=aux_file_name[0],
        quadratic=True,
        sign=sign
    )

    reader.read()

    return solve_benchmark_instances(
        mps_parsers={reader._instance_data.name: reader}, 
        outpath=outpath, 
        methods=methods,
        ll_quad=ll_quad
    )


def batch_benchmarks_from_dir(
        root_dirs: list[str], 
        methods: list[str], 
        recurse: bool=False, 
        outpath: str=None,
        sign: int=1,
        hard: bool=False,
        ll_quad: bool=True
    ):
    """
    Batch benchmark instances from multiple directories.

    Args:
        root_dirs (list[str]): List of root directories to search for instances.
        methods (list[str]): List of methods to use for benchmarking. Options include:
            - 'relaxed': Solve the relaxed formulation.
            - 'ub': Solve the upper bound formulation.
            - 'milp': Solve the exact MILP formulation.
            - 'dca-linf-n': Solve DCA with linf norm, starting from zero.
            - 'dca-linf-r': Solve DCA with linf norm, starting from relaxed solution.
            - 'dca-linf-e': Solve DCA with linf norm, starting from 1.0.
            - 'dca-l1-n': Solve DCA with l1 norm, starting from zero.
            - 'dca-l1-r': Solve DCA with l1 norm, starting from relaxed solution.
            - 'dca-l1-e': Solve DCA with l1 norm, starting from 1.0.
        recurse (bool): Whether to recursively search subdirectories (default: False).
        outpath (str): Path to save the benchmark results as an Excel file. If None, results are not saved.
        sign (int): Sign of the quadratic term (1 for convex, 0 for indefinite, -1 for nonconvex).
        hard (bool): Whether to use 'hard' instances (default: False).
        ll_quad (bool): Whether to use low-level quadratic data from MPS file (default
    Returns:
        pd.DataFrame: DataFrame containing the benchmark results for all instances found in the specified directories.
    Raises:
        NotADirectoryError: If any of the provided root directories is not a valid directory.
    """
    parsers = {}
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Provided root_dir '{root_dir}' is not a valid directory.")
        
        parsers_temp = MPS_AUX_Parser.parsers_from_directory(
            root_dir=root_dir, 
            recurse=recurse, 
            quadratic=True,
            sign=sign
        )
        parsers.update(parsers_temp)

    if hard:
        baseline_file = './results/benchmark_results/milp-baseline-master-hard-only.xlsx'
    else:
        baseline_file = './results/benchmark_results/milp-baseline-master.xlsx'
    
    baseline_results = pd.read_excel(baseline_file,index_col=[0,1])
    qp_name = Path(outpath).parent.name if outpath else None
    # prev_results_file = os.path.join(Path(outpath).parent, f"{qp_name}-hard.xlsx") if outpath else None
    prev_results_file = None
    if prev_results_file is None or not os.path.isfile(prev_results_file):
        prev_results = pd.DataFrame()
    else:
        
        prev_results = pd.read_excel(prev_results_file, index_col=0)
    
    
    parsers_new = {}
    already_solved = []
    skipped_instances = []
    for inst_name, parser in parsers.items():
        if inst_name in already_solved:
            print(f"Skipping {inst_name} as it is already solved")
            skipped_instances.append(inst_name)
            continue
        elif inst_name not in baseline_results.index.get_level_values(level=1):
            print(f"Skipping {inst_name} as it not in baseline results.")
            skipped_instances.append(inst_name)
        elif inst_name in prev_results.index:
            print(f"Skipping {inst_name} as it is already in previous results.")
            skipped_instances.append(inst_name)
        elif parser._instance_data.valid is False:
            print(f"Skipping {inst_name} as it failed sanity checks.")
        else:
            parsers_new[inst_name] = parser
        
    print(f"Skipped {len(skipped_instances)} TOTAL.")
    print(f"Left with {len(parsers_new)} total instances, solving these... ")
    return solve_benchmark_instances(mps_parsers=parsers_new, outpath=outpath, methods=methods, ll_quad=ll_quad)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run SPDCA.")
    parser.add_argument("--root_dirs", type=str, nargs='+', help="List of root directories to search for instances.")
    parser.add_argument("--root_dir", type=str, nargs='+', help="Root directory or directories to search for instances.")
    parser.add_argument("--stem", type=str, help="Instance stem name to search for.")
    parser.add_argument("--outpath", type=str, required=True, help="Output path for benchmark results (Excel file).")
    parser.add_argument("--recurse", action="store_true", help="Recursively search subdirectories.")
    parser.add_argument("--indefinite", action="store_true", help="Indefinite QP.")
    parser.add_argument("--ll_linear", action="store_true", help="Use low-level quadratic data from MPS file.")
    parser.add_argument("--hard", action="store_true", help="Use 'hard' instances.")
    parser.add_argument(
        "--methods", 
        type=str, 
        nargs='+', 
        default=['dca-l1-n', 'dca-l1-r', 'dca-linf-n', 'dca-linf-r', 'milp'],
        help="List of methods to benchmark. Options: 'dca-l1-n', 'dca-l1-r', 'dca-linf-n', 'dca-linf-r', 'milp'. Default is all methods."
    )

    
    args = parser.parse_args()

    print(f"Running benchmark with methods: {args.methods}")

    if args.indefinite: sign = 0
    else: sign = 1

    if not args.root_dirs and not args.stem:
        parser.error("You must provide either --root_dirs or --stem.")

    if args.stem:
        batch_benchmark_from_stem(
            mps_stem=args.stem,
            outpath=args.outpath,
            methods=args.methods,
            sign=sign,
            ll_quad=not args.ll_linear
        )

    elif args.root_dirs:

        batch_benchmarks_from_dir(
            root_dirs=args.root_dirs,
            recurse=args.recurse,
            outpath=args.outpath,
            methods=args.methods,
            sign=sign,
            hard=args.hard,
            ll_quad=not args.ll_linear
        )

    
    



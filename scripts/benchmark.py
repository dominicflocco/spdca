from src.spdca import SPDCALinearSolver
from mps_aux_file_pair_parser.mps_aux_parser import MPS_AUX_Parser
from utils.utils import *
import argparse, csv

bobilib_dir = os.path.join(os.getcwd(), 'bobilib')

def solve_exact(
        spdca: SPDCALinearSolver, 
        solver_name: str='gurobi_persistent', 
        milp: bool=False,
        log_file: str=None
    ) -> dict[str, Any]:

    
    optimizer = pyo.SolverFactory(solver_name)
    optimizer.options['NonConvex'] = 2
    optimizer.options['TimeLimit'] = 1800
    optimizer.options['Threads'] = 12
    optimizer.options['IntFeasTol'] = 1e-9
    optimizer.options['FeasibilityTol'] = 1e-9
    if log_file is not None:
        optimizer.options['LogFile'] = log_file
        optimizer.options['OutputFlag'] = 1
    

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
        'error': error,
        'runtime': rt,
        'status': status,
        'solved_model': instance
    }


def solve_relaxed(spdca: SPDCALinearSolver, solver_name: str='gurobi', obj_sense=pyo.minimize):

    optimizer = pyo.SolverFactory(solver_name)
    optimizer.options['NonConvex'] = 0

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
        spdca: SPDCALinearSolver, 
        dca_options: dict[str, Any], 
        solver_name: str='gurobi_persistent',
        norm: str=None, 
        starting_pt: pyo.ConcreteModel=None, 
        opt_cut_value: float=None, 
        output_stem: str="test",
        save_plot: bool=False
    ) -> dict[str, Any]:

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
        outpath=spdca.log_file.replace('.log', f"{output_stem}-progress.csv") if spdca.log_file else None
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

def solve_benchmark_instances(mps_parsers: dict[str,MPS_AUX_Parser], methods: list[str], outpath: str=None):


    adaptive_prox = False

    if adaptive_prox: suff = 'adapt_prox'
    else: suff = ''
    
    outpath_root = os.path.dirname(outpath) if outpath else None

    results = {}
    for inst_name, reader in mps_parsers.items():

        results[reader] = {}
        spdca = SPDCALinearSolver(
            instance_data=reader._instance_data,
            name=f"SPDCA Instance {inst_name}",
            log_file=os.path.join(outpath_root, 'spdca_logs', f'{inst_name}_spdca.log') if outpath_root else None,
            init_val=0
        )

        relaxed_results = solve_relaxed(spdca)
        if 'relaxed' in methods:
            results[reader].update({f'relaxed': relaxed_results})
        if 'ub' in methods:
            ub_results = solve_relaxed(spdca, obj_sense=pyo.maximize)
            results[reader].update({f'ub': ub_results})


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
                    "verbose":False,
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
                    "delta2": 1,
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

def batch_benchmark_from_stem(mps_stem: str, methods: list[str], outpath: str=None):
    # for each mps/aux pair in a directory

    files = find_files_by_stem(root_dir=bobilib_dir, stem=mps_stem)

    if len(files) < 2:
        raise ValueError(f"Could not find mps/aux file pair with stem {mps_stem}")
    elif len(files) > 2:
        raise ValueError(f"Found more than one mps/aux file pair with stem {mps_stem}")
    
    mps_file_name = [f for f in files if '.mps' in f]
    aux_file_name = [f for f in files if '.aux' in f]

    reader = MPS_AUX_Parser(
        mps_file_name=mps_file_name[0],
        aux_file_name=aux_file_name[0]
    )

    reader.read()

    return solve_benchmark_instances(mps_parsers={reader._instance_data.name: reader}, outpath=outpath, methods=methods)


def batch_benchmarks_from_dir(
        root_dirs: list[str], 
        methods: list[str], 
        recurse: bool=False, 
        outpath: str=None,
        sign: int=1,
        ll_quad: bool=True
    ):

    parsers = {}
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"Provided root_dir '{root_dir}' is not a valid directory.")
        
        parsers_temp = MPS_AUX_Parser.parsers_from_directory(
            root_dir=root_dir, 
            recurse=recurse
        )
        parsers.update(parsers_temp)

    baseline_file = './results/benchmark_results/milp-baseline-master.xlsx'
    baseline_results = pd.read_excel(baseline_file,index_col=[0,1])
    prev_results_file = os.path.join(Path(outpath).parent, f"linear-internal.xlsx") if outpath else None
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
    return solve_benchmark_instances(mps_parsers=parsers_new, outpath=outpath, methods=methods)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run SPDCA.")
    parser.add_argument("--root_dirs", type=str, nargs='+', help="List of root directories to search for instances.")
    parser.add_argument("--root_dir", type=str, help="Root directory to search for instances.")
    parser.add_argument("--stem", type=str, help="Instance stem name to search for.")
    parser.add_argument("--outpath", type=str, required=True, help="Output path for benchmark results (Excel file).")
    parser.add_argument("--recurse", action="store_true", help="Recursively search subdirectories.")
    parser.add_argument("--from_list", action="store_true", help="Solve instances in hard-coded list.")
    parser.add_argument("--methods", type=str, nargs='+', default=['dca-l1-n', 'dca-l1-r', 'dca-linf-n', 'dca-linf-r', 'milp'],
                        help="List of methods to benchmark. Options: 'dca-l1-n', 'dca-l1-r', 'dca-linf-n', 'dca-linf-r', 'milp'. Default is all methods.")
    parser.add_argument("--solve_all", action="store_true", help="Only run instances in MILP baseline file",
                        default=False)
    
    args = parser.parse_args()

    print(f"Running benchmark with methods: {args.methods}")

    if not args.root_dirs and not args.stem and not args.from_list:
        parser.error("You must provide either --root_dir or --stem.")

    if args.stem:
        batch_benchmark_from_stem(
            mps_stem=args.stem,
            outpath=args.outpath,
            methods=args.methods
        )

    elif args.root_dirs:
        for rd in args.root_dirs:
            if not os.path.isdir(rd):
                raise NotADirectoryError(f"Provided root_dir '{rd}' is not a valid directory.")

        batch_benchmarks_from_dir(
            root_dirs=args.root_dirs,
            recurse=args.recurse,
            outpath=args.outpath,
            methods=args.methods
        )
    elif args.from_list:
        to_solve = [
            'plusBCPIns_10_7_9',
            'plusBCPIns_10_9_6',
            'plusBCPIns_10_7_7',
            'plusBCPIns_10_7_8',
            'bmilplib_460_5',
            'plusBCPIns_10_7_2',
            'xuLarge600-2',
            'interKP-500-100-2-3',
            'interKP-300-100-1-3',
            'interKP-300-100-1-2',
            'interKP-400-100-2-3',
            'interKP-500-100-1-3',
            'interKP-300-100-2-2',
            'interKP-400-100-1-2',
            'interKP-500-100-1-2',
            'interKP-400-100-1-4',
            'interKP-400-100-2-2',
            'bmilplib_410_2',
            'xuLarge600-1',
        ]
        parsers = {}
        for mps_stem in to_solve:
            files = find_files_by_stem(root_dir=bobilib_dir, stem=mps_stem)

            if len(files) < 2:
                raise ValueError(f"Could not find mps/aux file pair with stem {mps_stem}")
            elif len(files) > 2:
                raise ValueError(f"Found more than one mps/aux file pair with stem {mps_stem}")
            
            mps_file_name = [f for f in files if '.mps' in f]
            aux_file_name = [f for f in files if '.aux' in f]

            reader = MPS_AUX_Parser(
                mps_file_name=mps_file_name[0],
                aux_file_name=aux_file_name[0]
            )

            reader.read()

            parsers[mps_stem] = reader
        for k in parsers.keys(): print(k)
        solve_benchmark_instances(mps_parsers=parsers, outpath=args.outpath, methods=args.methods)




import numpy as np
import pyomo.environ as pyo
import os, time, logging
from typing import Any
from types import SimpleNamespace

from utils.utils import *

DEFAULT_OPTIONS = {
    "verbose": False, 
    "conv_tol": 1e-4,
    "feas_tol": 1e-6,
    "gamma0": 1,
    "delta": 1,
    "prox_param": 1,
    "max_iters": 100000,
    "use_prox": True,
    "accelerate": False,
    "delta2": 100,
    "time_limit": 1800,
    "beta": 1
}
BIG_M = 1e4
GAMMA_MAX = 1e6
TAU_MIN = 1e-6


class SPDCALinearSolver:
    """
    Solver for linear bilevel problems using the SPDCA method.
    """
    def __init__(self, instance_data, name: str = "SPDCA test", log_file: str = None, init_val: float=0.0):
        """
        Attributes:
            instance_data: Instance data containing problem matrices and vectors.
            name: Name of the solver instance.
            log_file: Path to the log file.
            init_val: Initial value for the solver.

        """

        self.name = name
        self.data = instance_data
        self.arrays = instance_data.arrays
        self.init_val = init_val

        self.A = self.arrays['leader_ul_constr_mat']
        self.B = self.arrays['follower_ul_constr_mat']
        self.C = self.arrays['leader_ll_constr_mat']
        self.D = self.arrays['follower_ll_constr_mat']
        self.a = self.arrays['ul_rhs']
        self.b = self.arrays['ll_rhs']
        self.c_u = self.arrays['leader_ul_obj']
        self.d_u = self.arrays['follower_ul_obj']
        self.d_l = self.arrays['follower_ll_obj']
        self.leader_lbs = self.arrays['leader_lbs']
        self.leader_ubs = self.arrays['leader_ubs']
        self.follower_lbs = self.arrays['follower_lbs']
        self.follower_ubs = self.arrays['follower_ubs']
        self.leader_bd_mat = self.arrays['leader_bd_mat']
        self.follower_bd_mat = self.arrays['follower_bd_mat']
        self.leader_bd_vec = self.arrays['leader_bd_vec']
        self.follower_bd_vec = self.arrays['follower_bd_vec']

        self._augment_ll_matrix_data()
        self._dimension_check()
        self._feasibility_check()

        self.use_dca = True

        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = os.path.join(os.getcwd(), 'logs', f"{self.name}.log")
        
        os.environ["GRB_LOGFILE"] = ''


        self.norm = None
        self.N = np.array([[1,1],[1,-1]])
        self._set_transform(self.N)

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.StreamHandler(),  # Console
                logging.FileHandler(self.log_file, mode='w')  # File
            ]
        )
    
    def _set_transform(self, N: np.ndarray):
        """
        Set the linear transformation matrix N and its inverse.

        Args:
            N (np.ndarray): 2x2 transformation matrix.
        """
        self.N = N
        self.N_inv = np.linalg.inv(N)
        self.alpha = N[0,0] * N[1,0]
        self.beta = -N[0,1] * N[1,1]


    def create_exact_instance(self, relax: bool=True, milp: bool=False, obj_sense=pyo.minimize) -> pyo.ConcreteModel:
        """
        Create instance of the bilevel problem with components:
            - primal feasibility
            - dual feasibility
            - upper-level objective
            - complementarity slackness (if relax=False)
        
        If relax=False, this method returns a MINLP model using big-M formulation.
        If relax=True, this method returns a MLP model with continuous variables.

        Args:
            relax (bool): Whether to relax the complementarity slackness constraints.
            milp (bool): Whether to use big-M formulation for complementarity slackness.
            obj_sense: Objective sense (pyo.minimize or pyo.maximize).

        """

        self.reduce_vars = False
        model = pyo.ConcreteModel()

        model = self._initialize_bilevel_variables(model, milp)

        model = self._initialize_bilevel_constraints(model)

        if not relax:
            # add convex bilinear constraint lam * ll_aux = 0
            model = self._initialize_nonconvex_constraints(model, milp)

        model.obj = pyo.Objective(
            expr=self.f(model),
            sense=obj_sense
        )

        return model

    
    def create_dca_instance(
            self, 
            N: np.ndarray = None, 
            norm: str=None, 
            starting_pt: pyo.ConcreteModel=None, 
            reduce_vars: bool=True, 
            opt_cut_value: float=None
        ) -> pyo.ConcreteModel:
        """
        Create SPDCA convex subproblem.

        Args:
            N (np.ndarray): 2x2 transformation matrix.
            norm (str): Norm type for penalty function ('l1', 'linfty', 'l2', or None).
            starting_pt (pyo.ConcreteModel): Starting point for DCA.
            reduce_vars (bool): Whether to use variable reduction.
            opt_cut_value (float): Optimality cut value.
        
        Returns:
            pyo.ConcreteModel: The constructed SPDCA model instance.
        
        """
        if N is not None:
            self._set_transform(N)
        
        self.norm = norm
        self.reduce_vars = reduce_vars
        
        self.name += f"norm = {self.norm}"

        model = pyo.ConcreteModel()

        if isinstance(starting_pt, float):
            
            self.init_val = starting_pt

        model = self._initialize_bilevel_variables(model)

        model = self._initialize_dca_variables(model)
        
        model = self._initialize_dca_parameters(model)

        model = self._initialize_bilevel_constraints(model)

        model = self._initialize_dca_constraints(
            model,
            opt_cut_value=opt_cut_value
        )

        self.dim += len(model.u) + len(model.v)
        

        model.obj = pyo.Objective(
            expr=self.dca_obj(model),
            sense=pyo.minimize
        )

        if not isinstance(starting_pt, float):
            model = self._set_dca_starting_point(model, starting_pt)

        pprint_model(
            model,
            outpath=os.path.join('model_pprints', f'{self.data.name}_dca.txt')
        )

        return model


    
    def _dimension_check(self):
        """
        Checks if the dimensions of the input arrays are consistent.
        """

        # check row dimensions
        assert self.A.shape[0] == self.data.nr_ul_constrs, "A matrix row dimension mismatch"
        assert self.B.shape[0] == self.data.nr_ul_constrs, "B matrix row dimension mismatch"
        assert self.C.shape[0] == self.data.nr_ll_constrs, "C matrix row dimension mismatch"
        assert self.D.shape[0] == self.data.nr_ll_constrs, "D matrix row dimension mismatch"
        assert self.a.shape[0] == self.data.nr_ul_constrs, "a vector dimension mismatch"
        assert self.b.shape[0] == self.data.nr_ll_constrs, "b vector dimension mismatch"
        assert self.c_u.shape[0] == self.data.nr_ul_vars, "c_u vector dimension mismatch"
        assert self.d_u.shape[0] == self.data.nr_ll_vars, "d_u vector dimension mismatch"
        assert self.d_l.shape[0] == self.data.nr_ll_vars, "d_l vector dimension mismatch"

        # check column dimensions
        assert self.A.shape[1] == self.data.nr_ul_vars, "A matrix column dimension mismatch"
        assert self.B.shape[1] == self.data.nr_ll_vars, "B matrix column dimension mismatch"
        assert self.C.shape[1] == self.data.nr_ul_vars, "C matrix column dimension mismatch"
        assert self.D.shape[1] == self.data.nr_ll_vars, "D matrix column dimension mismatch"
       


    def _feasibility_check(self):
        """
        Checks if the instance data is feasible by determining it the lower level
        stationarity constraints are feasible. (i.e., does D^T * lambda = d_l have a solution?)
        """


        if is_in_row_space(self.d_l, self.D.T):
            print("Stationarity condition is consistent")
        else:
            raise ValueError("Stationarity condition is inconsistent (no solution)")

    def _augment_ll_matrix_data(self):
        """
        Augments lower level constraint matrices to incorporate bounds.
        This is not strictly necessary, but helps formulate lower level KKT conditions
        """


        # Augment D matrix with bounds
        if self.follower_bd_mat is not None and self.follower_bd_vec is not None:
            self.D = np.vstack([self.D, self.follower_bd_mat])
            self.b = np.hstack([self.b, self.follower_bd_vec])
            
            # leader coefficients are zeros
            self.C = np.vstack([self.C, np.zeros((self.follower_bd_mat.shape[0], self.data.nr_ul_vars))])

            self.data.nr_ll_constrs += self.follower_bd_mat.shape[0]
        

    
    def _initialize_dca_parameters(self, model):
        """
        Initializes the following DCA parameters in the model:
            - uk: DCA parameter for u variable.
            - vk: DCA parameter for v variable.
            - xk: DCA parameter for x variable.
            - yk: DCA parameter for y variable.
            - gk: penalty parameter.
            - tau_k: proximal parameter.

        Args:
            model: Pyomo model instance.
        Returns:
            model: Pyomo model instance with DCA parameters.
        """

        model.uk = pyo.Param(
            range(self.data.nr_ll_constrs),
            initialize={i: model.u[i].value for i in model.u.index_set()},
            mutable=True
        )
        model.vk = pyo.Param(
            range(self.data.nr_ll_constrs),
            initialize={i: model.v[i].value for i in model.v.index_set()},
            mutable=True
        )

        model.xk = pyo.Param(
            range(self.data.nr_ul_vars),
            initialize={i: model.x[i].value for i in model.x.index_set()},
            mutable=True
        )

        model.yk = pyo.Param(
            range(self.data.nr_ll_vars),
            initialize={i: model.y[i].value for i in model.y.index_set()},
            mutable=True
        )

        #penalty parameters
        model.gk = pyo.Param(
            initialize=1,
            mutable=True
        )
        
        model.tau_k = pyo.Param(
            initialize=1,
            mutable=True
        )

        return model

    def _initialize_nonconvex_constraints(self, model, milp: bool=False):
        """
        Creates the complementarity slackness constraints.

        If milp=True, uses big-M formulation with binary variables.
        If milp=False, uses bilinear equality constraints.

        Args:
            model: Pyomo model instance.
            milp (bool): Whether to use big-M formulation.
        Returns:
            model: Pyomo model instance with complementarity slackness constraints.
        """
        if milp:
            
            def _big_M_ll_aux(m, i):
                return m.ll_aux[i] <= BIG_M * (1 - m.z[i])
            model.big_M_ll_aux = pyo.Constraint(
                range(self.data.nr_ll_constrs), 
                rule=_big_M_ll_aux,
                name='big_M_ll_aux'
            )

            def _big_M_lam(m, i):
                return m.lam[i] <= BIG_M * m.z[i]
            model.big_M_lam = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_big_M_lam
            )

        else:
            def _compl_slackness_constraint_rule(m, i):
                return m.ll_aux[i]*m.lam[i] == 0
            
            model.compl_slackness_constraint = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_compl_slackness_constraint_rule,
                name='compl_slackness_constraint'
            )

        return model

   
    
    def _set_dca_starting_point(self, model: pyo.ConcreteModel, presolve_model: pyo.ConcreteModel) -> pyo.ConcreteModel:
        """
        Sets the starting point for the DCA model based on a presolve model.

        NOTE: This method assumes that the presolve model contains variables
        with the same names as those in the DCA model.

        Args:
            model (pyo.ConcreteModel): DCA model instance.
            presolve_model (pyo.ConcreteModel): Presolve model instance.
        Returns:
            pyo.ConcreteModel: DCA model instance with updated starting point.
        """

        var_names = ['x', 'y', 'lam', 'll_aux']
        for var_name in var_names:
            if hasattr(model, var_name) and hasattr(presolve_model, var_name):
                var = getattr(model, var_name)
                presolve_var = getattr(presolve_model, var_name)
                if var.is_indexed():
                    for idx in var.index_set():
                        if idx in presolve_var:
                            var[idx].set_value(presolve_var[idx].value)
                else:
                    var.set_value(presolve_var.value)

        if hasattr(model, 'u') and hasattr(model, 'v') and hasattr(presolve_model, 'll_aux') and hasattr(presolve_model, 'lam'):
            for i in model.u.index_set():
                ll_aux_val = presolve_model.ll_aux[i].value
                lam_val = presolve_model.lam[i].value
                uv = np.dot(self.N_inv, np.array([lam_val, ll_aux_val]))
                model.u[i].set_value(uv[0])
                model.uk[i].set_value(uv[0])
                model.v[i].set_value(uv[1])
                model.vk[i].set_value(uv[1])

        return model
    



    def _initialize_bilevel_variables(self, model, milp: bool=False):
        """
        Initializes the bilevel variables in the model:
            - x: upper level primal variables.
            - y: lower level primal variables.
            - lam: lower level dual variables (if not reducing vars).
            - ll_aux: lower level auxiliary variables (if not reducing vars).
            - z: binary variables for big-M formulation (if milp=True).

        Args:
            model: Pyomo model instance.
            milp (bool): Whether to use big-M formulation.
        Returns:
            model: Pyomo model instance with initialized variables.
        """

        # upper level primal vars
        model.x = pyo.Var(
            range(self.data.nr_ul_vars),
            bounds={i:(self.leader_lbs[i], self.leader_ubs[i]) 
                        for i in range(self.data.nr_ul_vars)},
            domain=pyo.Reals,
            initialize=self.init_val
        )

        # lower level primal vars
        model.y = pyo.Var(
            range(self.data.nr_ll_vars),
            bounds={i:(self.follower_lbs[i], self.follower_ubs[i]) 
                        for i in range(self.data.nr_ll_vars)},
            domain=pyo.Reals,
            initialize=self.init_val
        )

        if not self.reduce_vars:
            # lower level dual vars
            model.lam = pyo.Var(
                range(self.data.nr_ll_constrs),
                domain=pyo.NonNegativeReals,
                initialize=self.init_val
            )

            # lower level auxiliary variables (ll_aux = C * x + D * y - b)
            model.ll_aux = pyo.Var(
                range(self.data.nr_ll_constrs),
                domain=pyo.Reals,
                initialize=self.init_val
            )

        if milp:
            
            model.z = pyo.Var(
                range(self.data.nr_ll_constrs),
                initialize=0,
                domain=pyo.Binary
            )

        self.dim = len(model.x) + len(model.y)

        return model 

    def _initialize_bilevel_constraints(self, model):
        """
        Initializes the bilevel constraints in the model:
            - upper level feasibility constraints.
            - lower level feasibility constraints.
            - lower level stationarity constraints.
            - lower level auxiliary variable definition.
        
        Args:
            model: Pyomo model instance.   

        Returns:
            model: Pyomo model instance with initialized constraints.
        """

        # upper level feasibility constraints
        def _upper_level_constraint_rule(m, i):
            ul_var_expr = sum(m.x[j]*self.A[i, j] for j in range(self.data.nr_ul_vars))
            ll_var_expr = sum(m.y[j]*self.B[i, j] for j in range(self.data.nr_ll_vars))
            return (ul_var_expr + ll_var_expr) >= self.a[i]
        
        model.ul_constraint = pyo.Constraint(
            range(self.data.nr_ul_constrs),
            rule=_upper_level_constraint_rule,
            name='upper_level_constraint'
        )

        # lower level feasibility constraints
        def _lower_level_constraint_rule(m, i):
            ul_var_expr = sum(m.x[j]*self.C[i, j] for j in range(self.data.nr_ul_vars))
            ll_var_expr = sum(m.y[j]*self.D[i, j] for j in range(self.data.nr_ll_vars))
            return (ul_var_expr + ll_var_expr) >= self.b[i]
        
        model.ll_constraint = pyo.Constraint(
            range(self.data.nr_ll_constrs),
            rule=_lower_level_constraint_rule,
            name='lower_level_constraint'
        )

        
        # lower level stationarity constraints
        def _lower_level_stationarity(m, j):
            if self.reduce_vars:
                return sum(self.D[i, j] * (self.N[0,0] * m.u[i] + self.N[0,1] * m.v[i]) for i in range(self.data.nr_ll_constrs)) - self.d_l[j] == 0
            else:
                return sum(self.D[i, j] * m.lam[i] for i in range(self.data.nr_ll_constrs)) - self.d_l[j] == 0
        model.lower_level_stationarity = pyo.Constraint(
            range(self.data.nr_ll_vars),
            rule=_lower_level_stationarity,
            name='lower_level_stationarity'
        )

        # lower level auxiliary variable definition
        # ll_aux = C x + D y - b
        def _lower_level_aux_constraint_rule(m, i):
            ul_var_expr = sum(m.x[j]*self.C[i, j] for j in range(self.data.nr_ul_vars))
            ll_var_expr = sum(m.y[j]*self.D[i, j] for j in range(self.data.nr_ll_vars))
            if self.reduce_vars:
                return ul_var_expr + ll_var_expr - self.b[i] == self.N[1,0] * m.u[i] + self.N[1,1] * m.v[i]
            else:
                return ul_var_expr + ll_var_expr - self.b[i] == m.ll_aux[i]
        
        model.ll_aux_constraint = pyo.Constraint(
            range(self.data.nr_ll_constrs),
            rule=_lower_level_aux_constraint_rule,
            name='lower_level_aux_constraint'
        )
        
        return model
    
    def _initialize_dca_variables(self, model):
        """
        Initializes the DCA auxiliary variables in the model:

            - u: DCA auxiliary variable u.
            - v: DCA auxiliary variable v.
            - w: l1 norm auxiliary variable (if norm='l1').
            - slack: linfty norm auxiliary variable (if norm='linfty').

        Args:
            model: Pyomo model instance.

        Returns:
            model: Pyomo model instance with initialized DCA variables.
        """
        # dca auxiliary variables
        model.u = pyo.Var(
            range(self.data.nr_ll_constrs),
            domain=pyo.Reals,
            # initialize={i: self.N_inv[0,0]*model.lam[i].value + self.N_inv[0,1]*model.ll_aux[i].value
            #             for i in model.ll_aux.index_set()}
            initialize=self.init_val
        )
        model.v = pyo.Var(
            range(self.data.nr_ll_constrs),
            domain=pyo.Reals,
            # initialize={i: self.N_inv[1,0]*model.lam[i].value + self.N_inv[1,1]*model.ll_aux[i].value
            #             for i in model.lam.index_set()}
            initialize=self.init_val
        )
        if self.norm == 'l1':
            # l1 norm auxiliary variable
            model.w = pyo.Var(
                range(self.data.nr_ll_constrs),
                domain=pyo.NonNegativeReals,
                initialize=self.init_val
            )
        if self.norm == 'linfty':
            model.slack = pyo.Var(
                domain=pyo.NonNegativeReals,
                initialize=1
            )
        

        return model 
    
    def _initialize_dca_constraints(self, model, opt_cut_value: float=None):
        """
        Initializes the DCA constraints in the model based on the selected norm type.

        Args:
            model: Pyomo model instance.
            opt_cut_value (float): Optimality cut value.
        Returns:
            model: Pyomo model instance with initialized DCA constraints.
        """

        if self.norm == 'l1':
            # dca auxiliary variable constraints
            def _l1_penalty_max_rule_u(m, i):
                return m.w[i] >= self.alpha * m.u[i]**2
            model.l1_penalty_max_u = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_l1_penalty_max_rule_u,
                name='l1_penalty_max_u'
            )

            def _l1_penalty_max_rule_v(m, i):
                return m.w[i] >= self.beta * m.v[i]**2
            model.l1_penalty_max_v = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_l1_penalty_max_rule_v,
                name='l1_penalty_max_v'
            )
        
        elif self.norm == 'linfty':
            def _linfty_penalty_rule_u(m, i):
                return (self.alpha * model.u[i]**2 
                        - self.beta * model.vk[i] ** 2 
                        - 2 * self.beta * model.vk[i] * (model.v[i] - model.vk[i]) 
                        <= m.slack)
            model.linfty_penalty_rule_u = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_linfty_penalty_rule_u,
                name='linfty_penalty_rule_u'
            )

            def _linfty_penalty_rule_v(m, i):
                return (self.beta * model.v[i]**2 
                        - self.alpha * model.uk[i] ** 2 
                        - 2 * self.alpha * model.uk[i] * (model.u[i] - model.uk[i]) 
                        <= m.slack)
            model.linfty_penalty_rule_v = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_linfty_penalty_rule_v,
                name='linfty_penalty_rule_v'
            )
        if opt_cut_value is not None:
            model.opt_cut = pyo.Constraint(
                expr=self.f(model) >= opt_cut_value
            )


        if self.reduce_vars:
            def _ll_aux_nonnegativity(m, i):
                return self.N[1,0] * m.u[i] + self.N[1,1] * m.v[i] >= 0
            model.ll_aux_nonnegativity = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_ll_aux_nonnegativity,
                name='ll_aux_nonnegativity'
            )

            def _lam_nonnegativity(m, i):
                return self.N[0,0] * m.u[i] + self.N[0,1] * m.v[i] >= 0
            model.lam_nonnegativity = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_lam_nonnegativity,
                name='lam_nonnegativity'
            )

        else:
            def _lin_transform_ll_aux_rule(m,i):
                return m.ll_aux[i] == self.N[1,0] * m.u[i] + self.N[1,1] * m.v[i]
            model.lin_transform_ll_aux = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_lin_transform_ll_aux_rule,
                name='lin_transform_ll_aux'
            )

            def _lin_transform_lam_rule(m,i):
                return m.lam[i] == self.N[0,0] * m.u[i] + self.N[0,1] * m.v[i]
            model.lin_transform_lam = pyo.Constraint(
                range(self.data.nr_ll_constrs),
                rule=_lin_transform_lam_rule,
                name='lin_transform_lam'
            )
            

        return model

    
    def dca_obj(self, model: pyo.ConcreteModel) -> pyo.Expression:
        """
        SPDCA subproblem objective function. This function is a linear
        approximation of F = G - H at the current DCA parameters.
        """
        return self.f(model) + model.gk * (self.phi(model) - self.psi_linear_approx(model))

    def grad_F(self, x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Implements the gradient of the SPDCA objective function F = f + gk (theta).

        Args:
            x (np.ndarray): Upper level primal variables.
            y (np.ndarray): Lower level primal variables.
            u (np.ndarray): DCA auxiliary variable u.
            v (np.ndarray): DCA auxiliary variable v.
        Returns:
            np.ndarray: Gradient of the SPDCA objective function.
        """
        return (np.hstack([self.grad_f(x, y),np.zeros(u.shape[0] + v.shape[0])]) +
                np.hstack([np.zeros(x.shape[0] + y.shape[0]), self.gk * self.grad_theta(u, v)]))
    

    def f(self, model: pyo.ConcreteModel):
        """
        DC decomposition of upper level objective function f = g - h.
        """
        return self.g(model) - self.h(model)
        # return 0
    
    def grad_f(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of upper level objective function f = g - h.
        """
        return self.grad_g(x, y) - self.grad_h(x, y)
    
    def g(self, model: pyo.ConcreteModel):
        """
        Convex quadratic part of upper level objective function g.
        """
        return sum(self.c_u[i] * model.x[i] for i in range(self.data.nr_ul_vars)) + \
               sum(self.d_u[i] * model.y[i] for i in range(self.data.nr_ll_vars))
    
    def grad_g(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of convex quadratic part of upper level objective function g.

        Args:
            x (np.ndarray): Upper level primal variables.
            y (np.ndarray): Lower level primal variables.
        Returns:
            np.ndarray: Gradient of g at (x, y).
        """
        return np.hstack([[self.c_u[i] for i in range(x.shape[0])],
                          [self.d_u[i] for i in range(y.shape[0])]])
    
    def h(self, model: pyo.ConcreteModel):
        """
        Concave part of upper level objective function h.
        """
        return 0

    def grad_h(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Gradient of concave part of upper level objective function h.
        Args:
            x (np.ndarray): Upper level primal variables.
            y (np.ndarray): Lower level primal variables.
        Returns:
            np.ndarray: Gradient of h at (x, y).
        """
        return np.zeros(x.shape[0] + y.shape[0])

    def phi(self, model: pyo.ConcreteModel):
        """
        Convex part of penalty function phi.
        """
        if self.norm is None:
            return sum(self.alpha * model.u[i]**2 for i in range(self.data.nr_ll_constrs))
        elif self.norm == 'linfty':
            return model.slack
        elif self.norm == 'l1':
            return sum(2 * model.w[i] for i in range(self.data.nr_ll_constrs))
        elif self.norm == 'l2':
            return 2 * sum(
                self.alpha**2 * model.u[i]**4 + self.beta**2 * model.v[i]**4 
                    for i in range(self.data.nr_ll_constrs)
            )

    def grad_phi(self, u: np.ndarray, v: np.ndarray):
        """
        Gradient of convex part of penalty function phi.
        """
        if self.norm is None:
            return np.hstack([[2 * self.alpha * u[i] for i in range(u.shape[0])],
                              np.zeros(v.shape[0])])
        elif self.norm == 'linfty':
            return np.zeros(v.shape[0] + u.shape[0])
        elif self.norm == 'l2':
            return np.hstack([[8 * self.alpha**2 * u[i]**3 for i in range(u.shape[0])],
                              [8 * self.beta**2 * v[i]**3 for i in range(v.shape[0])]])
        else:
            raise ValueError(f"Nonsmooth gradients not supported get!")
        
    def psi(self, model):
        """
        Concave part of penalty function psi.
        """
        if self.norm is None:
            return sum(self.beta * model.v[i]**2 for i in range(self.data.nr_ll_constrs))
        elif self.norm == 'linfty':
            return 0
        elif self.norm == 'l1':
            return sum(self.alpha * model.u[i]**2 + self.beta * model.v[i]**2 for i in range(self.data.nr_ll_constrs))    
        elif self.norm == 'l2':
            return sum( (self.alpha * model.u[i]**2 + self.beta * model.v[i]**2)**2 for i in range(self.data.nr_ll_constrs))

    def grad_psi(self, u: np.ndarray, v: np.ndarray):
        """
        Gradient of concave part of penalty function psi.
        """
        if self.norm is None:
            return np.hstack([
                np.zeros(u.shape[0]),
                [
                    2 * self.beta * v[i] 
                    for i in range(v.shape[0])
                ]
            ])
        elif self.norm == 'linfty':
            return np.zeros(v.shape[0] + u.shape[0])
        elif self.norm == 'l2':
            partial_u = np.array([
                4 * self.alpha**2 * u[i]**3 + 4 * self.alpha * self.beta * v[i]**2 * u[i]
                for i in range(u.shape[0])
            ])
            partial_v = np.array([
                4 * self.beta**2 * v[i]**3 + 4 * self.alpha * self.beta * u[i]**2 * v[i] 
                for i in range(v.shape[0])
            ])
            return np.hstack([partial_u, partial_v])
        else:
            raise ValueError(f"Nonsmooth gradients not supported get!")
        
    def psi_linear_approx(self, model: pyo.ConcreteModel): 
        """
        Linear approximation using the Taylor expansion of the concave part 
        of the penalty function psi at the current DCA parameters.
        """
        if self.norm == 'linfty':
            return 0
        u_expr = sum(
            self.alpha * model.uk[i] * (model.u[i] - model.uk[i])
            for i in range(self.data.nr_ll_constrs)
        )
        v_expr = sum(
            self.beta * model.vk[i] * (model.v[i] - model.vk[i])
            for i in range(self.data.nr_ll_constrs)
        )
        if self.norm is None:
            return 2 * (v_expr)
        elif self.norm == 'l1':
            return 2 * (v_expr + u_expr)
        elif self.norm == 'l2':
            return 4 * sum(
                (self.alpha**2 * model.uk[i]**3 + self.alpha * self.beta * model.uk[i] * model.vk[i]**2) * model.u[i]
                + (self.beta**2 * model.vk[i]**3 + self.alpha * self.beta * model.vk[i] * model.uk[i]**2) * model.v[i]
                + 2 * self.alpha * self.beta * model.uk[i]**2 * model.vk[i]**2
                for i in range(self.data.nr_ll_constrs)
            )

    def h_linear_approx(self, model: pyo.ConcreteModel):
        """
        Linear approximation using the Taylor expansion of the concave part
        of the upper level objective function h at the current DCA parameters.
        """
        return 0
    
    def H_linear_approx(self, model: pyo.ConcreteModel):
        """
        Linear approximation of H = h + psi at the current DCA parameters.
        """
        return self.psi_linear_approx(self, model) + self.h_linear_approx(model)
    
    def G(self, model: pyo.ConcreteModel):
        """
        Convex part of main DCA objective function G = g + phi.
        """
        return self.g(model) + self.phi(model)
    
    def theta(self, model):
        """
        Penalty function as function of u, v
        """
        
        return self.phi(model) - self.psi(model)
    
    def grad_theta(self, u: np.ndarray, v:np.ndarray) -> np.ndarray:
        """
        Gradient of penalty function theta = phi - psi.
        """
        return self.grad_phi(u, v) - self.grad_psi(u, v)
    
    def penalty(self, model):
        """
        True nonconvex bilinear penalty
        """

        return np.linalg.norm(
            [
                model.lam[i].value*model.ll_aux[i].value 
                for i in range(self.data.nr_ll_constrs)
            ],
            ord=1
        )

    def _prox_term(self, model: pyo.ConcreteModel) -> pyo.Expression:
        """
        Proximal term for the SPDCA subproblem || z - z_k ||^2.

        Args:
            model (pyo.ConcreteModel): Pyomo model instance.

        Returns:
            pyo.Expression: Proximal term expression.
        """
        prox = sum(
            (model.u[i] - model.uk[i])**2 + (model.v[i] - model.vk[i])**2
                for i in model.u.index_set()
        )
        prox += sum(
            (model.x[i] - model.xk[i]) for i in model.x.index_set()
        )
        prox += sum(
            (model.y[i] - model.yk[i]) for i in model.y.index_set()
        )
        
        return prox


    def _update_prox_matrix(self, model: pyo.ConcreteModel, Bk: np.ndarray, check_psd: bool=False) -> np.ndarray:
        """
        Updates proximal parameter tau_k based on the current iteration.

        Args:
            model (pyo.ConcreteModel): Pyomo model instance.
            beta (float): Proximal update factor.
        Returns:
            float: Updated proximal parameter tau_k.
        """

        if check_psd:
            
            if is_positive_definite(Bk):
                return Bk
            else:
                raise ValueError("Prox matrix is not positive definite.")
        else:
            
            return Bk   
        
    
    def _update_prox_term(self, model: pyo.ConcreteModel, beta: float=1) -> float:
        """
        Updates proximal parameter tau_k based on the current iteration.

        Args:
            model (pyo.ConcreteModel): Pyomo model instance.
            beta (float): Proximal update factor.
        Returns:
            float: Updated proximal parameter tau_k.
        """
        return max(model.tau_k.value * beta, TAU_MIN)
    

    def _update_penalty(self, model: pyo.ConcreteModel, delta: float, delta2: float, z_diff: np.ndarray, tol: float=1e-6) -> float:
        """
        Updates the penalty parameter gk based on the current iteration.

        Args:
            model (pyo.ConcreteModel): Pyomo model instance.
            delta (float): Multiplicative penalty decrease factor.
            delta2 (float): Penalty increase threshold.
            z_diff (np.ndarray): Difference between current and previous iteration variables.
            tol (float): Feasibility tolerance.

        Returns:
            float: Updated penalty parameter gk.
        """
        
        if z_diff * model.gk.value < delta2 and np.abs(pyo.value(self.theta(model))) > tol:
        # if z_diff * model.gk.value < 1: 
            if self.gk > GAMMA_MAX:
                return self.gk
            else:
                self.gk = model.gk.value * delta
                return self.gk
        else:
            return self.gk
    
    def _fix_variables(self, model: pyo.ConcreteModel, optimizer: Any=None, tol: float=1e-6) -> float:
        """
        Fixes variables u, v (and lam, ll_aux if not reducing vars)
        that satisfy the complementarity slackness condition
        within a given tolerance. This only applies when options.accelerate=True.

        NOTE: This is experimental and does not improve performance in most cases.

        Args:
            model (pyo.ConcreteModel): Pyomo model instance.
            optimizer (Any): Optimizer instance (if persistent).
            tol (float): Tolerance for complementarity condition.
        Returns:
            float: Number of variables fixed.

        """

        num_fixed = 0
        for i in model.u.index_set():
            compl = np.abs(self.alpha * model.u[i].value**2 - self.beta * model.v[i].value**2)
            if (compl < tol) and not model.u[i].fixed:
                num_fixed += 1
                model.u[i].fix(model.u[i].value)
                model.v[i].fix(model.v[i].value)
                if optimizer is not None:
                    optimizer.update_var(model.u[i])
                    optimizer.update_var(model.v[i])
                if not self.reduce_vars:
                    model.lam[i].fix(model.lam[i])
                    model.ll_aux[i].fix(model.ll_aux[i])
                    if optimizer is not None:
                        optimizer.update_var(model.lam[i])
                        optimizer.update_var(model.ll_aux[i])
        
    
    def set_default_params(self, **input_options) -> SimpleNamespace:
        """
        Sets default parameters for the SPDCA solver.

        Args:
            input_options: Input options to override default parameters.
        
        Returns:
            SimpleNamespace: Options namespace with default parameters set.
        """
        options = {}

        for param, value in DEFAULT_OPTIONS.items():
            if param in input_options.keys():
                options[param] = input_options[param]
            else:
                options[param] = value 
        if options['gamma0'] < 1e-4:
            options['gamma0'] = 1e-4

        self.gk = options['gamma0']

        self.options = SimpleNamespace(**options)

        return self.options
    def solve(
            self, 
            instance: pyo.ConcreteModel, 
            optimizer: Any=None,
            **input_options
        ) -> pyo.ConcreteModel:
        """
        Main solve method for the SPDCA algorithm.

        Args:
            instance (pyo.ConcreteModel): Pyomo model instance.
            optimizer (Any): Optimizer instance.
            input_options: Input options to override default parameters.
        Returns:
            pyo.ConcreteModel: Solved Pyomo model instance.
        """

        options = self.set_default_params(**input_options)
        
        
        self.L = 2 * self.gk
       
        instance.gk.value = self.gk
        
        if self.L < 1e-4: self.L = 1 

        if options.use_prox:
            instance.tau_k.value = np.abs(self.L)
        else:
            instance.tau_k.value = 0
        
        zk = hstack_var_values(instance.x, instance.y, instance.u, instance.v)

        prox_term = self._prox_term(model=instance)
        
        instance.obj = pyo.Objective(
            expr=self.dca_obj(instance) + instance.tau_k / 2 * prox_term,
            sense=pyo.minimize
        )

        if optimizer.name == 'gurobi_persistent':
            persistent = True
            optimizer.set_instance(instance)
        else: 
            persistent = False
        start_summary(self, instance)

        logging.info(f"{'Iter':>5} {'z_diff':>12} {'penalty':>12} {'f':>12} {'gk':>12} {'tau':>12} {'status':>20}")
        iter_log = []

        st = time.time()
        k = 0 # iteration counter
        while True:
            

            # solve subproblem
            res = solve_subproblem(
                subproblem_model=instance, 
                optimizer=optimizer,
                verbose=options.verbose
            )

            if res is None:
                break 
            if persistent:
                optimizer.load_vars()

            zkp1 = hstack_var_values(instance.x, instance.y, instance.u, instance.v)
            
            z_diff = np.linalg.norm(zkp1 - zk, ord=1)

            conv_check = (z_diff < self.options.conv_tol)
            feas_check = (pyo.value(self.theta(instance)) < self.options.feas_tol)

            iter_log.append(iter_summary(self, instance, k, z_diff, st, res))
            # check convergence
            if (conv_check and feas_check) or ((time.time() - st) > options.time_limit):
                break
            
            
            
            # penalty update 
            instance.gk.set_value(self._update_penalty(instance, self.options.delta, self.options.delta2, z_diff, self.options.feas_tol))
            
            # prox update
            instance.tau_k.set_value(self._update_prox_term(instance, options.beta))

            # update DCA approximation
            set_values(instance.uk, instance.u)
            set_values(instance.vk, instance.v)
            set_values(instance.xk, instance.x)
            set_values(instance.yk, instance.y)

            if persistent:
                if options.accelerate:
                    self._fix_variables(instance, optimizer)
                optimizer.set_objective(instance.obj)
                
                optimizer.update()
            else:
                if options.accelerate:
                    self._fix_variables(instance, optimizer=None)
            
            zk = zkp1 
            k += 1
            
        end_summary(self, instance, k, z_diff, st, res)

        return instance, iter_log



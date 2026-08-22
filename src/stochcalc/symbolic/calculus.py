import sympy as sp
from typing import List, Union
from .sde import SymbolicSDE

def symbolic_ito(f: Union[sp.Expr, List[sp.Expr]], 
                 state_vars: List[sp.Symbol], 
                 sde: SymbolicSDE) -> SymbolicSDE:
    f_list = [f] if not isinstance(f, (list, tuple, sp.Matrix)) else list(f)
    F = sp.Matrix(f_list)
    out_dims = F.rows
    
    mapping = {state_vars[i]: sde.x[i] for i in range(len(state_vars))}
    F_mapped = F.subs(mapping)
    
    b_bT = sde.b * sde.b.T
    drift_output = []
    diff_output = []
    
    for i in range(out_dims):
        f_i = F_mapped[i]
        
        df_dt = sp.diff(f_i, sde.t)
        grad = sp.Matrix([sp.diff(f_i, x_j) for x_j in sde.x]).T
        hessian = sp.Matrix([[sp.diff(f_i, x_j, x_k) for x_k in sde.x] for x_j in sde.x])
        
        drift_1st = grad * sde.a
        drift_2nd = 0.5 * (hessian * b_bT).trace()
        
        total_drift = sp.simplify(df_dt + drift_1st[0, 0] + drift_2nd)
        drift_output.append(total_drift)
        
        total_diff = sp.simplify(grad * sde.b)
        diff_output.append(list(total_diff))
        
    transformed_vars = [sp.Symbol(f"Y_{i+1}") for i in range(out_dims)]
    return SymbolicSDE(
        state_vars=transformed_vars,
        drift=drift_output,
        diffusion=diff_output,
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )

def infinitesimal_generator(f: sp.Expr, 
                            state_vars: List[sp.Symbol], 
                            sde: SymbolicSDE) -> sp.Expr:
    mapping = {state_vars[i]: sde.x[i] for i in range(len(state_vars))}
    f_mapped = f.subs(mapping)
    
    df_dt = sp.diff(f_mapped, sde.t)
    grad = sp.Matrix([sp.diff(f_mapped, x_j) for x_j in sde.x]).T
    hessian = sp.Matrix([[sp.diff(f_mapped, x_j, x_k) for x_k in sde.x] for x_j in sde.x])
    
    b_bT = sde.b * sde.b.T
    drift_1st = grad * sde.a
    drift_2nd = 0.5 * (hessian * b_bT).trace()
    return sp.simplify(df_dt + drift_1st[0, 0] + drift_2nd)

def quadratic_covariation(sde: SymbolicSDE, idx_i: int, idx_j: int) -> sp.Expr:
    if idx_i >= sde.dims or idx_j >= sde.dims:
        raise IndexError("State indices out of bounds.")
    row_i = sde.b[idx_i, :]
    row_j = sde.b[idx_j, :]
    return sp.simplify(row_i.dot(row_j))

def stochastic_product_rule(sde: SymbolicSDE, idx_i: int, idx_j: int) -> SymbolicSDE:
    if idx_i >= sde.dims or idx_j >= sde.dims:
        raise IndexError("Indices out of bounds.")
    x_i = sde.x[idx_i]
    x_j = sde.x[idx_j]
    product_expr = x_i * x_j
    return symbolic_ito(product_expr, list(sde.x), sde)

def doleans_dade_exponential(sde: SymbolicSDE) -> SymbolicSDE:
    y_vars = [sp.Symbol(f"Y_{i+1}") for i in range(sde.dims)]
    new_drift = []
    new_diffusion = []
    
    for i in range(sde.dims):
        y_i = y_vars[i]
        new_drift.append(sp.simplify(y_i * sde.a[i]))
        new_row = [sp.simplify(y_i * sde.b[i, j]) for j in range(sde.noise_dims)]
        new_diffusion.append(new_row)
        
    return SymbolicSDE(
        state_vars=y_vars,
        drift=new_drift,
        diffusion=new_diffusion,
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )

def girsanov_transform(sde: SymbolicSDE, market_price_of_risk: List[sp.Expr]) -> SymbolicSDE:
    theta = sp.Matrix(market_price_of_risk)
    if theta.rows != sde.noise_dims:
        raise ValueError("Market price of risk vector must match noise dims.")
    new_drift = sp.simplify(sde.a - (sde.b * theta))
    wiener_Q = [sp.Symbol(f"dW^Q_{i+1}") for i in range(sde.noise_dims)]
    return SymbolicSDE(
        state_vars=list(sde.x),
        drift=list(new_drift),
        diffusion=[list(sde.b[i, :]) for i in range(sde.dims)],
        time_var=sde.t,
        wiener_vars=wiener_Q
    )

def stratonovich_to_ito(sde: SymbolicSDE) -> SymbolicSDE:
    new_drift = []
    for i in range(sde.dims):
        correction = 0
        for j in range(sde.noise_dims):
            for k in range(sde.dims):
                correction += sde.b[k, j] * sp.diff(sde.b[i, j], sde.x[k])
        new_drift.append(sp.simplify(sde.a[i] + 0.5 * correction))
        
    return SymbolicSDE(
        state_vars=list(sde.x),
        drift=new_drift,
        diffusion=[list(sde.b[i, :]) for i in range(sde.dims)],
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )

def ito_to_stratonovich(sde: SymbolicSDE) -> SymbolicSDE:
    new_drift = []
    for i in range(sde.dims):
        correction = 0
        for j in range(sde.noise_dims):
            for k in range(sde.dims):
                correction += sde.b[k, j] * sp.diff(sde.b[i, j], sde.x[k])
        new_drift.append(sp.simplify(sde.a[i] - 0.5 * correction))
        
    return SymbolicSDE(
        state_vars=list(sde.x),
        drift=new_drift,
        diffusion=[list(sde.b[i, :]) for i in range(sde.dims)],
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )

def girsanov_density_process(sde: SymbolicSDE, market_price_of_risk: List[sp.Expr]) -> SymbolicSDE:
    theta = list(market_price_of_risk)
    if len(theta) != sde.noise_dims:
        raise ValueError("Market price of risk dimension must match the SDE noise dimension.")
    
    z_var = sp.Symbol('Z')
    new_drift = [sp.Integer(0)]
    new_diffusion = [[sp.simplify(-z_var * theta[j]) for j in range(sde.noise_dims)]]
    
    return SymbolicSDE(
        state_vars=[z_var],
        drift=new_drift,
        diffusion=new_diffusion,
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )
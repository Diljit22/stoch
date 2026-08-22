import sympy as sp
from typing import Optional
from .sde import SymbolicSDE

def fokker_planck_pde(sde: SymbolicSDE, density_symbol: Optional[sp.Symbol] = None) -> sp.Expr:
    p = density_symbol if density_symbol is not None else sp.Function('p')(sde.t, *sde.x)
    lhs = sp.diff(p, sde.t)
    
    drift_term = 0
    for i in range(sde.dims):
        drift_term += sp.diff(sde.a[i] * p, sde.x[i])
    drift_term = -drift_term
    
    b_bT = sde.b * sde.b.T
    diffusion_term = 0
    for i in range(sde.dims):
        for j in range(sde.dims):
            diffusion_term += sp.diff(b_bT[i, j] * p, sde.x[i], sde.x[j])
    diffusion_term = 0.5 * diffusion_term
    
    rhs = sp.simplify(drift_term + diffusion_term)
    return sp.Eq(lhs, rhs)

def feynman_kac_pde(sde: SymbolicSDE, 
                    discount_rate: sp.Expr, 
                    u_function: Optional[sp.Symbol] = None) -> sp.Expr:
    u = u_function if u_function is not None else sp.Function('u')(sde.t, *sde.x)
    du_dt = sp.diff(u, sde.t)
    
    grad = sp.Matrix([sp.diff(u, x_i) for x_i in sde.x]).T
    drift_term = grad * sde.a
    
    hessian = sp.Matrix([[sp.diff(u, x_i, x_j) for x_j in sde.x] for x_i in sde.x])
    b_bT = sde.b * sde.b.T
    diffusion_term = 0.5 * (hessian * b_bT).trace()
    
    discount_term = - discount_rate * u
    lhs = sp.simplify(du_dt + drift_term[0, 0] + diffusion_term + discount_term)
    return sp.Eq(lhs, 0)

def lamperti_transform(sde: SymbolicSDE) -> SymbolicSDE:
    if sde.dims != 1:
        raise ValueError("Lamperti transformation requires a 1D SDE.")
    x_var = sde.x[0]
    a_expr = sde.a[0]
    b_expr = sde.b[0, 0]
    
    if b_expr == 0:
        raise ValueError("Diffusion coefficient cannot be zero.")
        
    F = sp.integrate(1 / b_expr, x_var)
    dF_dt = sp.diff(F, sde.t)
    db_dx = sp.diff(b_expr, x_var)
    
    new_drift = sp.simplify(dF_dt + (a_expr / b_expr) - 0.5 * db_dx)
    y_var = sp.Symbol('Y_1')
    
    transformed_sde = SymbolicSDE(
        state_vars=[y_var],
        drift=[new_drift],
        diffusion=[[sp.Integer(1)]],
        time_var=sde.t,
        wiener_vars=list(sde.dW)
    )
    transformed_sde.transformation_expr = F
    return transformed_sde

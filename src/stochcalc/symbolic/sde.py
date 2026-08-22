import sympy as sp
from typing import List, Optional

class SymbolicSDE:
    def __init__(self, 
                 state_vars: List[sp.Symbol], 
                 drift: List[sp.Expr], 
                 diffusion: List[List[sp.Expr]], 
                 time_var: sp.Symbol = sp.Symbol('t'),
                 wiener_vars: Optional[List[sp.Symbol]] = None):
        self.t = time_var
        self.x = sp.Matrix(state_vars)
        self.dims = len(state_vars)
        self.a = sp.Matrix(drift)
        self.b = sp.Matrix(diffusion)
        
        if self.a.rows != self.dims:
            raise ValueError("Drift vector length must match state variables.")
        if self.b.rows != self.dims:
            raise ValueError("Diffusion matrix rows must match state variables.")
            
        self.noise_dims = self.b.cols
        
        if wiener_vars is not None:
            self.dW = sp.Matrix(wiener_vars)
        else:
            self.dW = sp.Matrix([sp.Symbol(f"dW_{i+1}") for i in range(self.noise_dims)])

    def to_latex(self) -> str:
        equations = []
        for i in range(self.dims):
            dx_i = f"d{sp.latex(self.x[i])}"
            drift_val = self.a[i]
            drift_str = f"\\left({sp.latex(drift_val)}\\right) dt" if drift_val != 0 else ""
            
            diff_terms = []
            for j in range(self.noise_dims):
                coeff = self.b[i, j]
                if coeff != 0:
                    diff_terms.append(f"\\left({sp.latex(coeff)}\\right) {sp.latex(self.dW[j])}")
            
            rhs_parts = []
            if drift_str:
                rhs_parts.append(drift_str)
            rhs_parts.extend(diff_terms)
            
            rhs = " + ".join(rhs_parts) if rhs_parts else "0"
            equations.append(f"{dx_i} = {rhs}")
            
        return " \\\\\n".join(equations)

    def __repr__(self) -> str:
        eqs = []
        for i in range(self.dims):
            diff_str = " + ".join([f"({self.b[i, j]})*{self.dW[j]}" for j in range(self.noise_dims) if self.b[i, j] != 0])
            drift_str = f"({self.a[i]})*dt" if self.a[i] != 0 else ""
            rhs = " + ".join(filter(None, [drift_str, diff_str]))
            eqs.append(f"d{self.x[i]} = {rhs if rhs else '0'}")
        return "\n".join(eqs)
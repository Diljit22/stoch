import pytest
import sympy as sp
from stochcalc import (
    SymbolicSDE, symbolic_ito, infinitesimal_generator, 
    quadratic_covariation, stochastic_product_rule, 
    stratonovich_to_ito, ito_to_stratonovich, girsanov_density_process
)

@pytest.fixture
def geometric_sde():
    x, t = sp.symbols('x t')
    mu, sigma = sp.symbols('mu sigma')
    return SymbolicSDE(state_vars=[x], drift=[mu * x], diffusion=[[sigma * x]], time_var=t)

def test_symbolic_representation(geometric_sde):
    rep = str(geometric_sde)
    assert "dx = (mu*x)*dt + (sigma*x)*dW_1" in rep
    latex_rep = geometric_sde.to_latex()
    assert "left" in latex_rep or "right" in latex_rep

def test_ito_transformation(geometric_sde):
    x = sp.Symbol('x')
    f = sp.log(x)
    transformed = symbolic_ito(f, [x], geometric_sde)
    # d(log(X)) = (mu - 0.5 * sigma^2)*dt + sigma * dW
    assert sp.simplify(transformed.a[0] - (sp.Symbol('mu') - 0.5 * sp.Symbol('sigma')**2)) == 0
    assert transformed.b[0, 0] == sp.Symbol('sigma')

def test_infinitesimal_generator(geometric_sde):
    x = sp.Symbol('x')
    f = x**2
    gen = infinitesimal_generator(f, [x], geometric_sde)
    mu, sigma = sp.symbols('mu sigma')
    expected = 2 * mu * x**2 + sigma**2 * x**2
    assert sp.simplify(gen - expected) == 0

def test_stratonovich_conversions():
    x, t = sp.symbols('x t')
    mu, sigma = sp.symbols('mu sigma')
    # Let's create an Ito Gbm: dx = mu*x dt + sigma*x dW
    ito_gbm = SymbolicSDE(state_vars=[x], drift=[mu * x], diffusion=[[sigma * x]], time_var=t)
    
    # Convert Ito to Stratonovich: a_strato = a_ito - 0.5 * b * db_dx = mu*x - 0.5 * sigma*x * sigma = (mu - 0.5*sigma^2)*x
    strato_sde = ito_to_stratonovich(ito_gbm)
    assert sp.simplify(strato_sde.a[0] - (mu - 0.5 * sigma**2) * x) == 0
    
    # Converting it back should yield the original drift mu*x
    back_to_ito = stratonovich_to_ito(strato_sde)
    assert sp.simplify(back_to_ito.a[0] - mu * x) == 0

def test_girsanov_density(geometric_sde):
    theta = sp.Symbol('theta')
    density_sde = girsanov_density_process(geometric_sde, [theta])
    # dZ = -Z * theta * dW
    Z = sp.Symbol('Z')
    assert density_sde.a[0] == 0
    assert density_sde.b[0, 0] == -Z * theta
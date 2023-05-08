import numpy as np

# Compuation methods for pastic deformation point
DISLOCATION_MOVEMENT_METHOD = "disloc_movement"
DISLOCATION_SOURCE_ACTIVATION_METHOD = "disloc_source_activation"
DISLOCATION_REACTION_METHOD = "dislocation_reaction"

# Crystal Orientation variants
CRYSTAL_ORIENTATION_100 = "e100"
CRYSTAL_ORIENTATION_110 = "e110"
CRYSTAL_ORIENTATION_111 = "e111"
CRYSTAL_ORIENTATION_123 = "e123"

SCHMID_FACTOR = {
    "e100": 0.41,
    "e110": 0.41,
    "e111": 0.27,
    "e123": 0.47,
}


def estimate_stress_plastic_deformation_point(method, E, nu, rho_lomer0, rho_0, rho_S0, burger_vektor, crystal_orientation,
                                       c_fr, rho_tot=None, c_multi=None):
    tau_s = get_tau_s(method, E, nu, rho_lomer0, rho_0, rho_S0, burger_vektor, c_fr, rho_tot, c_multi)
    mmax_s = SCHMID_FACTOR[crystal_orientation]

    sigma_yy = tau_s / mmax_s
    return sigma_yy


def get_tau_s(method, E, nu, rho_lomer0, rho_0, rho_S0, burger_vektor, c_fr=None, rho_tot=None, c_multi=None):
    mu = E / (2 * ( 1 + nu))
    b_s = burger_vektor

    if rho_tot is None:
        rho_tot = calculate_rho_total(rho_0, rho_S0, rho_lomer0)

    l_s = get_l_s(c_fr, rho_tot)
    tau_y_s = get_tau_y_s(rho_lomer0, rho_0, rho_S0, mu, b_s)

    if method == DISLOCATION_MOVEMENT_METHOD:
        tau_s = tau_y_s
    elif method == DISLOCATION_SOURCE_ACTIVATION_METHOD:
        tau_s = tau_y_s + get_tau_s_crit(mu, b_s, l_s)
    elif method == DISLOCATION_REACTION_METHOD:
        tau_s = tau_y_s + c_multi * get_tau_s_crit(mu, b_s, l_s)
    else:
        raise Exception("Method unknown. Please choose one of the provided methods.")
    return tau_s


def euclidic_norm(vektor) -> float:
    squared = [element ** 2 for element in vektor]
    return np.sqrt(np.sum(squared))


def get_tau_y_s(rho_lomer0, rho0, rho_S0, mu, b_s) -> float:
    A_LOMER = 0.326
    A_SELF = 0.3
    A_HIRTH = 0.083
    A_COLL = 0.578
    A_COPLANAR = 0.152
    A_GLISS = 0.661

    tau_y_s = mu * b_s * np.sqrt((2 * A_LOMER) * 0.5 * rho_lomer0 + (1 * A_SELF + 2 * A_COPLANAR + 2 * A_HIRTH + 4 * A_GLISS + 1 * A_COLL) * (rho0 + rho_S0))
    return tau_y_s


def get_l_s(c_fr, rho_tot) -> float:
    return c_fr * (1 / np.sqrt(rho_tot))


def get_tau_s_crit(mu, b_s, l_s) -> float:
    return (mu * b_s) / l_s


def calculate_rho_total(rho0: float, rho_s0: float, rho_lomer0: float) -> float:
    rho0, rho_s0, rho_lomer0 = float(rho0), float(rho_s0), float(rho_lomer0)
    total_density_per_system = rho0 + rho_s0 + 0.5 * rho_lomer0
    return total_density_per_system * 12


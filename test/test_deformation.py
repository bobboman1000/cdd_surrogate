from unittest import TestCase

from fccd.deformation import estimate_stress_plastic_deformation_point, DISLOCATION_MOVEMENT_METHOD, \
    DISLOCATION_REACTION_METHOD, DISLOCATION_SOURCE_ACTIVATION_METHOD, get_tau_y_s, get_tau_s_crit, get_l_s, \
    calculate_rho_total

import numpy as np


class TestDeformation(TestCase):

    def test_dislocation_movement_method(self):
        psd = estimate_stress_plastic_deformation_point(
            method=DISLOCATION_MOVEMENT_METHOD,
            E=117,
            nu=0.35,
            rho_0=1,
            rho_lomer0=0,
            rho_S0=0,
            burger_vektor=0.000254,
            crystal_orientation="e111",
            c_fr=1.2,
            c_multi=0.1
        )
        self.assertTrue(np.round(psd * 1000) == 81)

    def test_dislocation_reaction_method(self):
        psd = estimate_stress_plastic_deformation_point(
            method=DISLOCATION_REACTION_METHOD,
            E=117,
            nu=0.35,
            rho_0=1,
            rho_lomer0=0,
            rho_S0=0,
            burger_vektor=0.000254,
            crystal_orientation="e111",
            c_fr=1.2,
            c_multi=0.1
        )
        self.assertTrue(np.round(psd * 1000) == 93)

    def test_source_activation(self):
        psd = estimate_stress_plastic_deformation_point(
            method=DISLOCATION_SOURCE_ACTIVATION_METHOD,
            E=117,
            nu=0.35,
            rho_0=1,
            rho_lomer0=0,
            rho_S0=0,
            burger_vektor=0.000254,
            crystal_orientation="e111",
            c_fr=1.2,
            c_multi=0.1,
        )
        self.assertTrue(np.round(psd * 1000) == 199)

    def test_real_example(self):
        # For plastic_ deformation_point_tensile_Cu_D5_125_100_1e12_0mob100net_Cfr15_Cmult05_Clom0032_Vact300_np4
        E = 117
        nu=0.35
        mu=E / (2 * ( 1 + nu))
        rho0=0.5
        rho_lomer0=0.5
        rho_S0=0.25
        b_s=0.000254
        a = estimate_stress_plastic_deformation_point(DISLOCATION_MOVEMENT_METHOD, E, nu, rho0, rho_S0, rho_lomer0, b_s, "e123", 1.5) * 1000
        b = estimate_stress_plastic_deformation_point(DISLOCATION_SOURCE_ACTIVATION_METHOD, E, nu, rho0, rho_S0, rho_lomer0, b_s, "e123", 1.5, None, 0.5) * 1000
        c = estimate_stress_plastic_deformation_point(DISLOCATION_REACTION_METHOD, E, nu, rho0, rho_S0, rho_lomer0, b_s, "e123", 1.5, None, 0.5) * 1000

        self.assertTrue(np.round(a, 1) == 41.5)
        self.assertTrue(np.round(b, 1) == 95.5)
        self.assertTrue(np.round(c, 1) == 68.5)

    def test_tau_y_s(self):
        E = 117
        nu=0.35
        psd = get_tau_y_s(
            mu=E / (2 * ( 1 + nu)),
            rho0=0.5,
            rho_lomer0=0.5,
            rho_S0=0.25,
            b_s=0.000254,
        ) * 1000
        self.assertTrue(np.round(psd, 1) == 19.6)

    def test_tau_crit_s(self):
        E = 117
        nu = 0.35
        mu = E / (2 * ( 1 + nu))
        rhot_tot = calculate_rho_total(0.5, 0.25, 0.5)
        l_s = get_l_s(1.5, rhot_tot)
        tau_crit = get_tau_s_crit(mu, 0.000254, l_s) * 1000
        self.assertTrue(np.round(tau_crit, 1) == 25.4)



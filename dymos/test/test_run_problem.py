from __future__ import print_function, division, absolute_import

import os
import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_rel_error
import dymos as dm
import numpy as np

from dymos.examples.hyper_sensitive.hyper_sensitive_ode import HyperSensitiveODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt

def hs_problem_radau(tf, num_seg=30):
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=True)
    p.driver.options['optimizer'] = optimizer

    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                               transcription=dm.Radau(num_segments=num_seg, order=3)))
    phase0.set_time_options(fix_initial=True, fix_duration=True)
    phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
    phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
    phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

    phase0.add_boundary_constraint('x', loc='final', equals=1)

    phase0.add_objective('xL', loc='final')

    phase0.set_refine_options(refine=True)

    p.setup(check=True)

    p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
    p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
    p.set_val('traj.phase0.t_initial', 0)
    p.set_val('traj.phase0.t_duration', tf)
    p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4], nodes='control_input'))
    return p


def brachistochrone_problem(num_seg=20):
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['iSumm'] = 6

    traj = p.model.add_subsystem('traj', dm.Trajectory())
    phase0 = traj.add_phase('phase0', dm.Phase(ode_class=BrachistochroneODE,
                                               transcription=dm.Radau(num_segments=num_seg, order=3)))
    phase0.set_time_options(fix_initial=True, fix_duration=False)
    phase0.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                     units=BrachistochroneODE.states['x']['units'],
                     fix_initial=True, fix_final=False, solve_segments=False)
    phase0.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                     units=BrachistochroneODE.states['y']['units'],
                     fix_initial=True, fix_final=False, solve_segments=False)
    phase0.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                     targets=BrachistochroneODE.states['v']['targets'],
                     units=BrachistochroneODE.states['v']['units'],
                     fix_initial=True, fix_final=False, solve_segments=False)
    phase0.add_control('theta', targets=BrachistochroneODE.parameters['theta']['targets'],
                       continuity=True, rate_continuity=True,
                       units='deg', lower=0.01, upper=179.9)
    phase0.add_input_parameter('g', targets=BrachistochroneODE.parameters['g']['targets'],
                               units='m/s**2', val=9.80665)

    phase0.add_boundary_constraint('x', loc='final', equals=10)
    phase0.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase0.add_objective('time_phase', loc='final', scaler=10)

    phase0.set_refine_options(refine=True)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 2.0)

    p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[0, 10], nodes='state_input'))
    p.set_val('traj.phase0.states:y', phase0.interpolate(ys=[10, 5], nodes='state_input'))
    p.set_val('traj.phase0.states:v', phase0.interpolate(ys=[0, 9.9], nodes='state_input'))
    p.set_val('traj.phase0.controls:theta', phase0.interpolate(ys=[5, 100], nodes='control_input'))
    p.set_val('traj.phase0.input_parameters:g', 9.80665)

    return p


class TestRunProblem(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def _assert_solution(self, p, tf):
        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
                   (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.controls:u')[0],
                         ui,
                         tolerance=5e-4)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.controls:u')[-1],
                         uf,
                         tolerance=5e-4)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.states:xL')[-1],
                         J,
                         tolerance=5e-4)

    def test_run_HS_problem_radau(self):
        tf = 100
        p = hs_problem_radau(tf)
        dm.run_problem(p, True)

        self._assert_solution(p, tf)

    def test_run_HS_problem_radau_restart_refine_no_interp(self):
        """
        Test that using a restart file which does not require reinterpolation of the solution
        works with Radau transcription.  We're taking the grid structure from the first
        problem's restart and initializing the second problem with that grid and the
        associated values.
        """
        # first run a problem to generate a 'dymos_solution.db' restart file
        tf = 100
        p1 = hs_problem_radau(tf, num_seg=20)
        dm.run_problem(p1, refine=True, refine_iteration_limit=10)

        self._assert_solution(p1, tf)

        # create a new problem restarting from the last
        p2 = hs_problem_radau(tf)
        dm.run_problem(p2, refine=False, restart='dymos_solution.db')

        self._assert_solution(p2, tf)

    def test_run_brach_problem_radau_restart_no_refine_no_interp(self):
        """
        In this case the grid used in the second problem is the same as that
        defined in the first problem (no refinement is performed).
        """
        # first run a problem to generate a 'dymos_solution.db' restart file
        p1 = brachistochrone_problem()
        dm.run_problem(p1, refine=False)

        # create a new problem restarting from the last
        p2 = brachistochrone_problem()
        dm.run_problem(p2, refine=False, restart='dymos_solution.db')


    def test_run_HS_problem_gl(self):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=True)
        p.driver.options['optimizer'] = optimizer

        traj = p.model.add_subsystem('traj', dm.Trajectory())
        phase0 = traj.add_phase('phase0', dm.Phase(ode_class=HyperSensitiveODE,
                                                   transcription=dm.GaussLobatto(num_segments=30, order=3)))
        phase0.set_time_options(fix_initial=True, fix_duration=True)
        phase0.add_state('x', fix_initial=True, fix_final=False, rate_source='x_dot', targets=['x'])
        phase0.add_state('xL', fix_initial=True, fix_final=False, rate_source='L', targets=['xL'])
        phase0.add_control('u', opt=True, targets=['u'], rate_continuity=False)

        phase0.add_boundary_constraint('x', loc='final', equals=1)

        phase0.add_objective('xL', loc='final')

        phase0.set_refine_options(refine=True)

        p.setup(check=True)

        tf = 100

        p.set_val('traj.phase0.states:x', phase0.interpolate(ys=[1.5, 1], nodes='state_input'))
        p.set_val('traj.phase0.states:xL', phase0.interpolate(ys=[0, 1], nodes='state_input'))
        p.set_val('traj.phase0.t_initial', 0)
        p.set_val('traj.phase0.t_duration', tf)
        p.set_val('traj.phase0.controls:u', phase0.interpolate(ys=[-0.6, 2.4],
                                                               nodes='control_input'))
        dm.run_problem(p, refine=True)

        sqrt_two = np.sqrt(2)
        val = sqrt_two * tf
        c1 = (1.5 * np.exp(-val) - 1) / (np.exp(-val) - np.exp(val))
        c2 = (1 - 1.5 * np.exp(val)) / (np.exp(-val) - np.exp(val))

        ui = c1 * (1 + sqrt_two) + c2 * (1 - sqrt_two)
        uf = c1 * (1 + sqrt_two) * np.exp(val) + c2 * (1 - sqrt_two) * np.exp(-val)
        J = 0.5 * (c1 ** 2 * (1 + sqrt_two) * np.exp(2 * val) + c2 ** 2 * (1 - sqrt_two) * np.exp(-2 * val) -
                   (1 + sqrt_two) * c1 ** 2 - (1 - sqrt_two) * c2 ** 2)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.controls:u')[0],
                         ui,
                         tolerance=1e-2)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.controls:u')[-1],
                         uf,
                         tolerance=1e-2)

        assert_rel_error(self,
                         p.get_val('traj.phase0.timeseries.states:xL')[-1],
                         J,
                         tolerance=5e-4)

    def test_run_brachistochrone_problem(self):
        p = brachistochrone_problem()

        dm.run_problem(p, True)

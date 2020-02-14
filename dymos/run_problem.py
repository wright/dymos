from .grid_refinement.ph_adaptive.ph_adaptive import PHAdaptive
from .phase.phase import Phase

import openmdao.api as om
import dymos as dm
import dymos.run_problem as rp
import os

'''
--- run_problem call sequence when being called from a user script:

python user_script.py
    script sets up problem
    script calls run_problem(), hooked is False
        problem.run_driver()
            final_setup()
            model._clear_iprint()
            driver.run()
        if refine: do grid refinement, loop:
            problem.setup()
            re_interpolate_solution()
            problem.run_driver()
                final_setup()
                model._clear_iprint()
                driver.run()
    script might do post run analysis or plotting

--- run_problem call sequence when being called from Dymos command line:

dymos user_script.py
    command_line.dymos_cmd()
        parse arguments and extract options for run_problem
        modify_enabled = True to enable hook's effect
        register pre hook function for final_setup
        exec script.py
            script sets up problem
            script calls run_problem()
                    problem.run_driver()
                        final_setup(), pre hook function is triggered just before this starts
        hook function hijacks final_setup
            modify_enabled = False to disable hook's effect the next time it is called
            modify_problem() is called with parsed options
                driver.add_recorder
                if restart: load problem configuration from supplied database using load_case
                # if simulate: ?
                # if no_solve: ?
                # if reset_grid: ?
            hook returns
                        final_setup() resumes
                        model._clear_iprint()
                        driver.run()
                    override refine from command line options, if applicable
                    if refine: do grid refinement, loop:
                        problem.setup()
                        re_interpolate_solution()
                        problem.run_driver()
                            final_setup()
                            model._clear_iprint()
                            driver.run()
            script might do post run analysis or plotting
'''

options = {}


def modify_problem(problem, opts):
    rp.options = opts  # save options for potential use in run_problem
    print('modify_problem is starting')  # TODO delete this when we know it is working

    restart = opts.get('restart')
    if restart:  # restore variables from database file specified by 'restart'
        print('Restarting run_problem using the %s database.' % restart)
        cr = om.CaseReader(restart)
        cases = cr.list_cases()
        if len(cases) < 1:
            print('WARNING: the requested %s database file does not have any cases to load.')
        else:
            case = cr.get_case(cases[-1])  # TODO: use last case, ideally it should be the only one, but there are many

            # Initialize the system with values from the case.
            # We unnecessarily call setup again just to make sure we obliterate the previous solution
            # First reset the connections at the top level model until fixed in OpenMDAO
            problem.setup()

            # Load the values from the previous solution
            dm.load_case(problem, case)

    # record variables to database when running driver under hook
    # pre-hook is important, because recording initialization is skipped if final_setup has run once
    save_db = os.getcwd() + '/dymos_solution.db'

    try:
        os.remove(save_db)
    except FileNotFoundError:
        pass  # OK if old database is not present to be deleted

    print('adding recorder at:', save_db)
    problem.driver.add_recorder(om.SqliteRecorder(save_db))
    problem.driver.recording_options['includes'] = ['*']
    problem.driver.recording_options['record_inputs'] = True
    #problem.record_iteration('final')    # TODO: not working to save only last iteration?

    if opts.get('reset_grid'):  # TODO: implement this option
        pass

    print('modify problem is finished')
    return problem


def run_problem(problem, refine=False, refine_iteration_limit=10):
    problem.final_setup()  # make sure command line option hook has a chance to run

    if rp.options.get('no_solve'):
        problem.run_model()
    else:
        problem.run_driver()

    iteration_limit = rp.options.get('refine_iteration_limit')
    if iteration_limit:
        print('overriding run_problem arguments with refine_iteration_limit from command line')
        refine_iteration_limit = iteration_limit
        refine = True

    if refine:
        out_file = 'grid_refinement.out'

        phases = {phase_path: problem.model._get_subsystem(phase_path)
                  for phase_path in find_phases(problem.model)}

        ref = PHAdaptive(phases)
        with open(out_file, 'w+') as f:
            write_initial(f, phases)

            for i in range(refine_iteration_limit):
                need_refine = ref.check_error()
                write_iteration(f, i, phases, ref.error)

                if all(refine_segment is False for refine_segment in need_refine.values()):
                    break

                prev_soln = {'inputs': problem.model.list_inputs(out_stream=None, units=True),
                             'outputs': problem.model.list_outputs(out_stream=None, units=True)}

                refined_phases = ref.refine(need_refine)
                if not refined_phases:
                    break

                problem.setup()

                re_interpolate_solution(problem, phases, previous_solution=prev_soln)

                problem.run_driver()
            if i == refine_iteration_limit-1:
                f.write('\nIteration limit exceeded. Unable to satisfy specified tolerance')
            elif i == 0:
                f.write('\nError is within tolerance. Grid refinement is not required')
            else:
                f.write('\nSuccessfully completed grid refinement')

    if rp.options.get('simulate'):
        for subsys, local in problem.model._all_subsystem_iter():
            subsys.simulate()


def find_phases(sys):
    phase_paths = []
    if isinstance(sys, Phase):
        phase_paths.append(sys.pathname)
    elif isinstance(sys, om.Group):
        for subsys in sys._loc_subsys_map:
            phase_paths.extend(find_phases(getattr(sys, subsys)))
    return phase_paths


def re_interpolate_solution(problem, phases, previous_solution):

    phase_paths = phases.keys()

    prev_ip_dict = {k: v['value'] for k, v in previous_solution['inputs']}
    prev_op_dict = {k: v['value'] for k, v in previous_solution['outputs']}

    abs_to_prom_ip_map = {}
    for prom_name, abs_names in problem.model._var_allprocs_prom2abs_list['input'].items():
        for abs_name in abs_names:
            abs_to_prom_ip_map[abs_name] = prom_name

    abs_to_prom_op_map = {}
    for prom_name, abs_names in problem.model._var_allprocs_prom2abs_list['output'].items():
        for abs_name in abs_names:
            abs_to_prom_op_map[abs_name] = prom_name

    for phase_path, phase in phases.items():
        prom_to_abs_ip_map = phase._var_allprocs_prom2abs_list['input']
        prom_to_abs_op_map = phase._var_allprocs_prom2abs_list['output']

        ti_abs_name = prom_to_abs_op_map['t_initial'][0]
        ti_prom_name = abs_to_prom_op_map[f'{phase_path}.time_extents.t_initial']
        t_initial = prev_op_dict[ti_abs_name]

        td_abs_name = prom_to_abs_op_map['t_duration'][0]
        td_prom_name = abs_to_prom_op_map[f'{phase_path}.time_extents.t_duration']
        t_duration = prev_op_dict[td_abs_name]

        prev_time = prev_op_dict[f'{phase_path}.time.time']

        problem.set_val(ti_prom_name, t_initial)
        problem.set_val(td_prom_name, t_duration)

        for state_name, options in phase.state_options.items():
            state_abs_name = f'{phase_path}.indep_states.states:{state_name}'
            prev_state_soln_abs_name = f'{phase_path}.timeseries.states:{state_name}'
            state_prom_name = abs_to_prom_op_map[state_abs_name]
            prev_state_val = prev_op_dict[prev_state_soln_abs_name]
            problem.set_val(state_prom_name,
                            phase.interpolate(xs=prev_time, ys=prev_state_val, nodes='state_input', kind='slinear'))

        for control_name, options in phase.control_options.items():
            control_abs_name = f'{phase_path}.control_group.indep_controls.controls:{control_name}'
            prev_control_soln_abs_name = f'{phase_path}.timeseries.controls:{control_name}'
            control_prom_name = abs_to_prom_op_map[control_abs_name]
            prev_control_val = prev_op_dict[prev_control_soln_abs_name]
            problem.set_val(control_prom_name,
                            phase.interpolate(xs=prev_time, ys=prev_control_val, nodes='control_input', kind='slinear'))


def write_initial(f, phases):
    f.write('======================\n')
    f.write('   Grid Refinement\n')
    f.write('======================\n')
    f.write('ph-refinement\n')
    for phase_path, phase in phases.items():
        f.write('Phase: {}\n'.format(phase_path))
        f.write('Tolerance: {}\n'.format(phase.refine_options['tolerance']))
        f.write('Minimum Order: {}\n'.format(phase.refine_options['min_order']))
        f.write('Maximum Order: {}\n'.format(phase.refine_options['max_order']))


def write_iteration(f, iter_number, phases, error):
    f.write('\n\n')
    f.write('Iteration number: {}\n'.format(iter_number))
    for phase_path, phase in phases.items():
        f.write('Phase: {}\n'.format(phase_path))
        T = phase.options['transcription']
        gd = phase.options['transcription'].grid_data

        f.write('Number of Segments = {}\n'.format(T.options['num_segments']))

        f.write('Segment Ends = [')
        f.write(', '.join(str(round(elem, 4)) for elem in gd.segment_ends))
        f.write(']\n')

        if isinstance(T.options['order'], int):
            f.write('Segment Order = {}\n'.format(T.options['order']))
        else:
            f.write('Segment Order = [')
            f.write(', '.join(str(elem) for elem in T.options['order']))
            f.write(']\n')

        f.write('Error = [')
        f.write(', '.join(str(elem) for elem in error[phase_path]))
        f.write(']\n')

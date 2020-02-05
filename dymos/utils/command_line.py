#from __future__ import print_function

#import openmdao.api as om
import dymos as dm
#import numpy as np
import argparse


def dymos_cmd():
    parser = argparse.ArgumentParser(description='Dymos Command Line Tool')
    parser.add_argument('script', type=str,
                        help='Python script that creates a Dymos problem to run')
    parser.add_argument('-r', '--restart', default=None,
                        help='Provide a database file to continue from a previous run (default: None)')
    parser.add_argument('-g', '--grid', action='store_true',
                        help='Enable grid refinement')
    parser.add_argument('-l', '--limit', default=10,
                        help='Grid refinement iteration limit (default: 10)')
    args = parser.parse_args()

    print('args', args)
    # TODO: how to convert script name to problem object?
    #dm.run_problem(probemObject, refine=args['grid'], refine_iteration_limit=args['limit'], restart=args['restart'])

if __name__ == '__main__':
    dymos_cmd()

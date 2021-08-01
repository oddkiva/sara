#!/usr/bin/env python3
import argparse
import os
import platform
import sys
from os import path
from subprocess import Popen, PIPE


SARA_BUILD_DIR = path.abspath(path.dirname(__file__))
if platform.system() == 'Darwin':
    SARA_BUILD_DIR = path.join(SARA_BUILD_DIR, 'Release')

# The dataset.
DATASET_DIR = path.join(os.environ['HOME'],'Desktop/Datasets/sfm/castle_int')
RECONSTRUCTION_DATA_FILEPATH = path.join(os.environ['HOME'],
                                         'Desktop/Datasets/sfm/castle_int.h5')


def make_base_command_options(dataset_dir, reconstruction_data_h5_filepath):
    return


def run(program, dataset_dir, reconstruction_data_h5_filepath, *args, **kwargs):
    cmd_options = ['--dirpath', dataset_dir,
                   '--out_h5_file', reconstruction_data_h5_filepath]

    cmd_options += ['--{}'.format(arg) for arg in args]
    cmd_options += ['--{} {}'.format(k, v) for k, v in kwargs]

    program_path = path.join(SARA_BUILD_DIR, program)
    if not path.exists(program_path):
        raise ValueError('{} does not exist!'.format(program_path))
    print('Running', program_path, ' '.join(cmd_options))
    cmd = [program_path] + cmd_options
    process = Popen(cmd, stderr=sys.stderr, stdout=sys.stdout)
    output, err = process.communicate()
    exit_code = process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Structure-from-Motion Pipeline')

    parser.add_argument('--detect_sift', dest='detect_sift',
                        action='store_true', help='Detect SIFT keypoints')
    parser.add_argument('--view_sift', dest='view_sift', action='store_true',
                        help='View SIFT keypoints')

    parser.add_argument('--match_keypoints', dest='match_keypoints',
                        action='store_true', help='Match keypoints')

    parser.add_argument('--estimate_fundamental_matrices',
                        dest='estimate_fundamental_matrices',
                        action='store_true',
                        help='Estimate fundamental matrices')
    parser.add_argument('--inspect_fundamental_matrices',
                        dest='inspect_fundamental_matrices',
                        action='store_true',
                        help='Inspect fundamental matrices')

    parser.add_argument('--estimate_essential_matrices',
                        dest='estimate_essential_matrices',
                        action='store_true',
                        help='Estimate essential matrices')
    parser.add_argument('--inspect_essential_matrices',
                        dest='inspect_essential_matrices',
                        action='store_true',
                        help='Inspect essential matrices')

    parser.add_argument('--triangulate', dest='triangulate',
                        action='store_true',
                        help='Triangulate from essential matrices')

    args = parser.parse_args()

    if args.detect_sift:
        run('detect_sift', DATASET_DIR, RECONSTRUCTION_DATA_FILEPATH,
            'overwrite')
    elif args.view_sift:
        run('detect_sift', DATASET_DIR, RECONSTRUCTION_DATA_FILEPATH, 'read')
    elif args.match_keypoints:
        run('match_keypoints', DATASET_DIR, RECONSTRUCTION_DATA_FILEPATH,
            'overwrite')
    elif args.estimate_fundamental_matrices:
        run('estimate_fundamental_matrices', DATASET_DIR,
            RECONSTRUCTION_DATA_FILEPATH, 'overwrite', 'debug')
    elif args.inspect_fundamental_matrices:
        run('estimate_fundamental_matrices', DATASET_DIR,
            RECONSTRUCTION_DATA_FILEPATH, 'read', 'wait_key')
    elif args.estimate_essential_matrices:
        run('estimate_essential_matrices', DATASET_DIR,
            RECONSTRUCTION_DATA_FILEPATH, 'overwrite', 'debug')
    elif args.inspect_essential_matrices:
        run('estimate_essential_matrices', DATASET_DIR,
            RECONSTRUCTION_DATA_FILEPATH, 'read', 'wait_key')
    elif args.triangulate:
        run('triangulate', DATASET_DIR,
            RECONSTRUCTION_DATA_FILEPATH, 'overwrite', 'debug')
    else:
        parser.print_help()

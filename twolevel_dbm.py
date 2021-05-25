import argparse
import os
import subprocess

from csv import reader, Error
from errno import EEXIST
from math import log, sqrt
from pathlib import PurePath  # Better path manipulation
from sys import exit

import pathos.threading as threading  # Better multiprocessing
import tqdm  # Progress bar

script = 'twolevel_dbm.py'
image_ext = '.nii'
jac_types = 'relative', 'absolute', 'nlin'
join_warp = 'delin', 'affine'
reg_mat = '0GenericAffine.mat'
reg_warp = '1Warp.nii.gz'


def is_non_zero_file(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0


def run_command(command, dry_run=False, verbose=False):
    if dry_run:
        print(f'[{script} INFO]: {command}')
        fake_return = subprocess.CompletedProcess
        fake_return.stdout = b""
        return fake_return
    else:
        if verbose:
            print(f'[{script} RUN]: {command}')
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, shell=True)
            return result
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit(f'[{script} ERROR]: Subprocess Error in {command}')


def mkdirp(path, dry_run=False):
    """mkdir -p"""
    new_path = os.path.join(path)
    if dry_run:
        print(f"[{script} INFO]: mkdir -p {new_path}")
    else:
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != EEXIST:
                raise
    return new_path


def which(program):
    """Check for existence of important programs. C.f.
    http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python"""

    def is_exe(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    fpath = os.path.split(program)[0]
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def setup_and_check_inputs(inputs, args):
    # Check if every file is accessible
    for row in inputs:
        for file in row:
            if not is_non_zero_file(file):
                exit(f"[{script} ERROR]: File {file} not exist or is empty")
    if args.resample_to_common_space and (not is_non_zero_file(args.resample_to_common_space)):
        exit(f"[{script} ERROR]: File {args.resample_to_common_space} not exist or is empty")

    # Check if multiple columns and 1level set
    if args.type == "1level":
        for row in inputs:
            if len(row) > 1:
                print(f"[{script} WARNING]: 1level model build specified but multiple columns detected in input csv")
                break

    # Warn about rows with single items in 2level builds
    if args.type == "2level":
        for row in inputs:
            if len(row) == 1:
                print(f"[{script} WARNING]: 2level model build specified but row with single scan provided, "
                      f"subject will only have overall jacobians")
                break

    # Find minimum resolution of input files unless blurs are set, or rigid model is provided
    cmd = 'PrintHeader'
    if not args.jacobian_sigmas:
        if args.rigid_model_target:
            if args.dry_run:
                run_command(f" {cmd} {args.rigid_model_target} 1", args.dry_run, args.verbose)
                args.jacobian_sigmas = [0]
            else:
                args.jacobian_sigmas = \
                    [min(map(abs, map(float, run_command(
                            f"{cmd} {args.rigid_model_target} 1", args.dry_run, args.verbose).stdout.split("x"))))
                     / sqrt(2 * log(2))]
        else:
            min_res = 1e6
            for row in inputs:
                for file in row:
                    if args.dry_run:
                        run_command(f"{cmd} {file} 1", args.dry_run, args.verbose)
                        min_res = 0
                    else:
                        res = min(
                            map(abs, map(float,
                                         run_command(f"{cmd} {file} 1", args.dry_run, args.verbose).stdout.split("x"))))
                        min_res = res if res < min_res else res
            args.jacobian_sigmas = [min_res / sqrt(2 * log(2))]


def ants_apply_transform_all(cmd_list, args, scan, i, run_type='', run_case=0):
    run_types = {'join-warp', 'resample-jac', 'combo'},
    refs = 'group', 'overall'
    base_cmd = 'antsApplyTransforms -d 3 -v '

    if run_type in run_types[0]:
        if run_type == 'join-warp':
            if run_case == 1:
                base_cmd += '-r output/level2/template0.nii.gz '
            elif run_case == 2:
                base_cmd += f'-r output/subj{i}/subj{i}-template0.nii.gz '

            if run_case == 1:
                for method in join_warp:
                    cmd_option = f'-o [output/join-warps/level2/{scan}_{method}.nii.gz,1] '
                    if method == 'delin':
                        cmd_list.append(base_cmd + cmd_option + f'-t [output/join-warps/level2/{scan}_{method}.mat,1]')
                    elif method == 'affine':
                        cmd_list.append(base_cmd + cmd_option + f'-t output/level2/{scan}-{reg_mat}')
            elif run_case == 2:
                for method in join_warp:
                    cmd_option = f'-o [output/join-warps/group/subj{scan}_{method}.nii.gz,1] '
                    if method == 'delin':
                        cmd_list.append(base_cmd + cmd_option +
                                        f'-t [output/join-warps/subj{i}/{scan}_{join_warp[0]}.mat,1]')
                    elif method == 'affine':
                        cmd_list.append(base_cmd + cmd_option + f'-t output/subj{i}/{scan}-{i}-{reg_mat}')
        elif run_type == 'resample-jac':
            path = 'output/level2'
            if run_case == 1:
                base_cmd += f'-r {args.resample_to_common_space} -t {path}/template0_ref_{reg_warp} ' \
                    f'-t {path}/template0_ref_{reg_mat} '
            elif run_case == 2:
                base_cmd += f'-r {path}/template0.nii.gz -t {path}/{scan}_template0-{reg_warp} ' \
                    f'-t {path}/{scan}_template0-{reg_mat} '

            path = 'output/jac'
            for jac_type in jac_types:
                if run_case == 1:
                    cmd_list.append(base_cmd + f'-i {path}/overall/level2_{scan}_{jac_type}.nii.gz '
                                    f'-o {path}/ref/level2_{scan}_{jac_type}.nii.gz')
                elif run_case == 2:
                    cmd_list.append(base_cmd + f'-i {path}/group/{scan}_{jac_type}.nii.gz '
                                    f'-o {path}/resampled/{scan}_{jac_type}.nii.gz')
        elif run_type == 'combo':
            tpath = 'output/level2'
            path = 'output/jac/ref'
            base_cmd += f'-r {args.resample_to_common_space} '
            cmd_option = f'-t {tpath}/template0_ref_{reg_warp} -t {tpath}/template0_ref_{reg_mat} '
            for ref in refs:
                for method in join_warp:
                    cmd_option += f'-i output/jac/{ref}/subj{i}_{scan}_{method}.nii.gz '
                    if ref == 'group':
                        cmd_option += f'-o {path}/subj{i}_{scan}_{method}.nii.gz '
                    elif ref == 'overall':
                        cmd_option += f'-o {path}/{ref}/subj{i}_{scan}_{method}.nii.gz '
                    cmd_list.append(base_cmd + cmd_option + f'-t {tpath}/subj{i}_template0-{scan}-{reg_warp} '
                                    f'-t {tpath}/subj{i}_template0-{scan}-{reg_mat}')
        else:
            exit(f'[{script} ERROR]: no such run_case, exiting...')
    else:
        exit(f'[{script} ERROR]: no such run_type, exiting...')
    return cmd_list


def build_base(args, prefix, second=False):
    """Base command, default: bootstrap model builds with rigid pre-alignment w/o update"""
    cmd = f'{args.modelbuild_command} -d 3 -o output/{prefix}_ -r 1 -y 0 ' \
        f'-a {args.average_type} -e {args.float} -g {args.gradient_step} -i {args.model_iterations} ' \
        f'-m {args.metric} -t {args.transform} ' \
        f'-f {args.reg_shrinks} -q {args.reg_iterations} -s {args.reg_smoothing} ' \
        f'-c {args.cluster_type} -u {args.walltime} -v {args.memory_request} '
    if not second:
        cmd += f'-n {int(args.N4)} '
    else:
        cmd += f"-n {int(args.N4) if not second else '0'} "
    return cmd


def first_level(inputs, args):
    cmds, images, results = [], [], []

    for subject in inputs:
        # TODO: check if inputs follow path pattern "input/[scan].nii.gz,..."
        # if [scan] contains both id & visit, probably no need for subj[i] indexing

        scan = PurePath(str(subject)).name.rsplit(image_ext)[0]
        if not is_non_zero_file(f"output/{scan}/COMPLETE"):
            if len(subject) == 1:
                command = f"mkdir -p output/{scan} && cp -p {subject[0]} output/{scan}/template0.nii.gz " \
                    f"&& ImageMath 3 output/{scan}/0-{reg_mat} MakeAffineTransform 1 "\
                    f"&& CreateImage 3 {subject[0]} output/{scan}/0-{reg_warp} 1 " \
                    f"&& CreateDisplacementField 3 1 "
                command += f"output/{scan}/0-{reg_warp} " * 3 + \
                    f"output/{scan}/0-1InverseWarp.nii.gz && CreateDisplacementField 3 1 " + \
                    f"output/{scan}/0-{reg_warp} " * 4
            else:
                command = build_base(args, f'{scan}/{scan}')

                if args.rigid_model_target:
                    command += f" -z {args.rigid_model_target} "
                command += " ".join(subject)

            command += f"&& echo DONE > output/{scan}/COMPLETE"
            cmds.append(command)
        images.append(subject + [f"output/{scan}/template0.nii.gz"])

    # TODO: add the ability to limit the number of commands submitted

    if len(cmds):
        pool = threading.ThreadPool(nodes=args.local_threads)

        print(f"[{script} INFO]: Running {len(cmds)} Level1 Model builds")
        for item in tqdm.tqdm(pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), cmds),
                              total=len(cmds)):
            results.append(item)
        if not args.dry_run:
            for i, result in enumerate(results):
                print(result)
                scan = 'abc'
                with open(f"output/{scan}.log", "wb") as l1_log:
                    l1_log.write(cmds[i].encode())
                    l1_log.write(result.stdout)

        # Completely destroy the pool so that pathos doesn't reuse
        pool.close()
        pool.clear()
    second_level(images, args, bool(1))


def second_level(inputs, args, second=False):
    path = 'output/jac'
    mkdirp(f"{path}/overall", args.dry_run)
    mkdirp(f"{path}/ref", args.dry_run)
    mkdirp(f"{path}/ref/overall", args.dry_run)
    mkdirp(f"{path}/jac/resampled", args.dry_run)
    mkdirp("output/join-warps/level2", args.dry_run)

    if second:
        input_images = [row[-1] for row in inputs]
    else:
        input_images = [val for sublist in inputs for val in sublist]

    path = 'output/level2'
    if not is_non_zero_file(f"{path}/COMPLETE"):
        cmd = build_base(args, 'level2/level2', bool(1))
        if args.rigid_model_target:
            cmd += f" -z {args.rigid_model_target} "
        if not second:
            cmd += f"{args.input} "
        else:
            cmd += " ".join(input_images)
        cmd += f"&& echo DONE > {path}/COMPLETE"
        print(f"[{script} INFO]: Running Level2 Model build")
        results = run_command(cmd, args.dry_run, args.verbose)

        # TODO: add the ability to limit the number of commands submitted
        if not args.dry_run:
            with open(f"{path}/level2.log", "wb") as logfile:
                logfile.write(cmd.encode())
                logfile.write(results.stdout)

    pool = threading.ThreadPool(nodes=args.local_threads)

    if args.skip_dbm:
        print(f"[{script} INFO]: Skipping generation of DBM outputs\n"
              f"[{script} INFO]: Pipeline Complete")

    # Create mask for delin
    run_command(f"ThresholdImage 3 {path}/template0.nii.gz {path}/otsumask.nii.gz Otsu 1",
                args.dry_run, args.verbose)

    # Register final model to common space
    if not is_non_zero_file(f"{path}/template0_ref_COMPLETE") and args.resample_to_common_space:
        print(f"[{script} INFO]: Registering final model build to target common space")
        run_command(f"antsRegistrationSyN.sh -d 3 -o {path}/template0_ref_-f {args.resample_to_common_space} "
                    f"-m {path}/template0.nii.gz ", args.dry_run, args.verbose)
        run_command(f"echo DONE > {path}/template0_ref_COMPLETE", args.dry_run, args.verbose)

    print(f"[{script} INFO]: Processing Level2 DBM outputs")

    # Loop over input file warp fields to produce delin
    jacobians, cmds = [], []

    for i, subject in enumerate(tqdm.tqdm(input_images)):
        scan = PurePath(subject).name.rsplit(image_ext)[0]
        if not is_non_zero_file("output/join-warps/level2/COMPLETE"):
            # Compute delin
            run_command(f"ANTSUseDeformationFieldToGetAffineTransform "
                        f"{path}/{scan}-{i}-{reg_warp} 0.25 {join_warp[1]} "
                        f"output/join-warps/level2/{scan}_{join_warp[0]}.mat "
                        f"{path}/level2_otsumask.nii.gz", args.dry_run, args.verbose)

            # Create composite field of delin & affine
            pool.map(lambda x: run_command(x, args.dry_run, args.verbose),
                     ants_apply_transform_all(cmds, args, scan, i, run_type='join-warp', run_case=1))

            cmds.clear()
            # Generate jacobians of composite affine fields and nonlinear fields
            cmds.append(f"CreateJacobianDeterminantImage 3 {path}/{scan}-{i}-{reg_warp} "
                        f"output/jac/overall/level2_{scan}_{jac_types[2]}.nii.gz 1 1")
            for method in join_warp:
                cmds.append(f"CreateJacobianDeterminantImage 3 output/join-warps/level2/{scan}_{method}.nii.gz "
                            f"output/jac/overall/level2_{scan}_{method}.nii.gz 1 1")

            pool.map(lambda x: run_command(x, args.dry_run, args.verbose), cmds)

            cmds.clear()
            base_cmd = f"ImageMath 3 output/jac/overall/level2_{scan}_{jac_types[2]}.nii.gz + "
            cmds.append(base_cmd + f"output/jac/overall/level2_{scan}_{jac_types[0]}.nii.gz "
                        f"output/jac/overall/level2_{scan}_{join_warp[0]}.nii.gz")
            cmds.append(base_cmd + f"output/jac/overall/level2_{scan}_{jac_types[1]}.nii.gz "
                        f"output/jac/overall/level2_{scan}_{join_warp[1]}.nii.gz")
            pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), cmds)

        for jac_type in jac_types:
            jacobians.append(f"output/jac/overall/level2_{scan}_{jac_type}.nii.gz")

    run_command("echo DONE > output/join-warps/level2/COMPLETE", args.dry_run, args.verbose)

    path = 'output/jac/ref'
    if not second and args.resample_to_common_space:
        for i, subject in enumerate(tqdm.tqdm(input_images)):
            scan = PurePath(subject).name.rsplit(image_ext)[0]
            if not is_non_zero_file(f"{path}/COMPLETE"):
                pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose),
                           ants_apply_transform_all(cmds, args, scan, i, run_type='resample-jac', run_case=1))

            for jac_type in jac_types:
                jacobians.append(f"{path}/level2_{scan}_{jac_type}.nii.gz")

    if not second and args.resample_to_common_space:
        run_command(f"echo DONE > {path}/COMPLETE", args.dry_run, args.verbose)

    if second:
        if args.resample_to_common_space:
            mkdirp("output/join-warps/group", args.dry_run)
            mkdirp("output/jac/group", args.dry_run)
        print(f"[{script} INFO]: Processing First-Level DBM Outputs")
        for i, row in enumerate(tqdm.tqdm([line[:-1] for line in inputs])):
            if not is_non_zero_file("output/jac/resampled/COMPLETE"):
                # Make a mask per subject
                run_command(f"ThresholdImage 3 output/subj{i}/template0.nii.gz "
                            f"output/subj{i}/otsumask.nii.gz Otsu 1", args.dry_run, args.verbose)

                for j, scans in enumerate(row):
                    scan = PurePath(scans).name.rsplit(image_ext)[0]
                    # Estimate affine residual from nonlinear and create composite warp and jacobian field
                    run_command(f"ANTSUseDeformationFieldToGetAffineTransform "
                                f"output/subj{i}/{scan}-{j}-{reg_warp} 0.25 {join_warp[1]} "
                                f"output/join-warps/group/{scan}_{join_warp[0]}.mat "
                                f"output/subj{i}/otsumask.nii.gz", args.dry_run, args.verbose)

                    cmds.clear()
                    # Create composite warp field from delin & affine
                    pool.map(lambda x: run_command(x, args.dry_run, args.verbose),
                             ants_apply_transform_all(cmds, args, scan, j, run_type='join-warp', run_case=2))

                    cmds.clear()
                    # Create jacobian images from nlin and composite warp fields
                    path = 'output/jac/group'
                    cmds.append(f"CreateJacobianDeterminantImage 3 output/subj{i}/{scan}-{j}-{reg_warp} "
                                f"{path}/{scan}_{jac_types[2]}.nii.gz 1 1")
                    for method in join_warp:
                        cmds.append(f"CreateJacobianDeterminantImage 3 output/join-warps/group/{scan}_{method}.nii.gz "
                                    f"{path}/{scan}_{method}.nii.gz 1 1")
                    pool.map(lambda x: run_command(x, args.dry_run, args.verbose), cmds)

                    cmds.clear()
                    # Create relative and absolute jacobians by adding affine/delin jacobians
                    base_cmd = f"ImageMath 3 {path}/{scan}_{jac_types[2]}.nii.gz + "
                    cmds.append(base_cmd + f"{path}/{scan}_{jac_types[0]}.nii.gz + "
                                f"{path}/{scan}_{join_warp[0]}.nii.gz")
                    cmds.append(base_cmd + f"{path}/{scan}_{jac_types[1]}.nii.gz + "
                                f"{path}/{scan}_{join_warp[1]}.nii.gz")
                    pool.map(lambda x: run_command(x, args.dry_run, args.verbose), cmds)

                    cmds.clear()
                    # Resample jacobian to common space
                    pool.map(lambda x: run_command(x, args.dry_run, args.verbose),
                             ants_apply_transform_all(cmds, args, scan, j, run_type='resample-jac', run_case=2))

                    cmds.clear()
                    path = 'output/jac'
                    for jac_type in jac_types:
                        cmds.append(f"ImageMath 3 {path}/overall/subj{i}_{scan}_{jac_type}.nii.gz + "
                                    f"{path}/resampled/subj{i}_{scan}_{jac_type}.nii.gz "
                                    f"{path}/overall/level2_subj{i}_template0_{jac_type}.nii.gz")
                    pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), cmds)

                    cmds.clear()
                    if args.resample_to_common_space:
                        pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose),
                                   ants_apply_transform_all(cmds, args, scan, j, run_type='combo'))

                    # Append jacobians to list
                    for jac_type in jac_types:
                        jacobians.append(f"{path}/resampled/{scan}_{jac_type}.nii.gz")
                        jacobians.append(f"{path}/overall/{scan}_{jac_type}.nii.gz")

                    if args.resample_to_common_space:
                        path = 'output/jac/ref'
                        for jac_type in jac_types:
                            jacobians.append(f"{path}/{scan}_{jac_type}.nii.gz")
                        for _ in range(args.model_iterations):
                            jacobians.append(f"{path}/overall/{scan}_{jac_types[1]}.nii.gz")

    print(f"[{script} INFO]: Blurring Jacobians")
    for blur in args.jacobian_sigmas:
        cmds.append(f"echo {blur} > smooth")
        for jacobian in jacobians:
            cmds.append(f"SmoothImage 3 {jacobian} {blur} {jacobian.rsplit(image_ext)[0]}_smooth.nii.gz 1 0")

    for _ in tqdm.tqdm(pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), cmds), total=len(cmds)):
        pass

    pool.close()
    pool.clear()
    run_command("echo DONE > output/jac/resampled/COMPLETE", args.dry_run, args.verbose)
    print(f"[{script} INFO]: Pipeline Complete")


def read_csv(input_file):
    inputs = []
    with open(input_file, newline="") as csv_file:
        csv = reader(csv_file)
        try:
            for row in csv:
                inputs.append(list(filter(None, row)))
        except Error as e:
            exit(f"malformed csv: file {input}, line {csv.line_num}: {e}")
    return inputs


def main():
    # Gives --option --no-option paired control, c.f.
    # https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/

    class BooleanAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super(BooleanAction, self).__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, False if option_string.startswith("--no") else 1)

    build_command = 'antsMultivariateTemplateConstruction2.sh'
    dbm_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"This pipeline performs one or two level model building on files using {build_command} and makes "
                    "smoothed jacobian determinent fields suitable for deformation based morphomometry (DBM) analysis.")

    dbm_parser.add_argument("type", choices=["1level", "2level"], help="Type of DBM processing to run")

    dbm_parser.add_argument("input",
                            help="Input CSV file for DBM.\n1level: a single column\n"
                                 "2level: each each row constructs a level1 model followed by a level2 model "
                                 "of the resulting level1 averages.")

    dbm_parser.add_argument("--jacobian-sigmas", nargs="+", type=float,
                            help="List of smoothing sigmas for final output\n"
                                 "defaults: FWHM of twice the finest resolution of "
                                 "input or rigid model target if provided.")

    dbm_parser.add_argument("--rigid-model-target",
                            help="Target image for rigid registration of the level2, otherwise use unbiased average")

    dbm_parser.add_argument("-t", "--resample-to-common-space",
                            help="Target atlas space to resample jacobians to after unbiased model build, "
                                 "typically an MNI model, triggers a registration to this target")

    dbm_parser.add_argument("--skip-dbm", action="store_true", help="Skip generating DBM outputs")

    dbm_parser.add_argument("-d", "--dry-run", action="store_true", help="Don't run commands, instead print to stdout")

    dbm_parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose about what is going on")

    advanced = dbm_parser.add_argument_group("advanced options")
    advanced.add_argument("--N4", "--no-N4", action=BooleanAction, dest="N4", default=False,
                          help="Run N4BiasFieldCorrection during model build on input files")

    advanced.add_argument("--metric", default="CC[4]", help="Specify metric used for non-linear stages")

    advanced.add_argument("--transform", default="SyN",
                          choices=["SyN", "BSplineSyN", "Affine", "Rigid",
                                   "TimeVaryingVelocityField", "TimeVaryingBSplineVelocityField"],
                          help="Transformation type to use")

    advanced.add_argument("-i", "--reg-iterations", default="100x100x70x20",
                          help="Max iterations for non-linear stages")

    advanced.add_argument("--reg-smoothing", default="3x2x1x0", help="Smoothing sigmas for non-linear stages")

    advanced.add_argument("--reg-shrinks", default="6x4x2x1", help="Shrink factors for non-linear stages")

    advanced.add_argument("--float", "--no-float", action=BooleanAction, dest="float", default=1,
                          help="Run registration with float (32 bit) or double (64 bit) values")

    advanced.add_argument("--average-type", default="normmean", choices=["mean", "normmean", "median"],
                          help="Type of average to use")

    advanced.add_argument("--gradient-step", default=0.25, type=float, help="Gradient step size at each iteration")

    advanced.add_argument("--model-iterations", default=4, type=int, help="Number of registration and average")

    advanced.add_argument("--modelbuild-command", default=build_command,
                          help=f"Command for model build, arguments must be same as in {build_command}")

    cluster = dbm_parser.add_argument_group("cluster options")
    cluster.add_argument("-c", "--cluster-type", default="local", choices=["local", "sge", "pbs", "slurm"],
                         help="Type of cluster for job submission")

    cluster.add_argument("--walltime", default="4:00:00", help="Specify requested time per pairwise registration")

    cluster.add_argument("--memory-request", default="8gb", help="Specify requested memory per pairwise registration")

    cluster.add_argument("-j", "--local-threads", type=int, default=threading.cpu_count() // 2,
                         help="# subject-wise model builds to run in parallel if run locally")

    args = dbm_parser.parse_args()
    # Convert inputs into values for model build command
    args.float = int(args.float)
    cluster_choices = {"local": 0, "sge": 1, "pbs": 4, "slurm": 5}
    args.cluster_type = cluster_choices[args.cluster_type]
    average_choices = {"mean": 0, "normmean": 1, "median": 2}
    args.average_type = average_choices[args.average_type]

    if not (len(args.reg_iterations.split("x")) == len(args.reg_shrinks.split("x"))
            == len(args.reg_smoothing.split("x"))):
        exit(f"[{script} ERROR]: iterations, shrinks and smoothing do not match in length")

    if not which(args.modelbuild_command):
        exit(f"{args.modelbuild_command} command not found")

    inputs = read_csv(args.input)
    setup_and_check_inputs(inputs, args)

    if args.type == "2level":
        first_level(inputs, args)
    else:
        second_level(inputs, args)


if __name__ == "__main__":
    main()

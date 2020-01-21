#!/usr/bin/env python3

import argparse
import csv
import os
import errno
import pathlib  # Better path manipulation
import shlex
import subprocess
import sys

import pathos.threading as threading  # Better multiprocessing
import tqdm  # Progress bar


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def run_command(command, dryrun, verbose=False):
    if dryrun:
        print(command)
        fakereturn = subprocess.CompletedProcess
        fakereturn.stdout = "".encode()
        return fakereturn
    else:
        if verbose:
            print(f"twolevel_dbm.py: running command {command}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            shell=True,
        )
        return result


def mkdirp(*p):
    """Like mkdir -p"""
    path = os.path.join(*p)

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise
    return path


def which(program):
    # Check for existence of important programs
    # Stolen from
    # http://stackoverflow.com/questions/377017/test-if-executable-exists-in-python # noqa
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
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
    # Check every file is accessible
    for row in inputs:
        for file in row:
            if not is_non_zero_file(file):
                sys.exit(
                    f"twolevel_dbm error: File {file} does "
                    + "not exist or is zero size"
                )
    if args.resample_to_common_space and (
        not is_non_zero_file(args.resample_to_common_space)
    ):
        sys.exit(
            f"twolevel_dbm error: File {args.resample_to_common_space} does not exist or is zero size"
        )

    # Find minimum resolution of input files unless blurs are set, or rigid
    # model is provided
    if not args.jacobian_sigmas:
        if args.rigid_model_target:
            if args.dry_run:
                run_command(
                    "PrintHeader {} 1".format(args.rigid_model_target), args.dry_run
                )
            else:
                args.jacobian_sigmas = [
                    2
                    * min(
                        map(
                            abs,
                            map(
                                float,
                                run_command(
                                    "PrintHeader {} 1".format(args.rigid_model_target),
                                    args.dry_run,
                                )
                                .stdout.decode("utf8")
                                .split("x"),
                            ),
                        )
                    )
                ]
        else:
            minres = 1e6
            for row in inputs:
                for file in row:
                    if args.dry_run:
                        run_command("PrintHeader {} 1".format(file), args.dry_run)
                    else:
                        curres = min(
                            map(
                                abs,
                                map(
                                    float,
                                    run_command(
                                        "PrintHeader {} 1".format(file), args.dry_run
                                    )
                                    .stdout.decode("utf8")
                                    .split("x"),
                                ),
                            )
                        )
                        minres = curres if curres < minres else curres
            args.jacobian_sigmas = [2 * minres]


def firstlevel(inputs, args):
    commands = list()
    imagelist = list()
    for i, subject in enumerate(inputs, start=0):
        if not is_non_zero_file("output/subject{}/COMPLETE".format(i)):
            # Base command
            command = f"{args.modelbuild_command} -d 3 "
            # Setup directory and naming
            command += "-o output/subject{}/subject{}_ ".format(i, i)
            # Defaults to bootstrap modelbuilds with rigid prealignmnet,
            # no rigid update
            command += "-r 1 -l 1 -y 0 "
            # Model build setup
            command += "-c {} -a {} -e {} -g {} -i {} -n {} -m {} -t {} -u {} -v {} ".format(
                args.cluster_type,
                args.average_type,
                args.float,
                args.gradient_step,
                args.model_iterations,
                int(args.N4),
                args.metric,
                args.transform,
                args.walltime,
                args.memory_request,
            )
            # Registrations Setup
            command += "-q {} -f {} -s {} ".format(
                args.reg_iterations, args.reg_shrinks, args.reg_smoothing
            )
            if args.rigid_model_target:
                command += "-z {} ".format(args.rigid_model_target)
            command += " ".join(subject)
            command += " && echo DONE > output/subject{}/COMPLETE".format(i)
            commands.append(command)

        imagelist.append(
            subject + ["output/subject{0}/subject{0}_template0.nii.gz".format(i)]
        )
    # Here we should add the ability to limit the number of commands submitted
    results = list()
    if len(commands) > 0:
        if args.cluster_type != 0:
            pool = threading.ThreadPool(
                nodes=min(len(commands), threading.cpu_count() // 2)
            )
        else:
            pool = threading.ThreadPool(nodes=args.local_threads)

        print(f"Running {len(commands)} First-Level Modelbuilds")
        for item in tqdm.tqdm(
            pool.uimap(lambda x: run_command(x, args.dry_run), commands),
            total=len(commands),
        ):
            results.append(item)
        if not args.dry_run:
            for i, subject in enumerate(results, start=0):
                with open(
                    "output/subject{0}/subject{0}.log".format(i), "wb"
                ) as logfile:
                    logfile.write(subject.stdout)
        pool.close()
        # Needed to completely destroy the pool so that pathos doesn't reuse
        pool.clear()
    secondlevel(imagelist, args, secondlevel=True)


def secondlevel(inputs, args, secondlevel=False):
    outputs = list()
    if secondlevel:
        input_images = [row[-1] for row in inputs]
    else:
        input_images = [val for sublist in inputs for val in sublist]
    if not is_non_zero_file("output/secondlevel/COMPLETE"):
        # Base command
        command = f"{args.modelbuild_command} -d 3 "
        # Setup directory and naming
        command += "-o output/secondlevel/secondlevel_ "
        # Defaults to bootstrap modelbuilds with rigid prealignmnet, no rigid
        # update
        command += """-r 1 -l 1 -y 0 """
        # Model build setup
        command += "-c {} -a {} -e {} -g {} -i {} -n {} -m {} -t {} -u {} -v {} ".format(
            args.cluster_type,
            args.average_type,
            args.float,
            args.gradient_step,
            args.model_iterations,
            (not secondlevel) and int(args.N4) or "0",
            args.metric,
            args.transform,
            args.walltime,
            args.memory_request,
        )
        # Registrations Setup
        command += "-q {} -f {} -s {} ".format(
            args.reg_iterations, args.reg_shrinks, args.reg_smoothing
        )
        if args.rigid_model_target:
            command += "-z {} ".format(args.rigid_model_target)
        command += " ".join(input_images)
        command += " && echo DONE > output/secondlevel/COMPLETE"
        print("Running Second-Level Modelbuild")
        results = run_command(command, args.dry_run)
        # Here we should add the ability to limit the number of commands submitted
        if not args.dry_run:
            with open("output/secondlevel/secondlevel.log", "wb") as logfile:
                logfile.write(results.stdout)

    pool = threading.ThreadPool(nodes=args.local_threads)

    if args.skip_dbm:
        print("Skipping generation of DBM outputs")
        sys.exit()

    mkdirp("output/jacobians/overall")
    mkdirp("output/compositewarps/secondlevel")
    # Create mask for delin
    run_command(
        "ThresholdImage 3 output/secondlevel/secondlevel_template0.nii.gz output/secondlevel/secondlevel_otsumask.nii.gz Otsu 1",
        args.dry_run,
    )
    # Register final model to common space
    if not is_non_zero_file("output/secondlevel/template0_common_space_COMPLETE") and args.resample_to_common_space:
        print("Registering final modelbuild to target common space")
        run_command(
            f"antsRegistrationSyN.sh -d 3 -f {args.resample_to_common_space} -m output/secondlevel/secondlevel_template0.nii.gz -o output/secondlevel/template0_common_space_",
            args.dry_run,
        )
        run_command("echo DONE > output/secondlevel/template0_common_space_COMPLETE", args.dry_run)

    print("Processing Second-Level DBM outputs")
    # Loop over input file warp fields to produce delin
    jacobians = list()
    for i, subject in enumerate(tqdm.tqdm(input_images), start=0):
        subjectname = pathlib.Path(subject).name.rsplit(".nii")[0]
        if not is_non_zero_file("output/compositewarps/secondlevel/COMPLETE"):
            commands = list()
            # Compute delin
            run_command(
                f"ANTSUseDeformationFieldToGetAffineTransform output/secondlevel/secondlevel_{subjectname}{i}1InverseWarp.nii.gz 0.25 "
                f"affine output/compositewarps/secondlevel/{subjectname}_delin.mat output/secondlevel/secondlevel_otsumask.nii.gz",
                args.dry_run,
            )

            # Create composite field of delin
            commands.append(
                f"antsApplyTransforms -d 3 -t [output/compositewarps/secondlevel/{subjectname}_delin.mat,1] "
                f"-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/secondlevel/{subjectname}_delin.nii.gz,1]"
            )

            # Create composite field of affine
            commands.append(
                f"antsApplyTransforms -d 3 -t [output/secondlevel/secondlevel_{subjectname}{i}0GenericAffine.mat,1] "
                f"-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/secondlevel/{subjectname}_affine.nii.gz,1]"
            )

            pool.map(lambda x: run_command(x, args.dry_run), commands)
            commands = list()

            # Generate jacobians of composite affine fields and nonlinear fields
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/secondlevel/secondlevel_{subjectname}{i}1Warp.nii.gz output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz 1 1"
            )
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel/{subjectname}_delin.nii.gz output/jacobians/overall/secondlevel_{subjectname}_delin.nii.gz 1 1"
            )
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel/{subjectname}_affine.nii.gz output/jacobians/overall/secondlevel_{subjectname}_affine.nii.gz 1 1"
            )

            pool.map(lambda x: run_command(x, args.dry_run), commands)
            commands = list()

            commands.append(
                f"ImageMath 3 output/jacobians/overall/secondlevel_{subjectname}_relative.nii.gz + output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz output/jacobians/overall/secondlevel_{subjectname}_delin.nii.gz"
            )
            commands.append(
                f"ImageMath 3 output/jacobians/overall/secondlevel_{subjectname}_absolute.nii.gz + output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz output/jacobians/overall/secondlevel_{subjectname}_affine.nii.gz"
            )
            pool.uimap(lambda x: run_command(x, args.dry_run), commands)
            commands = list()

        jacobians.append(
            f"output/jacobians/overall/secondlevel_{subjectname}_relative.nii.gz"
        )
        jacobians.append(
            f"output/jacobians/overall/secondlevel_{subjectname}_absolute.nii.gz"
        )
        jacobians.append(
            f"output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz"
        )

    run_command("echo DONE > output/compositewarps/secondlevel/COMPLETE", args.dry_run)

    if not secondlevel and args.resample_to_common_space:
        mkdirp("output/jacobians/common_space")
        for i, subject in enumerate(tqdm.tqdm(input_images), start=0):
            subjectname = pathlib.Path(subject).name.rsplit(".nii")[0]
            if not is_non_zero_file("output/jacobians/common_space/COMPLETE"):
                commands.append(
                    f"antsApplyTransforms -d 3 -i output/jacobians/overall/secondlevel_{subjectname}_relative.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/secondlevel_{subjectname}_relative.nii.gz"
                )
                commands.append(
                    f"antsApplyTransforms -d 3 -i output/jacobians/overall/secondlevel_{subjectname}_absolute.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/secondlevel_{subjectname}_absolute.nii.gz"
                )
                commands.append(
                    f"antsApplyTransforms -d 3 -i output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/secondlevel_{subjectname}_nlin.nii.gz"
                )

                pool.uimap(lambda x: run_command(x, args.dry_run), commands)
                commands = list()

            jacobians.append(
                f"output/jacobians/common_space/secondlevel_{subjectname}_relative.nii.gz"
            )
            jacobians.append(
                f"output/jacobians/common_space/secondlevel_{subjectname}_absolute.nii.gz"
            )
            jacobians.append(
                f"output/jacobians/common_space/secondlevel_{subjectname}_nlin.nii.gz"
            )

    run_command("echo DONE > output/jacobians/common_space/COMPLETE", args.dry_run)

    if secondlevel:
        mkdirp("output/compositewarps/groupwise")
        mkdirp("output/jacobians/resampled")
        mkdirp("output/jacobians/groupwise")
        print("Processing First-Level DBM Outputs")
        for subjectnum, row in enumerate(
            tqdm.tqdm([line[:-1] for line in inputs]), start=0
        ):
            if not is_non_zero_file("output/jacobians/resampled/COMPLETE"):
                # Make a mask per subject
                run_command(
                    f"ThresholdImage 3 output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz output/subject{subjectnum}/subject{subjectnum}_otsumask.nii.gz Otsu 1",
                    args.dry_run,
                )
                for scannum, scan in enumerate(row, start=0):
                    commands = list()
                    scanname = pathlib.Path(scan).name.rsplit(".nii")[0]
                    # Estimate affine residual from nonlinear and create composite warp and jacobian field
                    run_command(
                        f"ANTSUseDeformationFieldToGetAffineTransform output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}1InverseWarp.nii.gz 0.25 "
                        f"affine output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.mat output/subject{subjectnum}/subject{subjectnum}_otsumask.nii.gz",
                        args.dry_run,
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 -t [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.mat,1] -r output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz "
                        f"--verbose -o [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz,1]"
                    )
                    # Create composite warp field from affine
                    commands.append(
                        f"antsApplyTransforms -d 3 -t [output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}0GenericAffine.mat,1] -r output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz "
                        f"--verbose -o [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz,1]"
                    )

                    pool.map(lambda x: run_command(x, args.dry_run), commands)
                    commands = list()

                    # Create jacobian images from nlin and composite warp fields
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}1Warp.nii.gz output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz 1 1"
                    )
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz output/jacobians/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz 1 1"
                    )
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/compositewarps/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz output/jacobians/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz 1 1"
                    )

                    pool.map(lambda x: run_command(x, args.dry_run), commands)
                    commands = list()

                    # Create relative and absolute jacobians by adding affine/delin jacobians
                    commands.append(
                        f"ImageMath 3 output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz + output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz output/jacobians/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz"
                    )
                    commands.append(
                        f"ImageMath 3 output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz + output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz output/jacobians/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz"
                    )

                    pool.map(lambda x: run_command(x, args.dry_run), commands)
                    commands = list()

                    # Resample jacobian to common space
                    commands.append(
                        f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/jacobians/resampled/subject{subjectnum}_{scanname}_relative.nii.gz"
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/jacobians/resampled/subject{subjectnum}_{scanname}_absolute.nii.gz"
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/jacobians/resampled/subject{subjectnum}_{scanname}_nlin.nii.gz"
                    )

                    if args.resample_to_common_space:
                        mkdirp("output/jacobians/common_space")
                        commands.append(
                            f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/subject{subjectnum}_{scanname}_relative.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 -i output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz -t output/secondlevel/template0_common_space_1Warp.nii.gz -t output/secondlevel/template0_common_space_0GenericAffine.mat -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} --verbose -o output/jacobians/common_space/subject{subjectnum}_{scanname}_nlin.nii.gz"
                        )

                    pool.uimap(lambda x: run_command(x, args.dry_run), commands)
                    commands = list()

                # Append jacobians to list
                jacobians.append(
                    f"output/jacobians/resampled/subject{subjectnum}_{scanname}_relative.nii.gz"
                )
                jacobians.append(
                    f"output/jacobians/resampled/subject{subjectnum}_{scanname}_absolute.nii.gz"
                )
                jacobians.append(
                    f"output/jacobians/resampled/subject{subjectnum}_{scanname}_nlin.nii.gz"
                )
                if args.resample_to_common_space:
                    jacobians.append(
                        f"output/jacobians/common_space/subject{subjectnum}_{scanname}_relative.nii.gz"
                    )
                    jacobians.append(
                        f"output/jacobians/common_space/subject{subjectnum}_{scanname}_absolute.nii.gz"
                    )
                    jacobians.append(
                        f"output/jacobians/common_space/subject{subjectnum}_{scanname}_nlin.nii.gz"
                    )
        run_command("echo DONE > output/jacobians/resampled/COMPLETE", args.dry_run)

    commands = list()
    print("Blurring Jacobians")
    for jacobian in jacobians:
        for blur in args.jacobian_sigmas:
            commands.append(
                f"SmoothImage 3 {jacobian} {blur} {jacobian.rsplit('.nii')[0]}_smooth{blur}.nii.gz 1 0"
            )
    for _ in tqdm.tqdm(
        pool.uimap(lambda x: run_command(x, args.dry_run), commands),
        total=len(commands),
    ):
        pass

    pool.close()
    pool.clear()


def read_csv(inputfile):
    inputs = []
    with open(inputfile, newline="") as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row in reader:
                inputs.append(list(filter(None, row)))
        except csv.Error as e:
            sys.exit(
                "malformed csv: file {}, line {}: {}".format(
                    inputfile, reader.line_num, e
                )
            )
    return inputs


def main():

    # Stolen from https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
    # Gives me --option --no-option paried control
    class BooleanAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(BooleanAction, self).__init__(option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(
                namespace,
                self.dest,
                False if option_string.startswith("--no") else True,
            )

    parser = argparse.ArgumentParser(
        description="""This pipeline performs one or
        two level model building on files using antsMultivariateTemplateConstruction2.sh
        and generates smoothed jacobian determinent fields suitable for deformation
        based morphomometry (DBM) analysis.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "type",
        choices=["1level", "2level"],
        help="""What type of DBM processing to run on input file, see input
            for details on how to format CSV file for different types.""",
    )
    parser.add_argument(
        "input",
        help="""Input CSV file for DBM, for
        1level mode, a single column, for 2level, each each row constructs a first level model
        followed by a second level model of the resulting first level averages. File paths must
        be absolute.""",
    )
    parser.add_argument(
        "--jacobian-sigmas",
        nargs="+",
        type=float,
        help="""List of smoothing
        sigmas used for final output, defaults to 2x finest resolution input file or
        rigid model target if provided.""",
    )
    parser.add_argument(
        "--rigid-model-target",
        help="""Target file to use for rigid
        registration of the second level, otherwise unbiased average to start""",
    )
    parser.add_argument(
        "--resample-to-common-space",
        help="""Target nifti file of atlas space to resample to
        jacobians to after unbiased model build, typically an MNI model, triggers a
        registration to this target""",
    )
    parser.add_argument(
        "--skip-dbm",
        action="store_true",
        help="Skip generating DBM outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't run commands, instead print to stdout",
    )

    advanced = parser.add_argument_group("advanced options")
    advanced.add_argument(
        "--N4",
        "--no-N4",
        action=BooleanAction,
        dest="N4",
        default=False,
        help="Run N4BiasFieldCorrection during model build on input files.",
    )
    advanced.add_argument(
        "--metric",
        default="CC[4]",
        help="Specify metric used for non-linear template stages",
    )
    advanced.add_argument(
        "--transform",
        default="SyN",
        choices=[
            "SyN",
            "BSplineSyN",
            "TimeVaryingVelocityField",
            "TimeVaryingBSplineVelocityField",
            "Affine",
            "Rigid",
        ],
        help="Transformation type used during model build",
    )
    advanced.add_argument(
        "--reg-iterations",
        default="100x100x70x20",
        help="Max iterations for non-linear stages",
    )
    advanced.add_argument(
        "--reg-smoothing",
        default="3x2x1x0",
        help="Smoothing sigmas for non-linear stages",
    )
    advanced.add_argument(
        "--reg-shrinks", default="6x4x2x1", help="Shrink factors for non-linear stages"
    )
    advanced.add_argument(
        "--float",
        "--no-float",
        action=BooleanAction,
        dest="float",
        default=True,
        help="Run registration with float (32 bit) or double (64 bit) values",
    )
    advanced.add_argument(
        "--average-type",
        default="normmean",
        choices=["mean", "normmean", "median"],
        help="Type of average used during model build, default normalized mean",
    )
    advanced.add_argument(
        "--gradient-step",
        default=0.25,
        type=float,
        help="Gradient step size at each iteration during model build",
    )
    advanced.add_argument(
        "--model-iterations",
        default=3,
        type=int,
        help="How many registration and average rounds to do",
    )
    advanced.add_argument(
        "--modelbuild-command",
        default="antsMultivariateTemplateConstruction2.sh",
        help="""Command to use for performing model build, must accept same
        arguments as antsMultivariateTemplateConstruction2.sh""",
    )

    cluster = parser.add_argument_group("cluster options")
    cluster.add_argument(
        "--cluster-type",
        default="local",
        choices=["local", "sge", "pbs", "slurm"],
        help="Choose the type of cluster system to submit jobs to",
    )
    cluster.add_argument(
        "--walltime",
        default="20:00:00",
        help="""Option for job submission
        specifying requested time per pairwise registration.""",
    )
    cluster.add_argument(
        "--memory-request",
        default="8gb",
        help="""Option for job submission
        specifying requested memory per pairwise registration.""",
    )
    cluster.add_argument(
        "--local-threads",
        "-j",
        type=int,
        default=threading.cpu_count() // 2,
        help="""For local execution, how many subject-wise modelbuilds to run in parallel,
        defaults to number of CPUs""",
    )

    args = parser.parse_args()
    # Convert inputs into values for model build command
    args.float = int(args.float)
    clusterchoices = {"local": 0, "sge": 1, "pbs": 4, "slurm": 5}
    args.cluster_type = clusterchoices[args.cluster_type]
    averagechoices = {"mean": 0, "normmean": 1, "median": 2}
    args.average_type = averagechoices[args.average_type]

    if not (
        len(args.reg_iterations.split("x"))
        == len(args.reg_shrinks.split("x"))
        == len(args.reg_smoothing.split("x"))
    ):
        sys.exit(
            "twolevel_dbm.py error: iterations, shrinks and smoothing do "
            "not match in length"
        )

    if not which(args.modelbuild_command):
        sys.exit(f"{args.modelbuild_command} command not found")

    inputs = read_csv(args.input)
    setup_and_check_inputs(inputs, args)

    if args.type == "2level":
        firstlevel(inputs, args)
    else:
        secondlevel(inputs, args, secondlevel=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import os
import errno
import subprocess

from csv import reader, Error
from math import log, sqrt
from pathlib import PurePath  # Better path manipulation
from sys import exit

import pathos.threading as threading  # Better multiprocessing
import tqdm  # Progress bar

script_name = 'twolevel_dbm.py'
image_ext = '.nii'


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def run_command(command, dry_run=False, verbose=False):
    if dry_run:
        print(f'[{script_name} INFO]: {command}')
        fake_return = subprocess.CompletedProcess
        fake_return.stdout = b""
        return fake_return
    else:
        if verbose:
            print(f'[{script_name} INFO]: {command}')
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True, shell=True)
            return result
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit(f'[{script_name} INFO]: Subprocess Error in: {command}')


def mkdirp(path, dry_run=False):
    """mkdir -p"""
    new_path = os.path.join(path)
    if dry_run:
        print(f"{script_name}: would run mkdir -p {new_path}")
    else:
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
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
                exit(f"{script_name} error: File {file} not exist or is empty")
    if args.resample_to_common_space and (not is_non_zero_file(args.resample_to_common_space)):
        exit(f"{script_name} error: File {args.resample_to_common_space} not exist or is empty")

    # Check if multiple columns and 1level set
    if args.type == "1level":
        for row in inputs:
            if len(row) > 1:
                print(f"{script_name} warning: 1level model build specified but multiple columns detected in input csv")
                break

    # Warn about rows with single items in 2level builds
    if args.type == "2level":
        for row in inputs:
            if len(row) == 1:
                print(f"{script_name} warning: 2level model build specified but row with single scan provided, "
                      "subject will only have overall jacobains")
                break

    # Find minimum resolution of input files unless blurs are set, or rigid model is provided
    if not args.jacobian_sigmas:
        if args.rigid_model_target:
            if args.dry_run:
                run_command(f"PrintHeader {args.rigid_model_target} 1", args.dry_run, args.verbose)
                args.jacobian_sigmas = [0]
            else:
                args.jacobian_sigmas = [min(map(abs, map(float, run_command(
                    f"PrintHeader {args.rigid_model_target} 1", args.dry_run, args.verbose).stdout.split("x"))))
                    / sqrt(2 * log(2))]
        else:
            min_res = 1e6
            for row in inputs:
                for file in row:
                    if args.dry_run:
                        run_command(f"PrintHeader {file} 1", args.dry_run, args.verbose)
                        min_res = 0
                    else:
                        res = min(map(abs, map(float, run_command(
                            f"PrintHeader {file} 1", args.dry_run, args.verbose).stdout.split("x"))))
                        min_res = res if res < min_res else res
            args.jacobian_sigmas = [min_res / sqrt(2 * log(2))]


def first_level(inputs, args):
    cmds, images, results = list(), list(), list()

    for subject in inputs:
        # determine subj_id and visit
        # TODO: check if inputs follow path pattern "input/[subj_id]/[visit].nii.gz,..."

        subj_id = PurePath(str(subject)).parents[0].name
        visit = PurePath(str(subject)).name.rsplit(image_ext)[0]

        if not is_non_zero_file(f"output/{subj_id}/COMPLETE"):
            if len(subject) == 1:
                # cross section?
                command = f"mkdir -p output/{subj_id} && cp -p {subject[0]} output/{subj_id}/template0.nii.gz \
                          && ImageMath 3 output/{subj_id}/0-0GenericAffine.mat MakeAffineTransform 1 \
                          && CreateImage 3 {subject[0]} output/{subj_id}/0-1Warp.nii.gz 1 \
                          && CreateDisplacementField 3 1 "
                command += f"output/{subj_id}/0-1Warp.nii.gz " * 3 + \
                    f"output/{subj_id}/0-1InverseWarp.nii.gz && CreateDisplacementField 3 1 " + \
                    f"output/{subj_id}/0-1Warp.nii.gz " * 4
            else:
                # Base command, default: bootstrap model builds with rigid pre-alignment w/o update
                command = f"{args.modelbuild_command} -d 3 -i {args.model_iterations} -o output/{subj_id}/{visit}_ \
                            -a {args.average_type} -e {args.float} -f {args.reg_shrinks} -g {args.gradient_step} \
                            -m {args.metric} -n {int(args.N4)} -q {args.reg_iterations} -r 1 -s {args.reg_smoothing} \
                            -t {args.transform} -y 0 \
                            -c {args.cluster_type} -u {args.walltime} -v {args.memory_request}"

                if args.rigid_model_target:
                    command += f"-z {args.rigid_model_target} "
                command += " ".join(subject)

            command += f" && echo DONE > output/{subj_id}/COMPLETE"
            cmds.append(command)

        images.append(subject + f" output/{subj_id}/template0.nii.gz")

    # TODO: add the ability to limit the number of commands submitted

    if len(cmds):
        pool = threading.ThreadPool(nodes=args.local_threads)

        print(f"{script_name}: Running {len(cmds)} Level1 Model builds")
        for item in tqdm.tqdm(pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), cmds),
                              total=len(cmds)):
            results.append(item)
        if not args.dry_run:
            for i, result in enumerate(results):
                subj_id =
                visit =
                with open(f"output/{subj_id}/{visit}.log", "wb") as l1_log:
                    l1_log.write(cmds[i].encode())
                    l1_log.write(result.stdout)

        # Completely destroy the pool so that pathos doesn't reuse
        pool.close()
        pool.clear()
    second_level(images, args, bool(1))


def secondlevel(inputs, args, secondlevel=False):
    mkdirp("output/jacobians/overall", args.dry_run)
    mkdirp("output/compositewarps/secondlevel", args.dry_run)
    mkdirp("output/jacobians/common_space", args.dry_run)
    mkdirp("output/compositewarps/groupwise", args.dry_run)
    mkdirp("output/jacobians/resampled", args.dry_run)
    mkdirp("output/jacobians/groupwise", args.dry_run)
    mkdirp("output/jacobians/common_space/overall", args.dry_run)
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
        command += (
            f"-c {args.cluster_type} -a {args.average_type} "
            f"-e {args.float} -g {args.gradient_step} "
            f"-i {args.model_iterations} "
            f"-n {int(args.N4) if (not secondlevel) else '0'} -m {args.metric} "
            f"-t {args.transform} -u {args.walltime} -v {args.memory_request} "
        )
        # Registrations Setup
        command += (
            f"-q {args.reg_iterations} "
            f"-f {args.reg_shrinks} "
            f"-s {args.reg_smoothing} "
        )
        if args.rigid_model_target:
            command += f"-z {args.rigid_model_target} "
        if not secondlevel:
          command += f"{args.input} "
        else:
          command += " ".join(input_images)
        command += " && echo DONE > output/secondlevel/COMPLETE"
        print("twolevel_dbm.py: Running Second-Level Modelbuild")
        results = run_command(command, args.dry_run, args.verbose)
        # Here we should add the ability to limit the number of commands submitted
        if not args.dry_run:
            with open("output/secondlevel/secondlevel.log", "wb") as logfile:
                logfile.write(command.encode())
                logfile.write(results.stdout)

    pool = threading.ThreadPool(nodes=args.local_threads)

    if args.skip_dbm:
        print("twolevel_dbm.py: Skipping generation of DBM outputs")
        print("twolevel_dbm.py: Pipeline Complete")
        sys.exit(0)


    # Create mask for delin
    run_command(
        "ThresholdImage 3 output/secondlevel/secondlevel_template0.nii.gz "
        "output/secondlevel/secondlevel_otsumask.nii.gz Otsu 1",
        args.dry_run,
        args.verbose,
    )
    # Register final model to common space
    if (
        not is_non_zero_file("output/secondlevel/template0_common_space_COMPLETE")
        and args.resample_to_common_space
    ):
        print("twolevel_dbm.py: Registering final modelbuild to target common space")
        run_command(
            f"antsRegistrationSyN.sh -d 3 -f {args.resample_to_common_space} "
            "-m output/secondlevel/secondlevel_template0.nii.gz "
            "-o output/secondlevel/template0_common_space_",
            args.dry_run,
            args.verbose,
        )
        run_command(
            "echo DONE > output/secondlevel/template0_common_space_COMPLETE",
            args.dry_run,
            args.verbose,
        )

    print("twolevel_dbm.py: Processing Second-Level DBM outputs")
    # Loop over input file warp fields to produce delin
    jacobians = list()
    for i, subject in enumerate(tqdm.tqdm(input_images), start=0):
        subjectname = pathlib.Path(subject).name.rsplit(".nii")[0]
        if not is_non_zero_file("output/compositewarps/secondlevel/COMPLETE"):
            commands = list()
            # Compute delin
            run_command(
                f"ANTSUseDeformationFieldToGetAffineTransform "
                f"output/secondlevel/secondlevel_{subjectname}{i}1Warp.nii.gz 0.25 "
                f"affine output/compositewarps/secondlevel/{subjectname}_delin.mat "
                f"output/secondlevel/secondlevel_otsumask.nii.gz",
                args.dry_run,
                args.verbose,
            )

            # Create composite field of delin
            commands.append(
                f"antsApplyTransforms -d 3 --verbose "
                f"-t [output/compositewarps/secondlevel/{subjectname}_delin.mat,1] "
                f"-r output/secondlevel/secondlevel_template0.nii.gz "
                f"-o [output/compositewarps/secondlevel/{subjectname}_delin.nii.gz,1]"
            )

            # Create composite field of affine
            commands.append(
                f"antsApplyTransforms -d 3 --verbose "
                f"-t output/secondlevel/secondlevel_{subjectname}{i}0GenericAffine.mat "
                f"-r output/secondlevel/secondlevel_template0.nii.gz "
                f"-o [output/compositewarps/secondlevel/{subjectname}_affine.nii.gz,1]"
            )

            pool.map(lambda x: run_command(x, args.dry_run, args.verbose), commands)
            commands = list()

            # Generate jacobians of composite affine fields and nonlinear fields
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/secondlevel/secondlevel_{subjectname}{i}1Warp.nii.gz "
                f"output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz 1 1"
            )
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel/{subjectname}_delin.nii.gz "
                f"output/jacobians/overall/secondlevel_{subjectname}_delin.nii.gz 1 1"
            )
            commands.append(
                f"CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel/{subjectname}_affine.nii.gz "
                f"output/jacobians/overall/secondlevel_{subjectname}_affine.nii.gz 1 1"
            )

            pool.map(lambda x: run_command(x, args.dry_run, args.verbose), commands)
            commands = list()

            commands.append(
                f"ImageMath 3 output/jacobians/overall/secondlevel_{subjectname}_relative.nii.gz "
                f"+ output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz "
                f"output/jacobians/overall/secondlevel_{subjectname}_delin.nii.gz"
            )
            commands.append(
                f"ImageMath 3 output/jacobians/overall/secondlevel_{subjectname}_absolute.nii.gz "
                f"+ output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz "
                f"output/jacobians/overall/secondlevel_{subjectname}_affine.nii.gz"
            )
            pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), commands)
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

    run_command(
        "echo DONE > output/compositewarps/secondlevel/COMPLETE",
        args.dry_run,
        args.verbose,
    )

    if not secondlevel and args.resample_to_common_space:
        for i, subject in enumerate(tqdm.tqdm(input_images), start=0):
            subjectname = pathlib.Path(subject).name.rsplit(".nii")[0]
            if not is_non_zero_file("output/jacobians/common_space/COMPLETE"):
                commands.append(
                    f"antsApplyTransforms -d 3 --verbose "
                    f"-i output/jacobians/overall/secondlevel_{subjectname}_relative.nii.gz "
                    f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                    f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} "
                    f"-o output/jacobians/common_space/secondlevel_{subjectname}_relative.nii.gz"
                )
                commands.append(
                    f"antsApplyTransforms -d 3 --verbose "
                    f"-i output/jacobians/overall/secondlevel_{subjectname}_absolute.nii.gz "
                    f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                    f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} "
                    f"-o output/jacobians/common_space/secondlevel_{subjectname}_absolute.nii.gz"
                )
                commands.append(
                    f"antsApplyTransforms -d 3 --verbose "
                    f"-i output/jacobians/overall/secondlevel_{subjectname}_nlin.nii.gz "
                    f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                    f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                    f"-r {args.resample_to_common_space} "
                    f"-o output/jacobians/common_space/secondlevel_{subjectname}_nlin.nii.gz"
                )

                pool.uimap(
                    lambda x: run_command(x, args.dry_run, args.verbose), commands
                )
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

    if not secondlevel and args.resample_to_common_space:
        run_command(
            "echo DONE > output/jacobians/common_space/COMPLETE",
            args.dry_run,
            args.verbose,
        )

    if secondlevel:
        print("twolevel_dbm.py: Processing First-Level DBM Outputs")
        for subjectnum, row in enumerate(
            tqdm.tqdm([line[:-1] for line in inputs]), start=0
        ):
            if not is_non_zero_file("output/jacobians/resampled/COMPLETE"):
                # Make a mask per subject
                run_command(
                    f"ThresholdImage 3 output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz "
                    f"output/subject{subjectnum}/subject{subjectnum}_otsumask.nii.gz Otsu 1",
                    args.dry_run,
                    args.verbose,
                )
                for scannum, scan in enumerate(row, start=0):
                    commands = list()
                    scanname = pathlib.Path(scan).name.rsplit(".nii")[0]
                    # Estimate affine residual from nonlinear and create composite warp and jacobian field
                    run_command(
                        f"ANTSUseDeformationFieldToGetAffineTransform "
                        f"output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}1Warp.nii.gz 0.25 "
                        f"affine output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.mat "
                        f"output/subject{subjectnum}/subject{subjectnum}_otsumask.nii.gz",
                        args.dry_run,
                        args.verbose,
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 --verbose "
                        f"-t [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.mat,1] "
                        f"-r output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz "
                        f"-o [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz,1]"
                    )
                    # Create composite warp field from affine
                    commands.append(
                        f"antsApplyTransforms -d 3 --verbose "
                        f"-t output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}0GenericAffine.mat "
                        f"-r output/subject{subjectnum}/subject{subjectnum}_template0.nii.gz "
                        f"-o [output/compositewarps/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz,1]"
                    )

                    pool.map(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
                    commands = list()

                    # Create jacobian images from nlin and composite warp fields
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/subject{subjectnum}/subject{subjectnum}_{scanname}{scannum}1Warp.nii.gz "
                        f"output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz 1 1"
                    )
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/compositewarps/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz "
                        f"output/jacobians/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz 1 1"
                    )
                    commands.append(
                        f"CreateJacobianDeterminantImage 3 output/compositewarps/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz "
                        f"output/jacobians/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz 1 1"
                    )

                    pool.map(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
                    commands = list()

                    # Create relative and absolute jacobians by adding affine/delin jacobians
                    commands.append(
                        f"ImageMath 3 output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz "
                        f"+ output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz "
                        f"output/jacobians/groupwise/subject{subjectnum}_{scanname}_delin.nii.gz"
                    )
                    commands.append(
                        f"ImageMath 3 output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz "
                        f"+ output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz "
                        f"output/jacobians/groupwise/subject{subjectnum}_{scanname}_affine.nii.gz"
                    )

                    pool.map(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
                    commands = list()

                    # Resample jacobian to common space
                    commands.append(
                        f"antsApplyTransforms -d 3 --verbose "
                        f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz "
                        f"-o output/jacobians/resampled/subject{subjectnum}_{scanname}_relative.nii.gz"
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 --verbose "
                        f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz "
                        f"-o output/jacobians/resampled/subject{subjectnum}_{scanname}_absolute.nii.gz"
                    )
                    commands.append(
                        f"antsApplyTransforms -d 3 --verbose "
                        f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                        f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                        f"-r output/secondlevel/secondlevel_template0.nii.gz "
                        f"-o output/jacobians/resampled/subject{subjectnum}_{scanname}_nlin.nii.gz"
                    )

                    pool.map(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
                    commands = list()

                    commands.append(
                        f"ImageMath 3 output/jacobians/overall/subject{subjectnum}_{scanname}_relative.nii.gz + "
                        f"output/jacobians/resampled/subject{subjectnum}_{scanname}_relative.nii.gz "
                        f"output/jacobians/overall/secondlevel_subject{subjectnum}_template0_relative.nii.gz"
                    )

                    commands.append(
                        f"ImageMath 3 output/jacobians/overall/subject{subjectnum}_{scanname}_absolute.nii.gz + "
                        f"output/jacobians/resampled/subject{subjectnum}_{scanname}_absolute.nii.gz "
                        f"output/jacobians/overall/secondlevel_subject{subjectnum}_template0_absolute.nii.gz"
                    )

                    commands.append(
                        f"ImageMath 3 output/jacobians/overall/subject{subjectnum}_{scanname}_nlin.nii.gz + "
                        f"output/jacobians/resampled/subject{subjectnum}_{scanname}_nlin.nii.gz "
                        f"output/jacobians/overall/secondlevel_subject{subjectnum}_template0_nlin.nii.gz"
                    )

                    pool.uimap(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
                    commands = list()

                    if args.resample_to_common_space:
                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_relative.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/subject{subjectnum}_{scanname}_relative.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_absolute.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/groupwise/subject{subjectnum}_{scanname}_nlin.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}1Warp.nii.gz "
                            f"-t output/secondlevel/secondlevel_subject{subjectnum}_template0{subjectnum}0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/subject{subjectnum}_{scanname}_nlin.nii.gz"
                        )

                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/overall/subject{subjectnum}_{scanname}_relative.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_relative.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/overall/subject{subjectnum}_{scanname}_absolute.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )
                        commands.append(
                            f"antsApplyTransforms -d 3 --verbose "
                            f"-i output/jacobians/overall/subject{subjectnum}_{scanname}_nlin.nii.gz "
                            f"-t output/secondlevel/template0_common_space_1Warp.nii.gz "
                            f"-t output/secondlevel/template0_common_space_0GenericAffine.mat "
                            f"-r {args.resample_to_common_space} "
                            f"-o output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )

                    pool.uimap(
                        lambda x: run_command(x, args.dry_run, args.verbose), commands
                    )
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

                    jacobians.append(
                        f"output/jacobians/overall/subject{subjectnum}_{scanname}_relative.nii.gz"
                    )
                    jacobians.append(
                        f"output/jacobians/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                    )
                    jacobians.append(
                        f"output/jacobians/overall/subject{subjectnum}_{scanname}_nlin.nii.gz"
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
                        jacobians.append(
                            f"output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )
                        jacobians.append(
                            f"output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )
                        jacobians.append(
                            f"output/jacobians/common_space/overall/subject{subjectnum}_{scanname}_absolute.nii.gz"
                        )

    commands = list()
    print("twolevel_dbm.py: Blurring Jacobians")
    for jacobian in jacobians:
        for blur in args.jacobian_sigmas:
            commands.append(
                f"SmoothImage 3 {jacobian} {blur} {jacobian.rsplit('.nii')[0]}_smooth{blur}.nii.gz 1 0"
            )
    for _ in tqdm.tqdm(
        pool.uimap(lambda x: run_command(x, args.dry_run, args.verbose), commands),
        total=len(commands),
    ):
        pass

    pool.close()
    pool.clear()
    run_command(
        "echo DONE > output/jacobians/resampled/COMPLETE",
        args.dry_run,
        args.verbose,
    )
    print("twolevel_dbm.py: Pipeline Complete")
    sys.exit(0)


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

    dbm_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description="This pipeline performs one or two level model building on files "
                                                     "using antsMultivariateTemplateConstruction2.sh and generates "
                                                     "smoothed jacobian determinent fields suitable for "
                                                     "deformation based morphomometry (DBM) analysis.")

    dbm_parser.add_argument("type", choices=["1level", "2level"], help="Type of DBM processing to run")

    dbm_parser.add_argument("input",
                            help="Input CSV file for DBM.\n1level: a single column\n \
                                2level: each each row constructs a level1 model followed by a level2 model \
                                of the resulting level1 averages. File paths must be absolute.")

    dbm_parser.add_argument("--jacobian-sigmas", nargs="+", type=float,
                            help="List of smoothing sigmas for final output\n \
                                defaults: FWHM of twice the finest resolution input or rigid model target if provided.")

    dbm_parser.add_argument("--rigid-model-target",
                            help="Target image for rigid registration of the level2, \
                                 otherwise start with unbiased average")

    dbm_parser.add_argument("-t", "--resample-to-common-space",
                            help="Target atlas space to resample to jacobians to after unbiased model build, \
                                typically an MNI model, triggers a registration to this target")

    dbm_parser.add_argument("--skip-dbm", action="store_true", help="Skip generating DBM outputs")

    dbm_parser.add_argument("-d", "--dry-run", action="store_true", help="Don't run commands, instead print to stdout")

    dbm_parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose about what is going on")

    advanced = dbm_parser.add_argument_group("advanced options")
    advanced.add_argument("--N4", "--no-N4", action=BooleanAction, dest="N4", default=False,
                          help="Run N4BiasFieldCorrection during model build on input files")

    advanced.add_argument("--metric", default="CC[4]",
                          help="Specify metric used for non-linear stages")

    advanced.add_argument("--transform", default="SyN",
                          choices=["SyN", "BSplineSyN", "Affine", "Rigid",
                                   "TimeVaryingVelocityField", "TimeVaryingBSplineVelocityField"],
                          help="Transformation type to use")

    advanced.add_argument("-i", "--reg-iterations", default="100x100x70x20",
                          help="Max iterations for non-linear stages")

    advanced.add_argument("--reg-smoothing", default="3x2x1x0",
                          help="Smoothing sigmas for non-linear stages")

    advanced.add_argument("--reg-shrinks", default="6x4x2x1",
                          help="Shrink factors for non-linear stages")

    advanced.add_argument("--float", "--no-float", action=BooleanAction, dest="float", default=1,
                          help="Run registration with float (32 bit) or double (64 bit) values")

    advanced.add_argument("--average-type", default="normmean", choices=["mean", "normmean", "median"],
                          help="Type of average to use, default: normalized mean")

    advanced.add_argument("--gradient-step", default=0.25, type=float,
                          help="Gradient step size at each iteration")

    advanced.add_argument("--model-iterations", default=4, type=int,
                          help="Number of registration and average")

    advanced.add_argument("--modelbuild-command", default="antsMultivariateTemplateConstruction2.sh",
                          help="Command for model build, \
                            arguments must be same as in antsMultivariateTemplateConstruction2.sh")

    cluster = dbm_parser.add_argument_group("cluster options")
    cluster.add_argument("-c", "--cluster-type", default="local", choices=["local", "sge", "pbs", "slurm"],
                         help="Type of cluster for job submission")

    cluster.add_argument("--walltime", default="4:00:00",
                         help="Specify requested time per pairwise registration")

    cluster.add_argument("--memory-request", default="8gb",
                         help="Specify requested memory per pairwise registration")

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
        exit(f"{script_name} error: iterations, shrinks and smoothing do not match in length")

    if not which(args.modelbuild_command):
        exit(f"{args.modelbuild_command} command not found")

    inputs = read_csv(args.input)
    setup_and_check_inputs(inputs, args)

    if args.type == "2level":
        first_level(inputs, args)
    else:
        second_level(inputs, args, bool(0))


if __name__ == "__main__":
    main()

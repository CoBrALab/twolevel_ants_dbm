#!/usr/bin/env python3

import argparse
import csv
import os
import pathlib  # Better path manipulation
import shlex
import subprocess
import sys

import pathos.multiprocessing as multiprocessing  # Better multiprocessing
import tqdm  # Progress bar


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def run_command(command, dryrun, verbose=False):
    if dryrun:
        print(command)
        fakereturn = subprocess.CompletedProcess
        fakereturn.stdout = "".encode()
        return (fakereturn)
    else:
        if verbose:
            print(f"twolevel_dbm.py: running command {command}")
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            shell=True)
        return (result)


def mkdirp(*p):
    """Like mkdir -p"""
    path = os.path.join(*p)

    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == os.errno.EEXIST:
            pass
        else:
            raise
    return (path)


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
            if (not is_non_zero_file(file)):
                sys.exit(f"twolevel_dbm error: File {file} does "
                         + "not exist or is zero size")

    # Find minimum resolution of input files unless blurs are set, or rigid
    # model is provided
    if (not args.jacobian_sigmas):
        if (args.rigid_model_target):
            if args.dry_run:
                run_command("PrintHeader {} 1".format(args.rigid_model_target),
                            args.dry_run)
            else:
                args.jacobian_sigmas = [
                    2 * min(
                        map(
                            abs,
                            map(
                                float,
                                run_command(
                                    "PrintHeader {} 1".format(
                                        args.rigid_model_target), args.dry_run)
                                .stdout.decode('utf8').split('x'))))
                ]
        else:
            minres = 1e6
            for row in inputs:
                for file in row:
                    if args.dry_run:
                        run_command("PrintHeader {} 1".format(file),
                                    args.dry_run)
                    else:
                        curres = min(
                            map(
                                abs,
                                map(
                                    float,
                                    run_command(
                                        "PrintHeader {} 1".format(file),
                                        args.dry_run).stdout.decode('utf8')
                                    .split('x'))))
                        minres = curres if curres < minres else curres
            args.jacobian_sigmas = [2 * minres]


def firstlevel(inputs, args):
    commands = list()
    imagelist = list()
    for i, subject in enumerate(inputs, start=0):
        if not is_non_zero_file("output/subject{}/COMPLETE".format(i)):
            # Base command
            command = "antsMultivariateTemplateConstruction2.sh -d 3 "
            # Setup directory and naming
            command += "-o output/subject{}/subject{}_ ".format(i, i)
            # Defaults to bootstrap modelbuilds with rigid prealignmnet,
            # no rigid update
            command += "-r 1 -l 1 -y 0 "
            # Model build setup
            command += "-c {} -a {} -e {} -g {} -i {} -n {} -m {} -t {} ".format(
                args.cluster_type, args.average_type, args.float,
                args.gradient_step, args.model_iterations, int(args.N4),
                args.metric, args.transform)
            # Registrations Setup
            command += "-q {} -f {} -s {} ".format(
                args.reg_iterations, args.reg_shrinks, args.reg_smoothing)
            if args.rigid_model_target:
                command += "-z {} ".format(args.rigid_model_target)
            command += " ".join(subject)
            command += " && echo DONE > output/subject{}/COMPLETE".format(i)
            commands.append(command)

        imagelist.append(
            subject
            + ["output/subject{0}/subject{0}_template0.nii.gz".format(i)])
    # Here we should add the ability to limit the number of commands submitted
    results = list()
    if len(commands) > 0:
        if args.cluster_type != 0:
            pool = multiprocessing.Pool()
        else:
            pool = multiprocessing.Pool(processes=args.local_threads)

        for item in tqdm.tqdm(
                pool.imap_unordered(lambda x: run_command(x, args.dry_run),
                                    commands),
                total=len(commands)):
            results.append(item)
        if not args.dry_run:
            for i, subject in enumerate(results, start=0):
                with open('output/subject{0}/subject{0}.log'.format(i),
                          'wb') as logfile:
                    logfile.write(subject.stdout)
        pool.close()
    secondlevel(imagelist, args, secondlevel=True)


def secondlevel(inputs, args, secondlevel=False):
    outputs = list()
    if secondlevel:
        input_images = [row[-1] for row in inputs]
    else:
        input_images = [val for sublist in inputs for val in sublist]
    if not is_non_zero_file("output/secondlevel/COMPLETE"):
        # Base command
        command = "antsMultivariateTemplateConstruction2.sh -d 3 "
        # Setup directory and naming
        command += "-o output/secondlevel/secondlevel_ "
        # Defaults to bootstrap modelbuilds with rigid prealignmnet, no rigid
        # update
        command += """-r 1 -l 1 -y 0 """
        # Model build setup
        command += "-c {} -a {} -e {} -g {} -i {} -n {} -m {} -t {} ".format(
            args.cluster_type, args.average_type, args.float, args.gradient_step,
            args.model_iterations, (not secondlevel) and int(args.N4) or "0",
            args.metric, args.transform)
        # Registrations Setup
        command += "-q {} -f {} -s {} ".format(
            args.reg_iterations, args.reg_shrinks, args.reg_smoothing)
        if args.rigid_model_target:
            command += "-z {} ".format(args.rigid_model_target)
        command += " ".join(input_images)
        command += " && echo DONE > output/secondlevel/COMPLETE"
        results = run_command(command, args.dry_run)
        # Here we should add the ability to limit the number of commands submitted
        if not args.dry_run:
            with open('output/secondlevel/secondlevel.log', 'wb') as logfile:
                logfile.write(results.stdout)

    # Create mask for delin
    mkdirp("output/jacobians/resampled")
    mkdirp("output/jacobians/overall")
    mkdirp("output/compositewarps")
    run_command(
        "ThresholdImage 3 output/secondlevel/secondlevel_template0.nii.gz output/secondlevel/secondlevel_otsumask.nii.gz Otsu 1",
        args.dry_run)
    # Loop over input file warp fields to produce delin
    jacobians = list()
    for i, subject in enumerate(input_images, start=0):
        subjectname = pathlib.Path(subject).name.rsplit('.nii')[0]
        print(f"Processing subject {subject} DBM outputs")
        # Compute delin
        run_command(
            "ANTSUseDeformationFieldToGetAffineTransform output/secondlevel/secondlevel_{1}{0}1InverseWarp.nii.gz 0.25 "
            "affine output/compositewarps/secondlevel_{1}_delin.mat output/secondlevel/secondlevel_otsumask.nii.gz".format(
                i, subjectname), args.dry_run)
        # Remove the rigid components
        run_command(
            "AverageAffineTransformNoRigid 3 output/compositewarps/secondlevel_{0}_delin.mat output/compositewarps/secondlevel_{0}_delin.mat output/compositewarps/secondlevel_{0}_delin.mat".format(
                subjectname),
            args.dry_run)
        # Apply delin to create composite relative
        run_command(
            "antsApplyTransforms -d 3 -t output/secondlevel/secondlevel_{1}{0}1InverseWarp.nii.gz -t [output/compositewarps/secondlevel_{1}_delin.mat,1] "
            "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/secondlevel_{1}_relative.nii.gz,1] ".format(
                i, subjectname), args.dry_run)
        # Remove rigid component of affine
        run_command(
            "AverageAffineTransformNoRigid 3 output/secondlevel/secondlevel_{1}{0}0GenericAffine_norigid.mat output/secondlevel/secondlevel_{1}{0}0GenericAffine.mat output/secondlevel/secondlevel_{1}{0}0GenericAffine.mat".format(
                i,
                subjectname),
            args.dry_run)
        # Create composite of absolute
        run_command(
            "antsApplyTransforms -d 3 -t [output/secondlevel/secondlevel_{1}{0}0GenericAffine_norigid.mat,1] -t output/secondlevel/secondlevel_{1}{0}1InverseWarp.nii.gz "
            "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/secondlevel_{1}_absolute.nii.gz,1] ".format(
                i, subjectname), args.dry_run)
        run_command(
            "CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel_{0}_relative.nii.gz output/jacobians/overall/secondlevel_{0}_relative.nii.gz 1 1".format(
                subjectname),
            args.dry_run)
        run_command(
            "CreateJacobianDeterminantImage 3 output/compositewarps/secondlevel_{0}_absolute.nii.gz output/jacobians/overall/secondlevel_{0}_absolute.nii.gz 1 1".format(
                subjectname),
            args.dry_run)
        jacobians.append(
            "output/jacobians/overall/secondlevel_{}_relative.nii.gz".format(
                subjectname))
        jacobians.append(
            "output/jacobians/overall/secondlevel_{}_absolute.nii.gz".format(
                subjectname))

    if secondlevel:
        for subject, row in enumerate([line[:-1] for line in inputs], start=0):
            # Make a mask per subject
            run_command(
                "ThresholdImage 3 output/subject{0}/subject{0}_template0.nii.gz output/subject{0}/subject{0}_otsumask.nii.gz Otsu 1".
                format(subject), args.dry_run)
            for i, scan in enumerate(row, start=0):
                scanname = pathlib.Path(scan).name.rsplit('.nii')[0]
                print(f"Processing scan {scanname}")
                # Estimate affine residual from nonlinear
                run_command(
                    "ANTSUseDeformationFieldToGetAffineTransform output/subject{0}/subject{0}_{2}{1}1InverseWarp.nii.gz 0.25 "
                    "affine output/compositewarps/subject{0}_{2}_delin.mat output/subject{0}/subject{0}_otsumask.nii.gz".format(
                        subject, i, scanname), args.dry_run)
                # Remove the rigid component from the affine
                run_command(
                    "AverageAffineTransformNoRigid 3 output/compositewarps/subject{0}_{2}_delin.mat output/compositewarps/subject{0}_{2}_delin.mat output/compositewarps/subject{0}_{2}_delin.mat".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                # Create composite nonlinear field using the affine to remove
                # affine residuals
                run_command(
                    "antsApplyTransforms -d 3 -t output/subject{0}/subject{0}_{2}{1}1InverseWarp.nii.gz -t [output/compositewarps/subject{0}_{2}_delin.mat,1] "
                    "-r output/subject{0}/subject{0}_template0.nii.gz --verbose -o [output/compositewarps/subject{0}_{2}_relative.nii.gz,1] ".format(
                        subject, i, scanname), args.dry_run)
                # Remove rigid from absolute affine
                run_command(
                    "AverageAffineTransformNoRigid 3 output/subject{0}/subject{0}_{2}{1}0GenericAffine_norigid.mat output/subject{0}/subject{0}_{2}{1}0GenericAffine.mat output/subject{0}/subject{0}_{2}{1}0GenericAffine.mat".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                # Create composite absoloute warp
                run_command(
                    "antsApplyTransforms -d 3 -t [output/subject{0}/subject{0}_{2}{1}0GenericAffine_norigid.mat,1] -t output/subject{0}/subject{0}_{2}{1}1InverseWarp.nii.gz "
                    "-r output/subject{0}/subject{0}_template0.nii.gz --verbose -o [output/compositewarps/subject{0}_{2}_absolute.nii.gz,1]".format(
                        subject, i, scanname), args.dry_run)
                # Resample composite fields into common space
                run_command(
                    "antsApplyTransforms -d 3 -e 1 -i output/compositewarps/subject{0}_{2}_relative.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}0GenericAffine.mat "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/compositewarps/subject{0}_{2}_relative_commonspace.nii.gz".format(
                        subject, i, scanname), args.dry_run)
                run_command(
                    "antsApplyTransforms -d 3 -e 1 -i output/compositewarps/subject{0}_{2}_absolute.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}0GenericAffine.mat "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/compositewarps/subject{0}_{2}_absolute_commonspace.nii.gz".format(
                        subject, i, scanname), args.dry_run)
                # Create composite fields of overall transform
                run_command(
                    "antsApplyTransforms -d 3 -t output/compositewarps/subject{0}_{2}_relative_commonspace.nii.gz -t output/compositewarps/secondlevel_subject{1}_template0_relative.nii.gz "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/subject{0}_{2}_relative_commonspace_overall.nii.gz,1]".format(
                        subject, i, scanname), args.dry_run)
                run_command(
                    "antsApplyTransforms -d 3 -t output/compositewarps/subject{0}_{2}_absolute_commonspace.nii.gz -t output/compositewarps/secondlevel_subject{1}_template0_absolute.nii.gz "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o [output/compositewarps/subject{0}_{2}_absolute_commonspace_overall.nii.gz,1]".format(
                        subject, i, scanname), args.dry_run)
                # Create jacobians of overall composite warp fields
                run_command(
                    "CreateJacobianDeterminantImage 3 output/compositewarps/subject{0}_{2}_relative_commonspace_overall.nii.gz output/jacobians/overall/subject{0}_{2}_relative.nii.gz 1 1".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                run_command(
                    "CreateJacobianDeterminantImage 3 output/compositewarps/subject{0}_{2}_absolute_commonspace_overall.nii.gz output/jacobians/overall/subject{0}_{2}_absolute.nii.gz 1 1".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                jacobians.append(
                    "output/jacobians/overall/subject{0}_{2}_relative.nii.gz".format(subject, i,
                                                                                     scanname))
                jacobians.append(
                    "output/jacobians/overall/subject{0}_{2}_absolute.nii.gz".format(subject, i,
                                                                                     scanname))
                # Create jacobian images from two composite fields in subject
                # space
                run_command(
                    "CreateJacobianDeterminantImage 3 output/compositewarps/subject{0}_{2}_relative.nii.gz output/jacobians/subject{0}_{2}_relative.nii.gz 1 1".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                run_command(
                    "CreateJacobianDeterminantImage 3 output/compositewarps/subject{0}_{2}_absolute.nii.gz output/jacobians/subject{0}_{2}_absolute.nii.gz 1 1".format(
                        subject,
                        i,
                        scanname),
                    args.dry_run)
                # Resample jacobian to common space
                run_command(
                    "antsApplyTransforms -d 3 -i output/jacobians/subject{0}_{2}_relative.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}0GenericAffine.mat "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/jacobians/resampled/subject{0}_{2}_relative.nii.gz".format(
                        subject, i, scanname), args.dry_run)
                run_command(
                    "antsApplyTransforms -d 3 -i output/jacobians/subject{0}_{2}_absolute.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}1Warp.nii.gz -t output/secondlevel/secondlevel_subject{0}_template0{0}0GenericAffine.mat "
                    "-r output/secondlevel/secondlevel_template0.nii.gz --verbose -o output/jacobians/resampled/subject{0}_{2}_absolute.nii.gz".format(
                        subject, i, scanname), args.dry_run)
                # Append jacobians to list
                jacobians.append(
                    "output/jacobians/resampled/subject{0}_{2}_relative.nii.gz".format(subject, i,
                                                                                       scanname))
                jacobians.append(
                    "output/jacobians/resampled/subject{0}_{2}_absolute.nii.gz".format(subject, i,
                                                                                       scanname))
    #outputs = []
    #outputs.append(["overall_jacobian", "relative_jacobian", "blur", "grouping", "input_file"])
    for jacobian in jacobians:
        for blur in args.jacobian_sigmas:
            run_command(
                f"SmoothImage 3 {jacobian} {blur} {jacobian.split('.nii.gz')[0]}_smooth{blur}.nii.gz 1 0",
                args.dry_run)


def read_csv(inputfile):
    inputs = []
    with open(inputfile, newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row in reader:
                inputs.append(list(filter(None, row)))
        except csv.Error as e:
            sys.exit(
                'malformed csv: file {}, line {}: {}'.format(
                    inputfile, reader.line_num, e))
    return (inputs)


def main():

    # Stolen from https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
    # Gives me --option --no-option paried control
    class BooleanAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            super(BooleanAction, self).__init__(
                option_strings, dest, nargs=0, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, False
                    if option_string.startswith('--no') else True)

    parser = argparse.ArgumentParser(
        description="""This pipeline performs one or
        two level model building on files using antsMultivariateTemplateConstruction2.sh
        and generates smoothed jacobian determinent fields suitable for deformation
        based morphomometry (DBM) analysis.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help="""Input CSV file for DBM, for
        1level mode, a single column, for 2level, each each row constructs a first level model
        followed by a second level model of the resulting first level averages. File paths must
        be absolute.""")
    parser.add_argument(
        '--jacobian-sigmas',
        nargs='+',
        type=float,
        help="""List of smoothing
        sigmas used for final output, defaults to 2x finest resolution input file or
        rigid model target if provided.""")
    parser.add_argument(
        '--rigid-model-target',
        help="""Target file to use for rigid
        registration of the second level, otherwise unbiased average to start"""
    )
    parser.add_argument(
        '--resample-to-common-space',
        help="""Target space to resample
        jacobians to after unbiased model build, typically an MNI model, triggers a
        registration to this target""")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Don't run commands, instead print to stdout")
    parser.add_argument(
        "type",
        choices=['1level', '2level'],
        help="""What type of DBM processing to run on input file, see --input
            for details on how to format CSV file for different types.""")

    advanced = parser.add_argument_group('advanced options')
    advanced.add_argument(
        '--N4',
        '--no-N4',
        action=BooleanAction,
        dest='N4',
        default=False,
        help="Run N4BiasFieldCorrection during model build on input files.")
    advanced.add_argument(
        '--metric',
        default="CC[4]",
        help="Specify metric used for non-linear template stages")
    advanced.add_argument(
        '--transform',
        default="SyN",
        choices=[
            'SyN', 'BSplineSyN', 'TimeVaryingVelocityField',
            'TimeVaryingBSplineVelocityField', 'Affine', 'Rigid'
        ],
        help="Transformation type used during model build")
    advanced.add_argument(
        '--reg-iterations',
        default="100x100x70x20",
        help="Max iterations for non-linear stages")
    advanced.add_argument(
        '--reg-smoothing',
        default="3x2x1x0",
        help="Smoothing sigmas for non-linear stages")
    advanced.add_argument(
        '--reg-shrinks',
        default="6x4x2x1",
        help="Shrink factors for non-linear stages")
    advanced.add_argument(
        '--float',
        '--no-float',
        action=BooleanAction,
        dest='float',
        default=True,
        help="Run registration with float (32 bit) or double (64 bit) values")
    advanced.add_argument(
        '--average-type', default='normmean',
        choices=['mean', 'normmean', 'median'],
        help="Type of average used during model build, default normalized mean")
    advanced.add_argument(
        '--gradient-step',
        default=0.25,
        type=float,
        help="Gradient step size at each iteration during model build")
    advanced.add_argument(
        '--model-iterations',
        default=3,
        type=int,
        help="How many registration and average rounds to do")

    cluster = parser.add_argument_group('cluster options')
    cluster.add_argument(
        '--cluster-type',
        default="local",
        choices=["local", "sge", "pbs", "slurm"],
        help="Choose the type of cluster system to submit jobs to")
    cluster.add_argument(
        '--walltime',
        default="20:00:00",
        help="""Option for job submission
        specifying requested time per pairwise registration.""")
    cluster.add_argument(
        '--memory-request',
        default="8gb",
        help="""Option for job submission
        specifying requested memory per pairwise registration.""")
    cluster.add_argument(
        '--local-threads',
        '-j',
        type=int,
        default=multiprocessing.cpu_count(),
        help="""For local execution, how many subject-wise modelbuilds to run in parallel,
        defaults to number of CPUs""")

    args = parser.parse_args()
    # Convert inputs into values for model build command
    args.float = int(args.float)
    clusterchoices = {"local": 0, "sge": 1, "pbs": 4, "slurm": 5}
    args.cluster_type = clusterchoices[args.cluster_type]
    averagechoices = {"mean": 0, "normmean": 1, "median": 2}
    args.average_type = averagechoices[args.average_type]

    if (not (len(args.reg_iterations.split('x')) == len(
            args.reg_shrinks.split('x')) == len(
                args.reg_smoothing.split('x')))):
        sys.exit(
            "twolevel_dbm.py error: iterations, shrinks and smoothing do not match in length"
        )

    #if not which("antsMultivariateTemplateConstruction2.sh"):
    #    sys.exit("antsMultivariateTemplateConstruction2.sh command not found")

    inputs = read_csv(args.input)
    setup_and_check_inputs(inputs, args)

    if args.type == '2level':
        firstlevel(inputs, args)
    else:
        secondlevel(inputs, args, secondlevel=False)


if __name__ == '__main__':
    main()

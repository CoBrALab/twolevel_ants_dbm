Two Level DBM
----------------

This is a python pipeline which wraps around the ``antsMultivariateTemplateConstruction2.sh``
tool from [ANTsX/ANTs](https://github.com/ANTsX/ANTs) to produce unbiased produce deformation
based morphomometry outputs for statistical analysis for both cross-sectional (1 level)
and longitudinal (2 level) populations.

For 1 level modelbuilds, an unbiased ANTs modelbuild will be performed across all
input files, and jacobian determinant images in the model space will be computed.

For 2 level modelbuilds, an unbiased ANTs modelbuild will be performed across each
group at the first level, and then another unbiased modelbuild will be performed
at the second level using the unbiased averages from the first level as input.
Jacobian determinant images within the level one groups will be resampled into
the final unbiased space. In addition, overall determinants from each input file
to the final unbiased space are produced.

# Input

Input is a csv-formatted file with with one input NIFTI file per row for 1 level
model builds. For a 2 level modelbuild, each row can have 2 or more input files.
Currently mixed level models (2 level modelbuilds where some rows have a single file)
are not implemented.

Suggested input files are skull-stripped preprocessed brains, such as those output
by [CoBrALab/minc-bpipe-library](https://github.com/CobraLab/minc-bpipe-library).
Skull stripped files produce better initial affine matches during registration and
provide stronger features to SyN registration stages compared to unstripped files.

# Outputs

``twolevel_dbm.py`` produces three types of jacobian determinant images from the
model builds ``nlin``, ``relative`` and ``absolute``. ``nlin`` files are the
raw registration warp fields converted to jacobians, ``relative`` files have
residual affine components of the warp field removed using
``ANTSUseDeformationFieldToGetAffineTransform`` and ``absolute`` files have the
affine jacobian added to account for bulk volume changes. ``relative`` and ``absolute``
files are generally expected to be used for statistical analysis.

# Full help

```
usage: twolevel_dbm.py [-h] -i INPUT
                       [--jacobian-sigmas JACOBIAN_SIGMAS [JACOBIAN_SIGMAS ...]]
                       [--rigid-model-target RIGID_MODEL_TARGET]
                       [--resample-to-common-space RESAMPLE_TO_COMMON_SPACE]
                       [--dry-run] [--N4] [--metric METRIC]
                       [--transform {SyN,BSplineSyN,TimeVaryingVelocityField,TimeVaryingBSplineVelocityField,Affine,Rigid}]
                       [--reg-iterations REG_ITERATIONS]
                       [--reg-smoothing REG_SMOOTHING]
                       [--reg-shrinks REG_SHRINKS] [--float]
                       [--average-type {mean,normmean,median}]
                       [--gradient-step GRADIENT_STEP]
                       [--model-iterations MODEL_ITERATIONS]
                       [--cluster-type {local,sge,pbs,slurm}]
                       [--walltime WALLTIME] [--memory-request MEMORY_REQUEST]
                       [--local-threads LOCAL_THREADS]
                       {1level,2level}

This pipeline performs one or two level model building on files using
antsMultivariateTemplateConstruction2.sh and generates smoothed jacobian
determinent fields suitable for deformation based morphomometry (DBM)
analysis.

positional arguments:
  {1level,2level}       What type of DBM processing to run on input file, see
                        --input for details on how to format CSV file for
                        different types.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input CSV file for DBM, for 1level mode, a single
                        column, for 2level, each each row constructs a first
                        level model followed by a second level model of the
                        resulting first level averages. File paths must be
                        absolute. (default: None)
  --jacobian-sigmas JACOBIAN_SIGMAS [JACOBIAN_SIGMAS ...]
                        List of smoothing sigmas used for final output,
                        defaults to 2x finest resolution input file or rigid
                        model target if provided. (default: None)
  --rigid-model-target RIGID_MODEL_TARGET
                        Target file to use for rigid registration of the
                        second level, otherwise unbiased average to start
                        (default: None)
  --resample-to-common-space RESAMPLE_TO_COMMON_SPACE
                        NOT YET IMPLEMENTED -- Target space to resample
                        jacobians to after unbiased model build, typically an
                        MNI model, triggers a registration to this target
                        (default: None)
  --dry-run             Don't run commands, instead print to stdout (default:
                        False)

advanced options:
  --N4, --no-N4         Run N4BiasFieldCorrection during model build on input
                        files. (default: False)
  --metric METRIC       Specify metric used for non-linear template stages
                        (default: CC[4])
  --transform {SyN,BSplineSyN,TimeVaryingVelocityField,TimeVaryingBSplineVelocityField,Affine,Rigid}
                        Transformation type used during model build (default:
                        SyN)
  --reg-iterations REG_ITERATIONS
                        Max iterations for non-linear stages (default:
                        100x100x70x20)
  --reg-smoothing REG_SMOOTHING
                        Smoothing sigmas for non-linear stages (default:
                        3x2x1x0)
  --reg-shrinks REG_SHRINKS
                        Shrink factors for non-linear stages (default:
                        6x4x2x1)
  --float, --no-float   Run registration with float (32 bit) or double (64
                        bit) values (default: True)
  --average-type {mean,normmean,median}
                        Type of average used during model build, default
                        normalized mean (default: normmean)
  --gradient-step GRADIENT_STEP
                        Gradient step size at each iteration during model
                        build (default: 0.25)
  --model-iterations MODEL_ITERATIONS
                        How many registration and average rounds to do
                        (default: 3)

cluster options:
  --cluster-type {local,sge,pbs,slurm}
                        Choose the type of cluster system to submit jobs to
                        (default: local)
  --walltime WALLTIME   Option for job submission specifying requested time
                        per pairwise registration. (default: 20:00:00)
  --memory-request MEMORY_REQUEST
                        Option for job submission specifying requested memory
                        per pairwise registration. (default: 8gb)
  --local-threads LOCAL_THREADS, -j LOCAL_THREADS
                        For local execution, how many subject-wise modelbuilds
                        to run in parallel, defaults to number of CPUs
                        (default: 8)
```

# Next Generation Pipeline Development!

While this pipeline is fully functional, the underlying antsMultivariateTemplateConstruction2.sh
has limitations in its flexibility. As such, this entire pipeline was rewritten from scratch
along with lots of new features added.

Please see
https://github.com/CoBrALab/optimized_antsMultivariateTemplateConstruction








































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
model builds.

For a 2 level modelbuild, each row must have 2 or more input files. Mixed level
modelbuilds are supported, where some (but not all) rows may have one file. In
this case rows with a single file specified will be used in the second level
model build, but have resampled first-level jacobians of unity.

Suggested input files are skull-stripped preprocessed brains, such as those output
by [CoBrALab/minc-bpipe-library](https://github.com/CobraLab/minc-bpipe-library).
Skull stripped files produce better initial affine matches during registration and
provide stronger features to SyN registration stages compared to unstripped files.

## Configuration

Suggested configuration is to use a ``--rigid-model-target``, typically an MNI model,
with the brain extracted. If you have a sufficently large population, you may wish
to upsample the target to twice the resolution of your population, to take advantage
of the population random sampling of the underlying average anatomy.

# Outputs

``twolevel_dbm.py`` produces three types of log jacobian determinant images from the
model builds: ``nlin``, ``relative`` and ``absolute``. ``nlin`` files are the
raw registration warp fields converted to jacobians, ``relative`` files have
residual affine components of the warp field removed using
``ANTSUseDeformationFieldToGetAffineTransform`` and ``absolute`` files have the
affine jacobian added to the ``nlin`` to account for bulk volume changes.
``relative`` and ``absolute`` files are generally expected to be used for
statistical analysis.

For two-level pipelines, jacobians are produced for two different transformations,
``resampled`` are within-subject jacobians, resampled into the final average
anatomical space, ``overall`` are jacobains encoding all volumetric differences
between the individual subject input files to the final anatomical average.
For most applications the ``resampled`` jacobians are suggested for analysis.

For one-level pipelines, only ``overall`` jacobians are produced.

In all cases the values of the log jacobians are to be interpreted as follows:
- positive values indicate that the voxel in template space must be expanded to
get to the subject space, i.e. the subject voxel is larger than the template voxel
- negative values indicate that the voxel in template space must be reduced
to get to the subject space, i.e. the subject voxel is smaller than the template voxel

# Interrupted Pipelines

``twolevel_dbm.py`` keeps track of which level 1 group model builds have been completed and
will not re-process those files if a pipeline is interrupted and run again. Similarly
if the second level model build is complete it will not be re-run. There is partial
resume capability of post-processing, it will improve in the future.

# Requirements

## Python

``twolevel_dbm.py`` requires 3.6 or newer and the packages listed in ``requirements.txt``

## Other tools

This pipeline relies on the [ANTsX/ANTs](https://github.com/ANTsX/ANTs) tools v2.3.1 or newer,
build without the VTK addons. Note that in earlier releases, a bug in the ``antsMultivariateTemplateConstruction2.sh``
script aggressively strips all periods from the input filenames breaking the
naming expected by ``twolevel_dbm.py``. Either avoid using periods in your filenames
or install ANTs version https://github.com/ANTsX/ANTs/commit/412bb8fef534c0e9b6c1fc22c39492ab46ea22e4
or newer.

# Full help

```
usage: twolevel_dbm.py [-h]
                       [--jacobian-sigmas JACOBIAN_SIGMAS [JACOBIAN_SIGMAS ...]]
                       [--rigid-model-target RIGID_MODEL_TARGET]
                       [--resample-to-common-space RESAMPLE_TO_COMMON_SPACE]
                       [--skip-dbm] [--dry-run] [-v] [--N4] [--metric METRIC]
                       [--transform {SyN,BSplineSyN,TimeVaryingVelocityField,TimeVaryingBSplineVelocityField,Affine,Rigid}]
                       [--reg-iterations REG_ITERATIONS]
                       [--reg-smoothing REG_SMOOTHING]
                       [--reg-shrinks REG_SHRINKS] [--float]
                       [--average-type {mean,normmean,median}]
                       [--gradient-step GRADIENT_STEP]
                       [--model-iterations MODEL_ITERATIONS]
                       [--modelbuild-command MODELBUILD_COMMAND]
                       [--cluster-type {local,sge,pbs,slurm}]
                       [--walltime WALLTIME] [--memory-request MEMORY_REQUEST]
                       [--local-threads LOCAL_THREADS]
                       {1level,2level} input

This pipeline performs one or two level model building on files using
antsMultivariateTemplateConstruction2.sh and generates smoothed jacobian
determinent fields suitable for deformation based morphomometry (DBM)
analysis.

positional arguments:
  {1level,2level}       What type of DBM processing to run on input file, see
                        input for details on how to format CSV file for
                        different types.
  input                 Input CSV file for DBM, for 1level mode, a single
                        column, for 2level, each each row constructs a first
                        level model followed by a second level model of the
                        resulting first level averages. File paths must be
                        absolute.

optional arguments:
  -h, --help            show this help message and exit
  --jacobian-sigmas JACOBIAN_SIGMAS [JACOBIAN_SIGMAS ...]
                        List of smoothing sigmas used for final output,
                        defaults to FWHM of 2x finest resolution input file or
                        rigid model target if provided. (default: None)
  --rigid-model-target RIGID_MODEL_TARGET
                        Target file to use for rigid registration of the
                        second level, otherwise unbiased average to start
                        (default: None)
  --resample-to-common-space RESAMPLE_TO_COMMON_SPACE
                        Target nifti file of atlas space to resample to
                        jacobians to after unbiased model build, typically an
                        MNI model, triggers a registration to this target
                        (default: None)
  --skip-dbm            Skip generating DBM outputs (default: False)
  --dry-run             Don't run commands, instead print to stdout (default:
                        False)
  -v, --verbose         Be verbose about what is going on (default: False)

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
                        (default: 4)
  --modelbuild-command MODELBUILD_COMMAND
                        Command to use for performing model build, must accept
                        same arguments as
                        antsMultivariateTemplateConstruction2.sh (default:
                        antsMultivariateTemplateConstruction2.sh)

cluster options:
  --cluster-type {local,sge,pbs,slurm}
                        Choose the type of cluster system to submit jobs to
                        (default: local)
  --walltime WALLTIME   Option for job submission specifying requested time
                        per pairwise registration. (default: 4:00:00)
  --memory-request MEMORY_REQUEST
                        Option for job submission specifying requested memory
                        per pairwise registration. (default: 8gb)
  --local-threads LOCAL_THREADS, -j LOCAL_THREADS
                        For local execution, how many subject-wise modelbuilds
                        to run in parallel, defaults to number of CPUs
                        (default: 20)

```

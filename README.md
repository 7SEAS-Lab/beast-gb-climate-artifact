# Gordon Bell Prize for Climate Modeling 2026 submission repository for BEAST, "4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling"

[![Weights: Zenodo](https://www.freewebtools.com/api/badge/Zenodo-Weights-4c1.svg?style=flat&labelColor=555)](https://doi.org/10.5281/zenodo.23188053)
[![Paper: arXiv](https://www.freewebtools.com/api/badge/Paper-Arxiv-e05d44.svg?style=flat&labelColor=555)](https://arxiv.org/abs/2609.12815)

<!-- Badge by FreeWebTools.com -->
This repository presents the code associated with reproducing BEAST, the first-ever Bayesian Swin Transformer for atmospheric forecasting on 0.25 degree global resolution able to accurately quantify both aleatoric and epistemic uncertainty. We use a 4D-parallelization scheme that introduces a unique domain-tensor-parallelism strategy and a novel uncertainty parallel method, allowing us to achieve a peak performance of 3.96 EFLOP/s on 5120 NVIDIA GH200 nodes on the JUPITER supercomputer and 463.2 PFLOP/s on 1024 AMD MI250X nodes on the LUMI supercomputer. This contribution unlocks the potential of high-fidelity uncertainty quantification in atmospheric AI models.

This is a paper artifact, and not in active development. To stay updated on latest developments, or to report any issues, please visit [beast](https://github.com/7SEAS-Lab/beast).

## Setup

### Dev tools (for your working environment, and code quality)

**pre-commit**

1) Install pre-commit

   ```pip install pre-commit```

2) Install hooks

   ```pre-commit install```

3) Test on all files

   ```pre-commit run --all-files```

### System setup


1) Source system-specific configuration

```console
source systems/<system_file.sh>
```

2) Ensure the container image is there. If not, then download it (not for LUMI, for now):

```console
apptainer pull $SIF docker://nvcr.io/nvidia/pytorch:26.04-py3
```

3) Run init script

```console
init_system
```

4) Done

#### Troubleshooting

 * Remove any placeholder paths or Slurm accounts in the system files, or the experiment scripts.
 * Make sure the container image exists
 * Make sure the virtual environment exists (if not, run the init_system function)
 * If the library is pointing to the wrong path (should either be a folder inside the container, the virtual environment, or the code repository for editable libraries), either restart the environment (`rm -rf $VENV_DIR`, and run `init_system` again), or install the library manually (`apptainer exec $SIF systems/venv_entry.sh pip install <library>`)
 * For anything else, create an issue.

## Training the BEAST
 * 8-way domain- and tensor-parallelism (DTP) (via Jigsaw)
 * Uncertainty parallelism (UP), i.e. sample parallelism (for variational inference) and noise injection conditioning for ensembles (NICE, for diffusion)
 * Data parallelism (DP) via torch DDP

 * Training on hourly subsampled 0.25 degree ERA5 data (chunked stored) with 84 variables and 3 constant masks
     * Two timesteps serve as input (concatenated along the variable dimension) to predict the next two (12h) timesteps (the number of input and output timesteps can be configured)
     * Normalization values are in ``data/normalization/norms.nc``

To run training, configure in ``configs/your_config.yaml``
* Adjust the process mesh (``mesh_dims``) to match your number of GPUs/nodes
    * The last two dims are "fixed" to ``[4,2]``: 2 nodes (8 GPUs) on Jupiter and 1 node (8 GCDs) on LUMI
    * Distribute your nodes to the first two dims, such that their product equals the job node count ``<N_NODES>``; the order is ``[num_data_parallel, num_sample_parallel]``
* Adjust other parameters in the training configuration, e.g.
    * Where the training data and constant masks lie (originating from ``DATA_DIR="/path/to/data/"`` in ``systems/<system_file.sh>``)
    * Which constant masks to use (list of file names)
    * How much data to use (date format in ``train_time_range``)
    * How many timesteps to use as input (``n_in_timesteps``)
    * How big the model should be: **``embed_dim``**, ``heads``, etc.
    * How many epochs to train (``n_epochs``)
    * How many Bayesian samples per SP process (``n_samples``)

### On Jupiter
```console
sbatch -A your_account -p booster --nodes <N_NODES> --tasks-per-node 4 --gres=gpu:4 -t 00:06:00 --job-name=train_beast experiments/run_gb_script.sh systems/jupiter_container.sh src/main.py --config_file configs/predictive_config_jupiter.yaml mesh_dims=-1,2,4,2 training_batch_size=2 data_train_subset=50
```

### On LUMI
```console
sbatch -A your_account -p standard-g --nodes <N_NODES> --tasks-per-node 8 --gres=gpu:8 -t 00:06:00 --job-name=train_ experiments/run_gb_script.sh systems/lumi_singularity.sh src/main.py --config_file configs/hero_config_lumi.yaml mesh_dims=-1,2,4,2 training_batch_size=2 data_train_subset=50
```

Go grab a cup of coffee and stare at loss values.


## Additional functionalities

### MLflow

Runs are tracked using MLflow, with the data stored on the `$OUTPUT_DIR/mlflow` path of the system. In order to look at the interface, start the MLflow server on a login node, and tunnel port 5000 to your local computer.

A virtual environment with MLflow installed (`pip install mlflow`) needs to be active.

The server can be started like this:

```console
source systems/<current_system>.sh
utils/start_mlflow_server.sh
```

If the commands are run directly in the Visual Studio Code terminal, it will handle the tunnel creation for you. Otherwise, the tunnel can also be created manually in VSCode, or using ssh port forwarding.

```console
ssh -L 5000:localhost:5000 <username>@<hostname>
```


### Overriding the configuration using the command line

All parameters in the yaml configuration file can be overridden using the command line. The syntax is ``<root_category>_<parameter_name>=<value>``, where ``<root_category>`` is the first level category in the yaml file, e.g. ``training``, ``model``, etc. For example, to change the number of epochs to 10, constrain the training dataset to 2000 samples, and change the process mesh dimensions to `[1, 2, 2, 2]` add the following to the command line after all the `--` arguments:

```console
... --config_file configs/your_config.yaml training_n_epochs=10 data_train_subset=2000 mesh_dims=1,2,2,2
```

Arrays are passed as comma-separated values, e.g. for the mesh dimensions. Mesh dimensions should always have 4 values, corresponding to the 4 dimensions of the process mesh (DP, SP, JSpatial, JChannel), even if some of them are 1 (i.e. no parallelism in that dimension). If a value of -1 is passed, it will be automatically inferred from the number of nodes and GPUs per node, e.g. if mesh_dims is set to ``-1,1,2,2`` and the job is launched with 4 nodes and 4 GPUs per node (16 GPUs in total), the first dimension will be inferred to be 4, since $4\times1\times2\times2 = 16$.

### Performance monitoring

We have two ways to profile and monitor the performance of the model during training.

#### NVIDIA Nsight Systems/Rocprofv3

NVIDIA Nsight Systems (for Jupiter) and Rocprofv3 (for LUMI) are powerful profiling tools that can be used to analyze the performance of the model at a very detailed level, including GPU utilization, memory usage, and kernel execution times. They can be activated by passing the `--profsys` flag to `experiments/run_gb_script.sh` (before the system script), which will then run the training script with the profiler on ranks 0 to 3 and rank 64. The profiler will save the results in the output directory (`${OUTPUT_DIR}/${SLURM_JOB_PARTITION}/${SLURM_JOB_ID}/profiles/`), and can be visualized using the respective tools.

#### Perun

If you ever want to feel bad, add the `--perun` flag to `experiments/run_gb_script.sh` (before the system script), and run the training script with Perun. Gaze upon the `${WORKDIR}/results/perun_results` directory to come face to face with the staggering amount of energy that your training run has consumed, and the carbon emissions it has generated. Not for the faint of heart.

## Wiki

### Distributed Process Groups in 

Parallel process group keys:
 - JChannel: Jigsaw Row Tensor Parallel group (part of DTP)
 - JSpatial: Jigsaw Column Tensor Parallel group (part of DTP)
 - DTP: Domain Tensor Parallel group, equal to Column x Row
 - SP: Sampling Parallel group
 - DP: Data Parallel Group
 - DDP: Distributed Data Parallel Group (DP x SP), to be passed to the DDP module

Mesh Dimensions: [DP, SP, JSpatial, JChannel]

See the [beast](https://github.com/RAI-SCC/beast) and [jigsaw](https://github.com/RAI-SCC/jigsaw) packages for more details on the parallelization scheme.

### Checkpoints

Checkpoints are stored using `torch.save`, and are a dictionary with the following keys:
- model_state_dict: the state dict of the model
- optimizer_state_dict: the state dict of the optimizer
- epoch: the current epoch
- step: the current step (global)
- channel_group_rank: the rank of the channel parallel group
- loss: the loss value at the time of saving
- format: the format of the checkpoint (currently: 1), but can be used to distinguish between different checkpoint formats in the future, e.g. if we want to save additional information or change the way we save the model state dict (e.g. using sharded checkpoints for larger models)

## Model weights

The model weights for the trained BEAST model are available on [Zenodo](https://doi.org/10.5281/zenodo.23188053).

## Acknowledgments

This research is supported by:

- the German Federal Ministry of Research, Technology and Space (BMFTR) under
  the grant 01LK2313A SMARTWEATHER21-SCC-2,
- the Carl Zeiss Foundation under the Breakthroughs 2025 project "WOW – a World
  model of Our World",
- the European High Performance Computing Joint Undertaking (EuroHPC JU), the
  BMFTR, the Ministry of Culture and Science of North Rhine-Westphalia
  (MKW NRW) and the Hessian Ministry of Science and Research, Arts and Culture
  (HMWK) under grant agreement No 101250682 (JUPITER AI Factory, JAIF).

The JUPITER supercomputer was used through the JUPITER Research and Early Access
Program (JUREAP). JUPITER is funded by the EuroHPC Joint Undertaking, the BMFTR
and the Ministry of Culture and Science of the German state of North
Rhine-Westphalia. Computations on the LUMI supercomputer used resources provided
by CSC – IT Center for Science, Finland. LUMI is co-funded by the EuroHPC Joint
Undertaking and the LUMI Consortium.

## Citation

The recommended citation for this work is:

```bibtex
@misc{kieckhefen2026beast,
  title         = {4D Parallelism Unlocks Exascale Bayesian Neural Networks for High-Fidelity Atmospheric Modeling},
  author        = {Kieckhefen, Deifilia and Guti{\'e}rrez Hermosillo Muriedas, Juan Pedro and Heyen, Lars Helge and Bode, Mathis and Hakulinen, Iida and Herten, Andreas and John, Chelsea Maria and Kurth, Thorsten and Moisala, Anni and {\"O}zdemir, Asena Karolin and Phipps, Kaleb and Taubert, Oskar and Weyrauch, Arvid and G{\"o}tz, Markus and Debus, Charlotte},
  year          = {2026},
  eprint        = {2609.12815},
  archivePrefix = {arXiv},
  primaryClass  = {physics.ao-ph},
  doi           = {10.48550/arXiv.2609.12815},
  url           = {https://arxiv.org/abs/2609.12815}
}
```

The citation metadata is also in [CITATION.cff](CITATION.cff), and GitHub shows
it through the "Cite this repository" button.

> [!NOTE]
> Extreme weather event predictions in this repository use
[ExtremeWeatherBench](https://github.com/brightbandtech/ExtremeWeatherBench)
(EWB). See [`evaluation_documentation.md`](evaluation_documentation.md)
for how it is used.

> [!WARNING]
> ``logs`` and any other output from the runs will be moved to a directory indicated in the system files as ``OUTPUT_DIR``, as the home directory on some systems is limited, and this allows us to share results with everyone involved.

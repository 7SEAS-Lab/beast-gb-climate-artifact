#!/usr/bin/env bash
# =============================================================================
# Container configuration and paths
# =============================================================================
export WORKDIR="$( pwd )" # For source code

export PROJECT_DIR="/path/to/project"
export SCRATCH_DIR="/path/to/scratch"

export VENV_DIR="${HOME}/.venvs"
export VENV_PATH="${VENV_DIR}/gbb"

export CONTAINER_HOME="/e/data1/jureap-data/beast/images"
export CONTAINER_NAME="pytorch26.04.sif"
export SIF="${CONTAINER_HOME}/${CONTAINER_NAME}"

export DATA_DIR="/path/to/data/"
export OUTPUT_DIR="/path/to/output"

# =============================================================================
# Environment configuration
# =============================================================================
# Locale
export LC_ALL=C
export LANG=C
export PYTHONNOUSERSITE=1

# OpenMPI configuration for UCX transport
export OMPI_MCA_btl="^openib"
export OMPI_MCA_pml="ucx"
export OMPI_MCA_osc="ucx"
export OMPI_MCA_mpi_param_check=0

# UCX configuration for InfiniBand NDR200
export UCX_TLS="rc,cuda_copy,cuda_ipc"
export UCX_MEMTYPE_CACHE="n"
export UCX_RNDV_FRAG_MEM_TYPE="cuda"
export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"

# # NCCL configuration for multi-node GPU communication
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
#export NCCL_DEBUG=info
#export NCCL_DEBUG_FILE=$(pwd)/logs/nccl_log.%h.%p
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="ib0"
export GLOO_SOCKET_IFNAME="ib0"
export NCCL_SOCKET_FAMILY="AF_INET"
export NCCL_IB_HCA="mlx5"
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL

# Timeout configurations (for correctness, max values, maybe it does not work)
export NCCL_IB_TIMEOUT=10
export UCX_RC_TIMEOUT=8s
export NCCL_IB_RETRY_CNT=5

# TORCH
export LD_LIBRARY_PATH=/.singularity.d/libs:/lib64:/usr/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/.singularity.d/libs:$LIBRARY_PATH
export TRITON_LIBCUDA_PATH=/.singularity.d/libs

# PMIx for Slurm integration
export PMIX_MCA_gds="hash"

# BEAST
export BEAST_USE_TE_DPA=1

# =============================================================================
# Profiling configuration
# =============================================================================

export PROFILE_DIR="${OUTPUT_DIR}/${SLURM_JOB_PARTITION}/${SLURM_JOB_ID}/profiles"

profsys_cmd=(
    nsys profile
    --output "${PROFILE_DIR}/profile_%q{SLURM_NODEID}_%q{SLURM_PROCID}"
    --trace=cuda,cudnn,cublas,osrt,nvtx
    --cudabacktrace=all
    --python-backtrace=cuda
    --python-sampling=true
)
export PROFSYS_CMD="${profsys_cmd[@]}"

# =============================================================================
# Slurm configuration
# =============================================================================

export SYSTEM_SLURM_PARAMS=(
    -c71
    #--cpu-bind=v
    --gpu-bind=closest
    apptainer
    exec
    --nv
    --bind "$DATA_DIR":"/workspace/"
)

setup_system() {
    echo "Purging modules"
    ml purge
    # ml load Stages/2026 GCC/14.3.0 OpenMPI/5.0.8 MPI-settings/CUDA
    echo "Loaded modules:"
    module list
}

init_system() {
    setup_system
    rm -r $VENV_PATH
    mkdir -p $VENV_DIR
    apptainer exec --nv $SIF python -m venv $VENV_PATH --system-site-packages
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade pip
    apptainer exec --nv $SIF systems/venv_entry.sh pip install -r requirements.txt
    apptainer exec --nv $SIF systems/venv_entry.sh pip install --upgrade h5py
    echo "zipping venv"
    pushd $VENV_DIR
    venv_name=$(basename "${VENV_PATH}")
    echo $venv_name
    tar -czf  ${venv_name}.tar.gz  ${venv_name}
    popd
}

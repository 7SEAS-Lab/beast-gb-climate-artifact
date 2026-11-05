#!/usr/bin/env bash
# =============================================================================
# Container configuration and paths
# =============================================================================
export WORKDIR="$( pwd )"
export LOCAL_WORKDIR="$( pwd )"

export PROJECT_DIR=/path/to/project
export SCRATCH_DIR=/path/to/scratch

export VENV_DIR=$HOME/.venvs/
export VENV_PATH="${VENV_DIR}/gbb"

export CONTAINER_HOME="/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/"
export CONTAINER_NAME="lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif"
export SIF="${CONTAINER_HOME}/${CONTAINER_NAME}"

export DATA_DIR=$SCRATCH_DIR/data/
export OUTPUT_DIR=$SCRATCH_DIR/output

# =============================================================================
# Environment configuration
# =============================================================================
# Locale
export LC_ALL=C
export LANG=C
export PYTHONNOUSERSITE=1

# Cray MPICH Configuration
export MPICH_GPU_SUPPORT_ENABLED=1          # Enable GPU-aware MPI
export MPICH_OFI_NIC_POLICY="GPU"           # Use GPU-mapped NICs
export MPICH_RANK_REORDER_METHOD=1          # Optimize rank placement
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

# RCCL Configuration
export RCCL_DEBUG="${RCCL_DEBUG:-INFO}"
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
#export NCCL_NET_GDR_LEVEL="PHB" # might cause hangs https://github.com/lumi-ai-factory/laifs-container-recipes/issues/30

# Libfabric/CXI for Slingshot
# All recommended settings from https://462000265.lumidata.eu/ai-20260611/files/LUMI-ai-20260611-09-Extreme_scale_AI.pdf
export HSA_FORCE_FINE_GRAIN_PCIE=1 # Enable peer-to-peer access to large BAR addressing support.
export FI_MR_CACHE_MONITOR=userfaultfd # Sets the memory cache monitor to detect changes between virtual and physical memory pages. kdreg2 is another valid option.
export FI_CXI_DISABLE_HOST_REGISTER=1 # Avoids ROCm allocation calls from the provider that may cause RCCL deadlocks.
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Should be increased, especially for large jobs.
export FI_CXI_RDZV_PROTO=alt_read # Use the alt_read rendezvous protocol.
export FI_CXI_RX_MATCH_MODE=hybrid # It allows the network stack to transition to software matching if hardware resources are exhausted.
export FI_CXI_RDZV_EAGER_SIZE=0 # Prevents sending data before the receiver is ready.
export FI_CXI_RDZV_THRESHOLD=0 # Sets the message size threshold above which the rendezvous protocol is used; setting to 0 forces all messages through the rendezvous path.
export FI_CXI_RDZV_GET_MIN=0 # Disables the rendezvous get optimization; use with FI_CXI_RDZV_PROTO=alt_read.
export FI_CXI_DEFAULT_TX_SIZE=2048 # Should be set especially for large jobs that are dependent on unexpected rendezvous messaging.

# BEAST
export BEAST_USE_TE_DPA=1

# ROCm / HIP Settings
# export ROCM_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  # 8 GCDs per node
#
# =============================================================================
# Profiling configuration
# =============================================================================

export PROFILE_DIR="${OUTPUT_DIR}/${SLURM_JOB_PARTITION}/${SLURM_JOB_ID}/profiles"

profsys_cmd=(
    rocprofv3
    --sys-trace
    --output-format pftrace
    -d "$PROFILE_DIR"
    --
)
export PROFSYS_CMD="${profsys_cmd[@]}"

# =============================================================================
# Slurm configuration
# =============================================================================

# CPU mask for optimal GPU-CPU affinity on LUMI-G

# Each MI250X has 2 GCDs. The CPU-GPU mapping on LUMI-G is:
#   GCD 4,5 -> NUMA 0 (cores 0-15)  -> GCD 0,1
#   GCD 2,3 -> NUMA 1 (cores 16-31) -> GCD 2,3
#   GCD 6,7 -> NUMA 2 (cores 32-47) -> GCD 4,5
#   GCD 0,1 -> NUMA 3 (cores 48-63) -> GCD 6,7
#
# For 8 tasks (one per GCD), bind each task to 7 cores (mask 0xfe = 11111110)
# Leaving core 0 of each group for system tasks

# Set up the CPU bind masks
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

export SYSTEM_SLURM_PARAMS=(
    --cpu-bind="mask_cpu=$CPU_BIND_MASKS"
    --cpus-per-task=7
    --mpi=pmi2
    singularity
    exec
    --bind "$DATA_DIR"
)

setup_system(){
    echo "Setting up LUMI-G environment..."
    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings

    echo "Loaded modules:"
    module list
}

init_system(){
    rm -r $VENV_PATH
    mkdir -p $VENV_DIR
    module purge
    module use /appl/local/laifs/modules
    module load lumi-aif-singularity-bindings
    singularity exec $SIF python -m venv $VENV_PATH --system-site-packages
    singularity exec $SIF systems/venv_entry.sh pip install --upgrade pip
    singularity exec $SIF systems/venv_entry.sh pip install -r requirements.txt
    singularity exec $SIF systems/venv_entry.sh pip install --upgrade h5py
    echo "zipping venv"
    pushd $VENV_DIR
    venv_name=$(basename "${VENV_PATH}")
    echo $venv_name
    tar -czf  ${venv_name}.tar.gz  ${venv_name}
    popd
}

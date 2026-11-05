srun -n 1 apptainer exec --nv \
    --overlay cuda_overlay.img \
    $SIF \
    systems/venv_entry.sh python - <<'PY'
import triton
import triton.runtime.driver
print(triton.runtime.driver.active)
PY

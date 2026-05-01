import Pkg
Pkg.add(["ParallelStencil", "CUDA", "CairoMakie", "YAML", "ProgressBars", "JLD2"])

# ============================================================
# Change this to select which run to execute.
# The code reads  runs/<job_name>/input/params.yml
# and writes output to  runs/<job_name>/output/
# ============================================================
const job_name = "default"

# ---- GPU backend (must be a bare top-level macro call) ----
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

include("src/config.jl")
include("src/shallow_water.jl")
include("src/visualize.jl")

cfg = load_config(job_name)
run_simulation(cfg)
make_animation(cfg)
make_timeseries(cfg)


import Pkg
Pkg.add(["ParallelStencil", "CUDA", "CairoMakie", "YAML", "ProgressBars", "JLD2"])

length(ARGS) == 1 || error("Usage: julia run_single.jl <job_name>")
const job_name = ARGS[1]

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

include("src/shallow_water_solver/config.jl")
include("src/shallow_water_solver/shallow_water.jl")
include("src/shallow_water_solver/visualize.jl")

cfg = load_config(job_name)
run_simulation(cfg)
make_animation(cfg)
make_U_animation(cfg)
make_timeseries(cfg)

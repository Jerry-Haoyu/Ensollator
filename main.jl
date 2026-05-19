# Submit one SLURM batch job per experiment found in runs/.
# Each subdirectory that contains input/params.yml is treated as a job.
# Logs are written to runs/<job_name>/batch_out_<jobid> and batch_err_<jobid>.

# runs_dir = "runs/shallow_water_simulator"

# for entry in sort(readdir(runs_dir))
#     params_file = joinpath(runs_dir, entry, "input", "params.yml")
#     isfile(params_file) || continue

#     output_dir = joinpath(runs_dir, entry, "output")
#     isdir(output_dir) && !isempty(readdir(output_dir)) && continue

#     log_dir = joinpath(runs_dir, entry)
#     mkpath(log_dir)

#     cmd = `sbatch
#         --job-name=$entry
#         --output=$(joinpath(log_dir, "batch_out_%j"))
#         --error=$(joinpath(log_dir, "batch_err_%j"))
#         --export=JOB_NAME=$entry
#         launch.slurm`

#     result = readchomp(cmd)
#     println("$entry  →  $result")
# end


import Pkg; Pkg.add("LaTeXStrings")

include("src/shallow_water_solver/visualize.jl")
plot_fft()
plot_frequency_shift()
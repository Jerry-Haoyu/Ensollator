"""
---------------------------------
BY : Claude Code
Prompted/Reviewed BY: Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""


using CairoMakie
using ProgressBars
using JLD2
using Statistics
using Printf
using LaTeXStrings

include(joinpath(@__DIR__, "fft.jl"))

# ============================================================
# Animated heatmap of thermocline depth anomaly h(x,y,t)
# ============================================================
function make_animation(cfg::SimConfig)
    out_dir = cfg.output_dir

    data  = load(joinpath(out_dir, "h_frames.jld2"))
    h_frames = data["h_frames"]
    coord    = load(joinpath(out_dir, "simulation_coords.jld2"), "coord")

    xs_km = coord._xs_km
    ys_km = coord._ys_km

    nframes = length(h_frames)
    hmax    = max(maximum(maximum(abs, f) for f in h_frames), 0.1)

    fig = Figure(size = (900, 400), fontsize = 13)

    title_obs = Observable("Thermocline Depth Anomaly — Day 0.0")
    ax = Axis(fig[1, 1];
        xlabel = "Zonal distance  [km]",
        ylabel = "Meridional distance  [km]",
        title  = title_obs,
        aspect = DataAspect())

    h_obs = Observable(h_frames[1])
    hm = heatmap!(ax, xs_km, ys_km, h_obs;
        colormap   = Reverse(:RdBu),
        colorrange = (-hmax, hmax))
    Colorbar(fig[1, 2], hm; label = "h′  [m]", width = 15, ticklabelsize = 11)

    filename = joinpath(out_dir, "thermocline.mp4")
    pbar     = ProgressBar(total = nframes)

    record(fig, filename, 1:nframes; framerate = cfg.framerate) do iframe
        update(pbar, 1)
        h_obs[]     = h_frames[iframe]
        t_day       = iframe * cfg.nout * cfg.dt / 86_400.0
        title_obs[] = @sprintf("Thermocline Depth Anomaly — Day %.1f", t_day)
    end

    println("Animation saved → $filename")
end


# ============================================================
# Time series of domain-mean and east/west thermocline depth
# ============================================================
function make_timeseries(cfg::SimConfig)
    out_dir  = cfg.output_dir
    data     = load(joinpath(out_dir, "h_frames.jld2"))
    h_frames   = data["h_frames"]
    U_a_frames = data["U_a_frames"]
    U_bg       = data["U_bg"]

    nframes = length(h_frames)
    h_means = mean.(h_frames)
    h_wests = [mean(mat[1:end÷2, :])          for mat in h_frames]
    h_easts = [mean(mat[end÷2:end, :])  for mat in h_frames]

    U_total_means = [mean(mat .+ U_bg) for mat in U_a_frames]

    days = [i * cfg.nout * cfg.dt / 86_400.0 for i in 1:nframes]

    f   = Figure(size = (800, 600))
    ax1 = Axis(f[1, 1];
        ylabel = "Thermocline Depth Anomaly  [m]",
        title  = "Time Series of Thermocline Depth Anomaly")
    hidexdecorations!(ax1; grid = false)

    lines!(ax1, days, h_means, color = :black, label = "Domain mean")
    lines!(ax1, days, h_wests, color = :red,   label = "Depth Anomaly West")
    lines!(ax1, days, h_easts, color = :blue,  label = "Depth Anomaly East")
    axislegend(ax1, position = :rb)

    ax2 = Axis(f[2, 1];
        xlabel = "Days",
        ylabel = "Wind Stress  [m s⁻¹]",
        title  = "Domain-Mean Total Wind Stress  (U_a + U_bg)")

    lines!(ax2, days, U_total_means, color = :darkorange, label = "⟨U_a + U_bg⟩")
    axislegend(ax2, position = :rb)

    save(joinpath(cfg.output_dir, "time_series.png"), f)
    println("Time series saved → $(joinpath(cfg.output_dir, "time_series.png"))")
end


# ============================================================
# Animated heatmap of atmospheric zonal wind U_a(x,y,t)
# ============================================================
function make_U_animation(cfg::SimConfig)
    out_dir    = cfg.output_dir
    data       = load(joinpath(out_dir, "h_frames.jld2"))
    U_a_frames = data["U_a_frames"]
    coord      = load(joinpath(out_dir, "simulation_coords.jld2"), "coord")

    xs_km = coord._xs_km
    ys_km = coord._ys_km

    # u_surface = −U_a  (baroclinic mode: surface and upper winds are opposite)
    nframes = length(U_a_frames)
    Umax    = max(maximum(maximum(abs, f) for f in U_a_frames), 0.1)

    fig = Figure(size = (900, 400), fontsize = 13)

    title_obs = Observable("Surface Zonal Wind  (−U_a) — Day 0.0")
    ax = Axis(fig[1, 1];
        xlabel = "Zonal distance  [km]",
        ylabel = "Meridional distance  [km]",
        title  = title_obs,
        aspect = DataAspect())

    U_obs = Observable(-U_a_frames[1])
    hm = heatmap!(ax, xs_km, ys_km, U_obs;
        colormap   = Reverse(:RdBu),
        colorrange = (-Umax, Umax))
    Colorbar(fig[1, 2], hm; label = "u_surface  [m s⁻¹]", width = 15, ticklabelsize = 11)

    filename = joinpath(out_dir, "surface_U.mp4")
    pbar     = ProgressBar(total = nframes)

    record(fig, filename, 1:nframes; framerate = cfg.framerate) do iframe
        update(pbar, 1)
        U_obs[]     = -U_a_frames[iframe]
        t_day       = iframe * cfg.nout * cfg.dt / 86_400.0
        title_obs[] = @sprintf("Surface Zonal Wind  (−U_a) — Day %.1f", t_day)
    end

    println("Surface wind animation saved → $filename")
end


# ============================================================
# Heatmap of the time-mean atmospheric zonal wind U_a(x,y)
# ============================================================
function make_U_heatmap(cfg::SimConfig)
    out_dir    = cfg.output_dir
    data       = load(joinpath(out_dir, "h_frames.jld2"))
    U_a_frames = data["U_a_frames"]
    coord      = load(joinpath(out_dir, "simulation_coords.jld2"), "coord")

    xs_km = coord._xs_km
    ys_km = coord._ys_km

    # u_surface = −U_a  (baroclinic mode: surface and upper winds are opposite)
    U_surf_mean = -mean(U_a_frames)   # element-wise time mean, negated for surface
    Umax        = max(maximum(abs, U_surf_mean), 0.1)

    fig = Figure(size = (900, 400), fontsize = 13)
    ax  = Axis(fig[1, 1];
        xlabel = "Zonal distance  [km]",
        ylabel = "Meridional distance  [km]",
        title  = "Time-Mean Surface Zonal Wind  (−U_a)",
        aspect = DataAspect())

    hm = heatmap!(ax, xs_km, ys_km, U_surf_mean;
        colormap   = Reverse(:RdBu),
        colorrange = (-Umax, Umax))
    Colorbar(fig[1, 2], hm; label = "u_surface  [m s⁻¹]", width = 15, ticklabelsize = 11)

    filename = joinpath(out_dir, "atm_U_heatmap.png")
    save(filename, fig)
    println("U_a heatmap saved → $filename")
end


# ============================================================
# Power spectrum overlay across all experiments in runs/
# Auto-discovers any subdirectory that has both
#   runs/<exp>/input/params.yml  and  runs/<exp>/output/h_frames.jld2
# ============================================================
function plot_fft(; runs_dir::String = "runs/shallow_water_simulator",
                    filename::String = "fft_power_spectrum.png")
    entries = sort(readdir(runs_dir))
    experiments = String[]
    for entry in entries
        params_path = joinpath(runs_dir, entry, "input",  "params.yml")
        h_path      = joinpath(runs_dir, entry, "output", "h_frames.jld2")
        isfile(params_path) && isfile(h_path) && push!(experiments, entry)
    end
    isempty(experiments) && error("No experiments with h_frames.jld2 found under $runs_dir")
    job_for(entry) = relpath(joinpath(runs_dir, entry), "runs")

    fig = Figure(size = (900, 500), fontsize = 13)
    ax  = Axis(fig[1, 1];
        xlabel = "Frequency  [cycles / year]",
        ylabel = "Power",
        xscale = log10,
        yscale = log10,
        title  = "Power Spectrum of Eastern-Pacific Thermocline Anomaly")

    palette = cgrad(:tab10, max(length(experiments), 2); categorical = true)

    for (i, job) in enumerate(experiments)
        cfg          = load_config(job_for(job))
        series       = get_1d_time_series(cfg)
        dt_sample    = cfg.nout * cfg.dt                     # sampling interval [s]
        freqs, power = power_spectrum(series, dt_sample)

        # Drop DC bin so log-log axes are valid.
        f_cpy = freqs[2:end] .* (86_400.0 * 365.25)
        p     = power[2:end]
        lines!(ax, f_cpy, p; color = palette[i],
               label = "Lx = $(round(Int, cfg.Lx / 1000)) km")
    end

    axislegend(ax, position = :rt)

    out_file = joinpath(runs_dir, filename)
    save(out_file, fig)
    println("FFT power spectrum saved → $out_file")
end

function plot_frequency_shift(; runs_dir::String = "runs/shallow_water_simulator",
                    filename::String = "frequency_shift.png")
    entries = sort(readdir(runs_dir))
    experiments = String[]
    for entry in entries
        params_path = joinpath(runs_dir, entry, "input",  "params.yml")
        h_path      = joinpath(runs_dir, entry, "output", "h_frames.jld2")
        isfile(params_path) && isfile(h_path) && push!(experiments, entry)
    end
    isempty(experiments) && error("No experiments with h_frames.jld2 found under $runs_dir")
    job_for(entry) = relpath(joinpath(runs_dir, entry), "runs")

    basin_width = Vector{Float64}(undef, length(experiments))
    dom_freqs   = Vector{Float64}(undef, length(experiments))
    for (i, job) in enumerate(experiments)
        cfg          = load_config(job_for(job))
        series       = get_1d_time_series(cfg)
        dt_sample    = cfg.nout * cfg.dt                     # sampling interval [s]
        freqs, power = power_spectrum(series, dt_sample)

        f_cpy          = freqs .* (86_400.0 * 365.25)
        idx            = argmax(power[2:end]) + 1            # skip DC bin
        basin_width[i] = cfg.Lx
        dom_freqs[i]   = f_cpy[idx]
    end

    order       = sortperm(basin_width)
    basin_width = basin_width[order]
    dom_freqs   = dom_freqs[order]

    # Linear regression in log–log space:  log10(f) ≈ slope · log10(Lx) + intercept
    design_matrix    = hcat(log10.(basin_width), ones(length(basin_width)))
    slope, intercept = design_matrix \ log10.(dom_freqs)
    freq_fit         = 10 .^ (slope .* log10.(basin_width) .+ intercept)

    fig = Figure(size = (900, 500), fontsize = 13)
    ax  = Axis(fig[1, 1];
        xlabel = "Basin Width  [m]",
        ylabel = "Frequency  [cycles / year]",
        xscale = log10,
        yscale = log10,
        title  = "Variation Of Dominant Mode Frequency To Basin Width (loglog Plot)")
    scatter!(ax, basin_width, dom_freqs; color = :black)
    lines!(ax,   basin_width, freq_fit;
        color = :crimson,
        label = L"\log_{10}(f) = %$(round(slope, digits=2)) \cdot \log_{10}(L_x) + %$(round(intercept, digits=2))"
    )
    axislegend(ax, position = :rt)

    out_file = joinpath(runs_dir, filename)
    save(out_file, fig)
    println("Dominant Frequency Shift Plot Saved → $out_file")
    @printf("Slope is %.2f, Intercept is %.2f\n", slope, intercept)
end
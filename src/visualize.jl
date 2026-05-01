using CairoMakie
using ProgressBars
using JLD2
using Statistics

# ============================================================
# 3-D animated surface of thermocline depth anomaly h(x,y,t)
# ============================================================
function make_animation(cfg::SimConfig)
    out_dir = cfg.output_dir

    h_frames = load(joinpath(out_dir, "h_frames.jld2"), "h_frames")
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

    Colorbar(fig[1, 2], hm;
        label         = "h′  [m]",
        width         = 15,
        ticklabelsize = 11)

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
# plot a time-series of the thermocline depth, as well 
# as the east thermocline depth and west thermocline depth
# defined as an average of the respective half of the domain
# ============================================================
function make_timeseries(cfg::SimConfig)
    out_dir = cfg.output_dir
    h_frames = load((joinpath(out_dir, "h_frames.jld2")), "h_frames")

    nframes = length(h_frames)

    h_means = mean.(h_frames)
    h_wests = [mean(mat[1:end÷4, :]) for mat in h_frames]
    h_easts = [mean(mat[(3 * end)÷4+1:end, :]) for mat in h_frames]
    days = [i * cfg.nout * cfg.dt / 86_400.0 for i in 1:nframes]
    
    f = Figure()

    ax = Axis(
        f[1,1], 
        xlabel = "Days", 
        ylabel = "Thermocline Depth Anomaly", 
        title="Time Series of Thermocline Depth Anomaly"
    )

    lines!(ax, days, h_means, color=:black)
    lines!(ax, days, h_wests, color=:red)
    lines!(ax, days, h_easts, color=:blue)

    save(joinpath(cfg.output_dir, "time_series.png"), f)
end

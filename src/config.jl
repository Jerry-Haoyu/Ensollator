using YAML

# ============================================================
# SimConfig — all run parameters in one place
# ============================================================
struct SimConfig
    # ---- Physical ----
    β::Float64          # Coriolis gradient df/dy at equator   [m⁻¹ s⁻¹]
    g_p::Float64        # Reduced gravity g×Δρ/ρ₀              [m s⁻²]
    r_damp::Float64     # Rayleigh damping rate 1/τ            [s⁻¹]
    H_mean::Float64     # Mean thermocline depth               [m]
    γ::Float64          # Walker-circulation coupling coefficient     [m⁻¹]
    # ---- Domain ----
    Lx::Float64         # Zonal domain length                  [m]
    Ly::Float64         # Meridional domain length             [m]
    nx::Int             # Grid cells in x
    ny::Int             # Grid cells in y
    # ---- Time ----
    dt::Float64         # Time step                            [s]
    nt::Int             # Total steps
    nout::Int           # Save frame every nout steps
    # ---- ic : linear westward deepening × equatorial Gaussian trapping ----
    h_amp::Float64      # Amplitude                            [m]
    # ---- Sponge layer (open north/south boundaries) ----
    sponge_width_frac::Float64  # sponge width as fraction of Ly     [1]
    sponge_rate_max::Float64    # max damping rate at domain edge     [s⁻¹]
    # ---- Animation ----
    framerate::Int      # Output frame rate                    [fps]
    # ---- Paths ----
    job_name::String
    output_dir::String
end

# ============================================================
# load_config — read runs/<job_name>/input/params.yml
# ============================================================
function load_config(job_name::String)
    input_file = joinpath("runs", job_name, "input", "params.yml")
    isfile(input_file) || error("Parameter file not found: $input_file")

    d = YAML.load_file(input_file; dicttype=Dict{String,Any})

    output_dir = joinpath("runs", job_name, "output")
    mkpath(output_dir)

    get_f(key, default) = Float64(get(d, key, default))
    get_i(key, default) = Int(get(d, key, default))

    damping_days    = get_f("damping_days",    5.0)
    sponge_rate_days = get_f("sponge_rate_days", 1.0)

    return SimConfig(
        get_f("beta",               2.3e-11),
        get_f("g_prime",            0.02),
        1.0 / (damping_days * 86_400.0),
        get_f("H_mean",             50.0),
        get_f("γ",                  1e-8),
        get_f("Lx",                 6.0e6),
        get_f("Ly",                 3.0e6),
        get_i("nx",                 256),
        get_i("ny",                 128),
        get_f("dt",                 1800.0),
        get_i("nt",                 1440),
        get_i("nout",               20),
        get_f("h_amp",              20.0),
        get_f("sponge_width_frac",  0.15),
        1.0 / (sponge_rate_days * 86_400.0),
        get_i("framerate",          12),
        job_name,
        output_dir,
    )
end

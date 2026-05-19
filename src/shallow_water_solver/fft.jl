"""
---------------------------------
BY : Claude Code
Prompted/Reviewed BY: Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""


using Statistics
using JLD2

include(joinpath(@__DIR__, "config.jl"))

# Recursive radix-2 Cooley–Tukey FFT.
# Splits the input into even- and odd-indexed samples, recurses, and combines
# with twiddle factors. Length must be a power of 2.
function fft(x::AbstractVector{<:Number})
    N = length(x)
    if N == 1
        return ComplexF64[x[1]]
    end
    ispow2(N) || error("FFT input length must be a power of 2 (got $N)")

    even = fft(@view x[1:2:end])
    odd  = fft(@view x[2:2:end])

    half = N ÷ 2
    X = Vector{ComplexF64}(undef, N)
    @inbounds for k in 0:half-1
        t = exp(-2π * im * k / N) * odd[k+1]
        X[k+1]      = even[k+1] + t
        X[k+1+half] = even[k+1] - t
    end
    return X
end

# Inverse FFT via conjugation trick: ifft(X) = conj(fft(conj(X))) / N.
function ifft(X::AbstractVector{<:Number})
    N = length(X)
    return conj.(fft(conj.(X))) ./ N
end

# Get the eastern-Pacific 1D time series from the saved h frames.
function get_1d_time_series(cfg::SimConfig)
    out_dir  = cfg.output_dir
    data     = load(joinpath(out_dir, "h_frames.jld2"))
    h_frames = data["h_frames"]
    h_easts  = [mean(mat[end÷2:end, :]) for mat in h_frames]
    return h_easts
end

# One-sided power spectrum of a real time series sampled at spacing dt.
# The series is mean-removed and zero-padded to the next power of 2.
function power_spectrum(series::AbstractVector{<:Real}, dt::Real)
    N      = length(series)
    N2     = 16 * nextpow(2, N)
    padded = zeros(Float64, N2)
    padded[1:N] .= series .- mean(series)

    X    = fft(padded)
    half = N2 ÷ 2
    freqs = [k / (N2 * dt) for k in 0:half-1]
    power = [abs2(X[k+1]) for k in 0:half-1]
    return freqs, power
end

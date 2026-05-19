"""
---------------------------------
BY : Claude Code
Prompted BY: Haoyu Tang
Github : Jerry_Haoyu 
---------------------------------
"""


using Test

include(joinpath(@__DIR__, "..", "src", "fft.jl"))

# Naive O(N^2) DFT used as a reference implementation.
function dft_reference(x)
    N = length(x)
    X = zeros(ComplexF64, N)
    for k in 0:N-1, n in 0:N-1
        X[k+1] += x[n+1] * exp(-2π * im * k * n / N)
    end
    return X
end

@testset "fft.jl" begin

    @testset "impulse → flat spectrum" begin
        x = ComplexF64[1, 0, 0, 0]
        @test fft(x) ≈ ComplexF64[1, 1, 1, 1]
    end

    @testset "constant → DC peak" begin
        x = ones(8)
        X = fft(x)
        @test X[1] ≈ 8
        @test all(abs.(X[2:end]) .< 1e-10)
    end

    @testset "matches naive DFT (random input)" begin
        for N in (2, 4, 8, 16, 32, 64)
            x = randn(N) .+ im .* randn(N)
            @test fft(x) ≈ dft_reference(x)
        end
    end

    @testset "pure cosine peaks at expected bin" begin
        N  = 64
        k0 = 5
        x  = [cos(2π * k0 * n / N) for n in 0:N-1]
        X  = fft(x)
        amp = abs.(X)
        # Real cosine: peaks at bin k0 and N-k0, each of magnitude N/2.
        @test argmax(amp) - 1 == k0 || argmax(amp) - 1 == N - k0
        @test isapprox(amp[k0 + 1],     N/2; atol = 1e-8)
        @test isapprox(amp[N - k0 + 1], N/2; atol = 1e-8)
    end

    @testset "ifft inverts fft" begin
        x = randn(32) .+ im .* randn(32)
        @test ifft(fft(x)) ≈ x
    end

    @testset "non-power-of-2 errors" begin
        @test_throws ErrorException fft(randn(6))
    end

    @testset "power_spectrum locates known frequency" begin
        dt   = 0.01
        N    = 512
        f0   = 7.0
        t    = (0:N-1) .* dt
        sig  = sin.(2π * f0 .* t)
        freqs, power = power_spectrum(sig, dt)
        kmax = argmax(power)
        @test isapprox(freqs[kmax], f0; atol = 1 / (N * dt))
    end

end

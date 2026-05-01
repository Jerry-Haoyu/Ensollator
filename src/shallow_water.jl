# ============================================================
# Equatorial Shallow Water Equations on the Beta Plane
# Arakawa C-grid
# ============================================================
#
# Governing equations (linearized):
#
#   ∂u/∂t =  β·y·v̄  - g'·∂h/∂x - r·u + γ_force
#   ∂v/∂t = -β·y·ū  - g'·∂h/∂y - r·v
#   ∂h/∂t = -H·(∂u/∂x + ∂v/∂y)
#
# Walker-circulation wind feedback (Bjerknes coupling):
#   γ_force = γ · (mean(h_east_quarter) - mean(h_west_quarter))   [m s⁻²]
#   Evaluated once per time step; feedback timescale >> dt.
#
# Arakawa C-grid placement (cell (i,j)):
#   h[i,j]  at cell centre    (x_c[i],        y_c[j]      )
#   u[i,j]  at east  face     (x_c[i]+dx/2,   y_c[j]      )
#   v[i,j]  at north face     (x_c[i],        y_c[j]+dy/2 )
#
# Boundary conditions
#   x (west/east)  : solid wall   u = 0  at  x = ±Lx/2
#                    u[0,j] = 0 implicit (west);  u[Nx,j] = 0 stored (east)
#   y (south/north): open ocean   Neumann (∂φ/∂n = 0) + quadratic sponge
#                    south: v[i,0] = v[i,1] (Neumann, implicit)
#                    north: h[i,Ny+1] = h[i,Ny], u[i,Ny+1] = u[i,Ny] (Neumann)
#
# Sponge layer (north and south, width = sponge_width_frac·Ly):
#   r_s(y) = r_sponge_max · ((distance to inner edge) / d_sponge)²
#   Applied to all fields: du -= r_s·u, dv -= r_s·v, dh -= r_s·h
#
# Temporal: classic 4th-order Runge-Kutta (RK4)
# Backend : ParallelStencil + CUDA — initialized in main.jl
# ============================================================

using Printf
using Statistics
using JLD2

include("visualize.jl")

# ============================================================
# Stencil Kernel: RHS of the Shallow Water Equations (C-grid)
# ============================================================
@parallel_indices (ix, iy) function compute_rhs!(
        du, dv, dh,           # output: time tendencies
        u,  v,  h,            # input : current state
        Y_h, Y_v,             # y-coords at h/u-points and v-points  [m]
        Rs_h, Rs_v,           # sponge damping rate at h/u and v pts [s⁻¹]
        β, g_p, r, H,         # physical parameters
        γ_force,              # Walker-circulation coupling scalar     [m s⁻²]
        dx, dy,               # grid spacings                          [m]
        Nx, Ny)               # grid size

    if 1 ≤ ix ≤ Nx && 1 ≤ iy ≤ Ny

        invdx = 1.0 / dx
        invdy = 1.0 / dy

        # ----------------------------------------------------------
        # dh at h-point (ix, iy)
        #
        # West wall  : u[0, j] = 0  (solid, implicit)
        # South open : v[ix,0] = v[ix,1]  (Neumann) → ∂v/∂y = 0 at south face
        # Sponge     : -Rs_h · h
        # ----------------------------------------------------------
        u_west  = (ix == 1) ? 0.0       : u[ix-1, iy]
        v_south = (iy == 1) ? v[ix, iy] : v[ix, iy-1]   # Neumann: v[0]=v[1]

        dh[ix, iy] = -H  * ((u[ix, iy] - u_west ) * invdx +
                             (v[ix, iy] - v_south) * invdy) -
                     Rs_h[ix, iy] * h[ix, iy]

        # ----------------------------------------------------------
        # du at u-point (ix+½, iy)
        #
        # East wall  : u[Nx, j] = 0  → tendency = 0
        # South open : v[ix,0] = v[ix,1] and v[ix+1,0] = v[ix+1,1] (Neumann)
        # Sponge     : -(r + Rs_h) · u
        # ----------------------------------------------------------
        if ix < Nx
            dh_dx = (h[ix+1, iy] - h[ix, iy]) * invdx

            # four v-points surrounding u[ix,iy]; Neumann south fills ix,0 and ix+1,0
            v_nw  = v[ix,   iy]
            v_ne  = v[ix+1, iy]
            v_sw  = (iy == 1) ? v[ix,   iy] : v[ix,   iy-1]
            v_se  = (iy == 1) ? v[ix+1, iy] : v[ix+1, iy-1]
            v_avg = 0.25 * (v_nw + v_ne + v_sw + v_se)
            
            du[ix, iy] = β * Y_h[ix, iy] * v_avg - g_p * dh_dx -
                         (r + Rs_h[ix, iy]) * u[ix, iy] + γ_force
        else
            du[ix, iy] = 0.0   # east-wall u-point stays at zero
        end

        # ----------------------------------------------------------
        # dv at v-point (ix, iy+½)
        #
        # West wall  : u[0, j] = 0  (solid, implicit)
        # North open : h[ix,Ny+1]=h[ix,Ny], u[ix,Ny+1]=u[ix,Ny] (Neumann)
        #              → dh_dy = 0,  u_ne = u_se,  u_nw = u_sw
        # Sponge     : -(r + Rs_v) · v
        # ----------------------------------------------------------
        u_se = u[ix,   iy]
        u_sw = (ix == 1) ? 0.0 : u[ix-1, iy]

        if iy < Ny
            dh_dy = (h[ix, iy+1] - h[ix, iy]) * invdy
            u_ne  = u[ix,   iy+1]
            u_nw  = (ix == 1) ? 0.0 : u[ix-1, iy+1]
        else
            # iy == Ny: north open BC (Neumann extrapolation)
            dh_dy = 0.0
            u_ne  = u_se    # u[ix,   Ny+1] = u[ix,   Ny]
            u_nw  = u_sw    # u[ix-1, Ny+1] = u[ix-1, Ny]
        end
        u_avg = 0.25 * (u_se + u_sw + u_ne + u_nw)

        dv[ix, iy] = -β * Y_v[ix, iy] * u_avg - g_p * dh_dy -
                     (r + Rs_v[ix, iy]) * v[ix, iy]
    end
    return
end

# ============================================================
# Tendency wrapper
# ============================================================
function tendencies!(du, dv, dh, u, v, h,
                     Y_h, Y_v, Rs_h, Rs_v,
                     β, g_p, r, H, γ_force::Float64, dx, dy, Nx, Ny)
    @parallel (1:Nx, 1:Ny) compute_rhs!(
        du, dv, dh, u, v, h, Y_h, Y_v, Rs_h, Rs_v,
        β, g_p, r, H, γ_force, dx, dy, Nx, Ny)
end

# ============================================================
# RK4 step — all stage arrays pre-allocated by caller
# ============================================================
function rk4_step!(u, v, h, Y_h, Y_v, Rs_h, Rs_v,
                   β, g_p, r, H, γ_force::Float64, dx, dy, dt, Nx, Ny,
                   k1u, k1v, k1h,
                   k2u, k2v, k2h,
                   k3u, k3v, k3h,
                   k4u, k4v, k4h,
                   tu,  tv,  th)
    h2 = 0.5 * dt
    d6 = dt / 6.0

    tendencies!(k1u, k1v, k1h, u,  v,  h,  Y_h, Y_v, Rs_h, Rs_v, β, g_p, r, H, γ_force, dx, dy, Nx, Ny)

    tu .= u .+ h2 .* k1u;  tv .= v .+ h2 .* k1v;  th .= h .+ h2 .* k1h
    tendencies!(k2u, k2v, k2h, tu, tv, th, Y_h, Y_v, Rs_h, Rs_v, β, g_p, r, H, γ_force, dx, dy, Nx, Ny)

    tu .= u .+ h2 .* k2u;  tv .= v .+ h2 .* k2v;  th .= h .+ h2 .* k2h
    tendencies!(k3u, k3v, k3h, tu, tv, th, Y_h, Y_v, Rs_h, Rs_v, β, g_p, r, H, γ_force, dx, dy, Nx, Ny)

    tu .= u .+ dt .* k3u;  tv .= v .+ dt .* k3v;  th .= h .+ dt .* k3h
    tendencies!(k4u, k4v, k4h, tu, tv, th, Y_h, Y_v, Rs_h, Rs_v, β, g_p, r, H, γ_force, dx, dy, Nx, Ny)

    u .+= d6 .* (k1u .+ 2.0 .* k2u .+ 2.0 .* k3u .+ k4u)
    v .+= d6 .* (k1v .+ 2.0 .* k2v .+ 2.0 .* k3v .+ k4v)
    h .+= d6 .* (k1h .+ 2.0 .* k2h .+ 2.0 .* k3h .+ k4h)
end


struct Coordinates
    _xs_km::Vector{Float64}
    _ys_km::Vector{Float64}
end

# ============================================================
# Main simulation — driven entirely by cfg::SimConfig
# ============================================================
function run_simulation(cfg::SimConfig)
    (; β, g_p, r_damp, H_mean, γ,
       Lx, Ly, nx, ny, dt, nt, nout, h_amp,
       sponge_width_frac, sponge_rate_max,
       job_name) = cfg

    dx = Lx / nx
    dy = Ly / ny

    d_sponge = sponge_width_frac * Ly   # sponge layer width [m]
    c_wave   = sqrt(g_p * H_mean)

    println("=== Equatorial Beta-Plane Shallow Water Model (C-grid) ===")
    @printf "  Job           : %s\n"                      job_name
    @printf "  Domain        : %.0f km × %.0f km\n"       Lx/1e3 Ly/1e3
    @printf "  Grid          : %d × %d   (dx=%.1f km, dy=%.1f km)\n" nx ny dx/1e3 dy/1e3
    @printf "  β             : %.2e m⁻¹ s⁻¹\n"            β
    @printf "  g′            : %.4f m s⁻²\n"              g_p
    @printf "  H             : %.1f m\n"                  H_mean
    @printf "  γ             : %.3f m⁻¹"                  γ
    @printf "  Damping time  : %.1f days\n"               1.0 / (r_damp * 86_400.0)
    @printf "  Wave speed c  : %.2f m s⁻¹\n"              c_wave
    @printf "  dt            : %.0f s,  CFL = %.4f\n"     dt  c_wave*dt/dx
    @printf "  Total time    : %.1f days (%d steps)\n"    nt*dt/86_400.0 nt
    @printf "  x BC          : solid walls (west & east)\n"
    @printf "  y BC          : open ocean  (Neumann + sponge)\n"
    @printf "  Sponge width  : %.0f km  (%.0f%% of Ly per side)\n" d_sponge/1e3 sponge_width_frac*100
    @printf "  Sponge r_max  : %.2e s⁻¹  (τ_min = %.2f days)\n"   sponge_rate_max  1/(sponge_rate_max*86_400.0)
    @printf "  Walker coupling: γ = %.3e  (h_east_quarter − h_west_quarter)\n" γ
    println()

    c_wave * dt / dx > 1.0 && @warn "CFL > 1 — simulation may be unstable!"

    # ---- Grid coordinates ----
    # h/u points at cell centres; v points at north faces (dy/2 above)
    xs_c = collect(LinRange(-Lx/2 + dx/2, Lx/2 - dx/2, nx))
    ys_c = collect(LinRange(-Ly/2 + dy/2, Ly/2 - dy/2, ny))
    ys_v = ys_c .+ dy/2   # y of north face j;  ys_v[ny] = Ly/2

    Y_h_cpu = Float64[ys_c[j] for i in 1:nx, j in 1:ny]
    Y_v_cpu = Float64[ys_v[j] for i in 1:nx, j in 1:ny]

    # ---- Sponge layer: quadratic profile in y ----
    # r_s(y) = r_max · ((dist to inner edge) / d_sponge)²,  r_s = 0 inside core
    Ly_half = Ly / 2
    function sponge_at(y)
        if y < -Ly_half + d_sponge
            s = (-Ly_half + d_sponge - y) / d_sponge
            return sponge_rate_max * s * s
        elseif y > Ly_half - d_sponge
            s = (y - (Ly_half - d_sponge)) / d_sponge
            return sponge_rate_max * s * s
        else
            return 0.0
        end
    end

    Rs_h_cpu = Float64[sponge_at(ys_c[j]) for i in 1:nx, j in 1:ny]
    Rs_v_cpu = Float64[sponge_at(ys_v[j]) for i in 1:nx, j in 1:ny]

    # ---- Initial condition: linear westward deepening × equatorial Gaussian trapping ----
    # h = -h_amp · x/(Lx/2)  ×  exp(-y²/2L_d²)
    # Linear factor: +h_amp at western wall, 0 at centre, -h_amp at eastern wall.
    # Gaussian factor: equatorial trapping over the deformation radius L_d = sqrt(c/β).
    L_d = sqrt(sqrt(g_p * H_mean) / β)   # equatorial deformation radius [m]
    h_cpu = Float64[
        -h_amp * xs_c[i] / (Lx/2) * exp(-ys_c[j]^2 / (2 * L_d^2))
        for i in 1:nx, j in 1:ny
    ]

    Y_h  = Data.Array(Y_h_cpu)
    Y_v  = Data.Array(Y_v_cpu)
    Rs_h = Data.Array(Rs_h_cpu)
    Rs_v = Data.Array(Rs_v_cpu)
    u    = @zeros(nx, ny)
    v    = @zeros(nx, ny)
    h    = Data.Array(h_cpu)

    # Pre-allocate RK4 stage arrays
    k1u = @zeros(nx, ny);  k1v = @zeros(nx, ny);  k1h = @zeros(nx, ny)
    k2u = @zeros(nx, ny);  k2v = @zeros(nx, ny);  k2h = @zeros(nx, ny)
    k3u = @zeros(nx, ny);  k3v = @zeros(nx, ny);  k3h = @zeros(nx, ny)
    k4u = @zeros(nx, ny);  k4v = @zeros(nx, ny);  k4h = @zeros(nx, ny)
    tu  = @zeros(nx, ny);  tv  = @zeros(nx, ny);  th  = @zeros(nx, ny)

    nframes  = nt ÷ nout
    h_frames = Vector{Matrix{Float64}}(undef, nframes)

    println("Running simulation…")
    t_wall = time()
    frame_idx = 0
    
    boundary_width = 4 # boundary region is L_x/boundary_width
    west_boundary_end = nx ÷ boundary_width              # western-quarter  column index boundary
    east_boundary_start = ((boundary_width - 1) * nx) ÷ boundary_width + 1   # eastern-quarter  column index start

    for it in 1:nt
        γ_force = max(min(γ * Float64(mean(h[east_boundary_start:end, :]) - mean(h[1:west_boundary_end, :])), -1e-9), -5e-9)

        rk4_step!(u, v, h, Y_h, Y_v, Rs_h, Rs_v,
                  β, g_p, r_damp, H_mean, γ_force, dx, dy, dt, nx, ny,
                  k1u, k1v, k1h, k2u, k2v, k2h,
                  k3u, k3v, k3h, k4u, k4v, k4h,
                  tu, tv, th)

        # east-wall BC: u[Nx, j] = 0 (kernel already zeroes its tendency;
        # this guards against floating-point drift across RK4 stages)
        u[end, :] .= 0.0

        if it % nout == 0
            frame_idx += 1
            snap = Array(h)
            h_frames[frame_idx] = snap
            @printf("step %5d / %d  (day %6.2f)  max|h| = %8.4f m  γ_force = %+.3e m s⁻²\n",
                    it, nt, it*dt/86_400.0, maximum(abs, snap), γ_force)
        end
    end

    @printf "\nDone in %.1f s.  Saving data …\n\n" (time() - t_wall)

    xs_km = xs_c ./ 1e3
    ys_km = ys_c ./ 1e3
    
    coord = Coordinates(xs_km,ys_km)

    jldsave(joinpath(cfg.output_dir, "h_frames.jld2"); h_frames)
    jldsave(joinpath(cfg.output_dir, "simulation_coords.jld2"); coord)

    # make_animation(h_frames, xs_km, ys_km, cfg)
end

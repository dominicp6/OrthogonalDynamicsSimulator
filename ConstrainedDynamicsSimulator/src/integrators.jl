using ProgressMeter
using Molly
include("./simulation_helpers.jl")

# Euler-Maruyama method
@inline function euler_maruyama!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
    masses_nounits, friction_nounits, k_bT, dt = init_simulation(sys, sim, n_threads, run_loggers)

    noise_scaling = sqrt(2 * k_bT * dt / friction_nounits) / (1^(1/2))  # we assume an identity mass matrix
    @show progress = Progress(n_steps)
    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
        @. x .+= (P_F + k_bT * divP) * (dt / friction_nounits) + noise_scaling * Pnoise
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
        next!(progress)
    end

    return sys
end

# Euler-Maruyama with split time and lambda correction
@inline function euler_maruyama_split_time!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), tol=1e-6, max_iter=50, run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
    masses_nounits, friction_nounits, k_bT, dt = init_simulation(sys, sim, n_threads, run_loggers)
    
    noise_scaling = sqrt(2 * k_bT * dt / friction_nounits) / (1^(1/2))  # we assume an identity mass matrix
    constrained_CV = sim.φ_grid(x)
    stopping_condition_counts = [0, 0, 0]
    @show progress = Progress(n_steps)
    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
        @. x .+= (P_F / friction_nounits) * dt + noise_scaling * Pnoise
        λ, stopping_condition = find_lambda(sim.φ_grid, x, divP, k_bT, friction_nounits, dt, constrained_CV; tol=tol, max_iter=max_iter)
        stopping_condition_counts[stopping_condition] += 1
        @. x .+= (k_bT * divP / friction_nounits) * λ * dt
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
        next!(progress)
    end

    return sys, stopping_condition_counts
end

# PVD1 method
@inline function PVD1!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
    masses_nounits, friction_nounits, k_bT, dt = init_simulation(sys, sim, n_threads, run_loggers)
    
    noise_scaling = sqrt(k_bT * dt / (2 * friction_nounits))  # nm
    F_x_barₖ = (P_F + k_bT .* divP) ./ friction_nounits
    F_x_barₖ₋₁ = copy(F_x_barₖ)
    x_barₖ = copy(x)
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
        compute_Pvec_x!(Pnoise, gradφ, φ=sim.φ_grid, vec=noise, x=x)
        @. x_barₖ .= x + noise_scaling * Pnoise
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x_barₖ)
        @. F_x_barₖ .= (P_F + k_bT * divP) / friction_nounits
        @. x .+= dt * F_x_barₖ + 2 * noise_scaling * compute_Pvec_x(φ=sim.φ_grid, vec=noise, x=x + (1/4) * dt * F_x_barₖ₋₁)[1]
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
        F_x_barₖ₋₁ .= F_x_barₖ
        next!(progress)
    end

    return sys
end

# PVD2 method 
@inline function PVD2!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
    masses_nounits, friction_nounits, k_bT, dt = init_simulation(sys, sim, n_threads, run_loggers)
    
    noise_scaling = sqrt(k_bT * dt / (2 * friction_nounits))  # nm
    F_x_barₖ = (P_F + k_bT .* divP) ./ friction_nounits
    F_x_barₖ₋₁ = copy(F_x_barₖ)
    x_barₖ = copy(x)
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
        compute_Pvec_x!(Pnoise, gradφ, φ=sim.φ_grid, vec=noise, x=x)
        @. x_barₖ .= x .+ noise_scaling .* Pnoise
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x_barₖ)
        @. F_x_barₖ .= (P_F + k_bT * divP) / friction_nounits
        x .+= dt * F_x_barₖ + MT2(x .+ dt * F_x_barₖ₋₁ / 4, dt, k_bT, friction_nounits, sim.φ_flat, noise)
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
        F_x_barₖ₋₁ .= F_x_barₖ
        next!(progress)
    end

    return sys
end

# PVD2 with efficient implementation of MT2
@inline function PVD2_efficient!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
    masses_nounits, friction_nounits, k_bT, dt = init_simulation(sys, sim, n_threads, run_loggers)
    
    noise_scaling = sqrt(k_bT * dt / (2 * friction_nounits))  # nm
    F_x_barₖ = (P_F + k_bT .* divP) ./ friction_nounits
    F_x_barₖ₋₁ = copy(F_x_barₖ)
    x_barₖ = copy(x)
    progress = Progress(n_steps)

    # Initialising memory for quantities used in MT2 noise integrator
    sqrt_dt = sqrt(dt)
    x0_flat = similar(flatten(x))
    noise_flat = similar(x0_flat)
    d = length(x0_flat)
    gradφ_placeholder = similar(x0_flat) 
    prefactor1 = sqrt(k_bT / (2 * friction_nounits)) * (1)^(-(1/2)) / (friction_nounits)^(1/2)
    prefactor2 = sqrt(2 * k_bT) * (1)^(-(1/2))
    prefactor3 = sqrt(k_bT / (2 * friction_nounits)) * (1)^(-(1/2)) 
    prefactor4 = sqrt(k_bT / friction_nounits) * (1)^(-(1/2))
    χ = rand([-1.0, 1.0], d)
    result = similar(x0_flat)
    result_final = similar(x0_flat)
    PJa = similar(x0_flat)
    arg1 = similar(x0_flat)
    arg2 = similar(x0_flat)
    Pxi1 = similar(x0_flat)
    Pxi2 = similar(x0_flat)
    ei = zeros(d)
    J_row = similar(x0_flat)
    Pχ = similar(x0_flat)
    proj1 = similar(x0_flat)
    proj2 = similar(x0_flat)
    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
        compute_Pvec_x!(Pnoise, gradφ, φ=sim.φ_grid, vec=noise, x=x)
        @. x_barₖ .= x + noise_scaling * Pnoise
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x_barₖ)
        @. F_x_barₖ .= (P_F + k_bT * divP) / friction_nounits
        x .+= dt * F_x_barₖ + MT2_efficient!(x .+ dt * F_x_barₖ₋₁ / 4, 
                                                    dt, 
                                                    sqrt_dt, 
                                                    sim.φ_flat, 
                                                    noise, 
                                                    x0_flat, 
                                                    noise_flat, 
                                                    d, 
                                                    gradφ_placeholder, 
                                                    prefactor1, 
                                                    prefactor2, 
                                                    prefactor3, 
                                                    prefactor4, 
                                                    χ, 
                                                    result, 
                                                    result_final, 
                                                    PJa, 
                                                    arg1,
                                                    arg2,
                                                    Pxi1, 
                                                    Pxi2, 
                                                    ei, 
                                                    J_row, 
                                                    Pχ, 
                                                    proj1, 
                                                    proj2)
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
        F_x_barₖ₋₁ .= F_x_barₖ
        next!(progress)
    end

    return sys
end


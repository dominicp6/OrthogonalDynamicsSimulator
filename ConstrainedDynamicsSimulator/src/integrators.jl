using ProgressMeter
include("./simulation_helpers.jl")

# Euler-Maruyama method
@inline function euler_maruyama!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, params = init_simulation(sys, sim, n_threads, run_loggers)
    noise_scaling = sqrt(2 * params.k_bT * params.dt / params.friction_nounits)  # nm
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, params)
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
        x .+= ((P_F + params.k_bT * divP) ./ params.friction_nounits) .* params.dt .+ noise_scaling .* Pnoise
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, params)
        next!(progress)
    end

    return sys
end

# Euler-Maruyama with split time and lambda correction
@inline function euler_maruyama_split_time!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), tol=1e-6, max_iter=50, run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, params = init_simulation(sys, sim, n_threads, run_loggers)
    noise_scaling = sqrt(2 * params.k_bT * params.dt / params.friction_nounits)  # nm
    constrained_CV = sim.φ_grid(x)
    stopping_condition_counts = [0, 0, 0]
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, params)
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
        x .+= (P_F ./ params.friction_nounits) .* params.dt .+ noise_scaling .* Pnoise
        λ, stopping_condition = find_lambda(sim.φ_grid, x, divP, params.k_bT, params.friction_nounits, params.dt, constrained_CV; tol=tol, max_iter=max_iter)
        stopping_condition_counts[stopping_condition] += 1
        x .+= (params.k_bT * divP ./ params.friction_nounits) .* λ .* params.dt
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, params)
        next!(progress)
    end

    return sys, stopping_condition_counts
end

# PVD1 method
@inline function PVD1!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, params = init_simulation(sys, sim, n_threads, run_loggers)
    noise_scaling = sqrt(params.k_bT * params.dt / (2 * params.friction_nounits))  # nm
    F_x_barₖ = (P_F + params.k_bT .* divP) ./ params.friction_nounits
    F_x_barₖ₋₁ = copy(F_x_barₖ)
    x_barₖ = copy(x)
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, params)
        compute_Pvec_x!(Pnoise, gradφ, φ=sim.φ_grid, vec=noise, x=x)
        x_barₖ .= x .+ noise_scaling .* Pnoise
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x_barₖ)
        F_x_barₖ .= (P_F + params.k_bT .* divP) ./ params.friction_nounits
        x .+= params.dt * F_x_barₖ + 2 * noise_scaling * compute_Pvec_x(φ=sim.φ_grid, vec=noise, x=x + (1/4) * params.dt * F_x_barₖ₋₁)[1]
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, params)
        F_x_barₖ₋₁ .= F_x_barₖ
        next!(progress)
    end

    return sys
end

# PVD2 method (assuming MT2 is defined elsewhere)
@inline function PVD2!(sys, sim::CVConstrainedOverdampedLangevin, n_steps::Integer; n_threads=Threads.nthreads(), run_loggers=true)
    x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, params = init_simulation(sys, sim, n_threads, run_loggers)
    noise_scaling = sqrt(params.k_bT * params.dt / (2 * params.friction_nounits))  # nm
    F_x_barₖ = (P_F + params.k_bT .* divP) ./ params.friction_nounits
    F_x_barₖ₋₁ = copy(F_x_barₖ)
    x_barₖ = copy(x)
    progress = Progress(n_steps)

    for step_n in 1:n_steps
        compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, params)
        compute_Pvec_x!(Pnoise, gradφ, φ=sim.φ_grid, vec=noise, x=x)
        x_barₖ .= x .+ noise_scaling .* Pnoise
        compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x_barₖ)
        F_x_barₖ .= (P_F + params.k_bT .* divP) ./ params.friction_nounits
        x .+= params.dt * F_x_barₖ + MT2(x .+ params.dt * F_x_barₖ₋₁ / 4, params.dt, params.k_bT, params.friction_nounits, sim.φ_flat, noise)
        neighbors = finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, params)
        F_x_barₖ₋₁ .= F_x_barₖ
        next!(progress)
    end

    return sys
end


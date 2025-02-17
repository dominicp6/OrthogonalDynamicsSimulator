export ConstrainedIntegrator_PVD2, simulate!

struct ConstrainedIntegrator_PVD2{T, S, F} 
    dt::T
    sigma::S
    φ::F
end

ConstrainedIntegrator_PVD2(; dt, sigma, φ) = ConstrainedIntegrator_PVD2(dt, sigma, φ)

@inline function simulate!(sys,
                           sim::ConstrainedIntegrator_PVD2,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true)

    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)

    forces_nounits_t = ustrip_vec.(similar(sys.coords))
    forces_t = forces_nounits_t .* sys.force_units
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    num_coords = length(sys.velocities)
    noise = randn(num_coords)

    PgradV, gradφ = compute_PgradV(sim.φ, sys.coords, forces_t)
    divP = compute_divP(sim.φ, gradφ, sys.coords)
    F_x_barₖ₋₁ = - PgradV + sim.sigma^2 * divP / 2
    final_system_coords = similar(sys.coords)

    for step_n in 1:n_steps
        forces_nounits_t .= forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n; n_threads=n_threads)
        forces_t .= forces_nounits_t .* sys.force_units
        noise .= randn(num_coords)
        
        # Insert constrained integrator step here
        x_barₖ .= sys.coords + 0.5 * sqrt(sim.dt) * sim.sigma * compute_PdW(gradφ, noise) 
        PgradV .= compute_PgradV(sim.φ, x_barₖ, forces_nounits_t)
        divP .= compute_divP(sim.φ, x_barₖ)
        F_x_barₖ .= - PgradV + sim.sigma^2 * divP / 2
        sys.coords .+= sim.dt * F_x_barₖ + MT2!(sys, sim, x0, F_x_barₖ₋₁, num_coords, noise)
        
        final_system_coords .= wrap_coords.(sys.coords, (sys.boundary,))
        # Temporarily change system coords to x_barₖ for logging
        sys.coords .= x_barₖ
        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
        apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, current_forces=forces_t)
        # Revert system coords back to their intended values
        sys.coords .= final_system_coords

        F_x_barₖ₋₁ .= F_x_barₖ
    end

    return sys
end
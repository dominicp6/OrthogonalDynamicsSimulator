using Molly
using ProgressMeter
using Random
export CVConstrainedOverdampedLangevin, euler_maruyama!, euler_maruyama_split_time!

function are_units_same(u1::Unitful.Units, u2::Unitful.Units)
    # Check if dimensions match
    if dimension(u1) != dimension(u2)
        return false
    end
    # Check if they are numerically equivalent (e.g., 1 u1 = 1 u2 after conversion)
    q1 = 1 * u1
    q2 = 1 * u2
    return uconvert(u1, q2) == q1
end

struct CVConstrainedOverdampedLangevin{picosecond, kelvin, inverse_picosecond, F1, F2} 
    dt::picosecond
    T::kelvin
    γ::inverse_picosecond
    φ_grid::F1
    φ_flat::F2
    remove_CM_motion::Int
end

CVConstrainedOverdampedLangevin(; dt, T, γ, φ_grid, φ_flat, remove_CM_motion=1) = CVConstrainedOverdampedLangevin(dt, T, γ, φ_grid, φ_flat, Int(remove_CM_motion))

function sample_noise_vector(n)
    return [@SVector randn(3) for _ in 1:n]
end

function sample_noise_vector!(data)
    for i in eachindex(data)
        data[i] = @SVector randn(3)
    end
end

@inline function euler_maruyama!(sys,
                           sim::CVConstrainedOverdampedLangevin,
                           n_steps::Integer;
                           n_threads::Integer=Threads.nthreads(),
                           run_loggers=true)

    if length(sys.constraints) > 0
        @warn "CVConstrainedOverdampedLangevin is not currently compatible with constraints, " *
        "constraints will be ignored"
    end

    # Unit assertions 
    uX, ut, uT, uk, uγ, uM, uF = unit(sys.coords[1][1]), unit(sim.dt), unit(sim.T), unit(sys.k), unit(sim.γ), unit(masses(sys)[1]), sys.force_units
    @assert uX == u"nm"
    @assert ut == u"ps"
    @assert uT == u"K"
    @assert uk == u"kJ/(mol*K)"
    @assert uγ == u"ps^-1"
    @assert uM == u"g/mol"
    @assert uF == u"kJ/(nm*mol)"

    # Physical quantities
    masses_nounits = ustrip.(masses(sys))
    friction_nounits = ustrip(sim.γ)
    k_bT = ustrip(sys.k) * ustrip(sim.T)   # kJ mol^-1
    dt = ustrip(sim.dt)                    # ps
    
    # Coordinate initialisation
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    x_barₖ = ustrip.(sys.coords)           # nm
    num_coords = length(x_barₖ)

    # Forces and acceleration initialisation
    forces_nounits_t = ustrip_vec.(similar(sys.coords))   # kJ nm^-1 mol^-1
    forces_t = forces_nounits_t .* sys.force_units
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    accels_nounits_t = forces_nounits_t ./ 1  # we assume an identity mass matrix (in units of g mol^-1)
    @assert are_units_same((uX / ut^2), uF / uM)        
    
    # Noise initialisation
    noise = similar(sys.velocities)
    noise = sample_noise_vector(num_coords) # dimensionless 
    noise_scaling = sqrt(2 * k_bT * dt / friction_nounits) / (1^(1/2))  # we assume an identity mass matrix (in units of g mol^-1)
    # @assert are_units_same(uX, sqrt(uk * uT * ut / uγ) / sqrt(uM))

    # Constrained dynamics initialisation
    P_M_inv_F, gradφ = compute_Pvec_x(φ_grid=sim.φ_grid, vec=accels_nounits_t, x=x_barₖ)
    Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
    @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
    divP = compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x_barₖ)   # nm^-1
    @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))
    
    @show progress = Progress(n_steps)
    for step_n in 1:n_steps
        forces_nounits_t .= Molly.forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n; n_threads=n_threads)
        accels_nounits_t .= forces_nounits_t ./ 1  # we assume an identity mass matrix (in units of g mol^-1)
        
        sample_noise_vector!(noise)

        compute_Pvec_x!(P_M_inv_F, gradφ, φ_grid=sim.φ_grid, vec=accels_nounits_t, x=x_barₖ)
        divP .= compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x_barₖ)
        compute_Pvec!(Pnoise, gradφ=gradφ, vec=noise)
        x_barₖ .+= ((P_M_inv_F + k_bT * divP) ./friction_nounits) .* dt .+ noise_scaling .* Pnoise   # no explicit division by m here because we assume an identity mass matrix (in units of g mol^-1)
        sys.coords .= wrap_coords.(x_barₖ * uX, (sys.boundary,))

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
        apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, current_forces=forces_nounits_t * uF)

        next!(progress)

    end

    return sys
end

@inline function find_lambda(φ_grid, x_barₖ, divP, k_bT, friction_nounits, dt, target_CV; tol=1e-6, max_iter=50)
    # Define the correction term
    correction = (k_bT * divP ./ friction_nounits) .* dt
    # Define the function f(λ) = φ_grid(x_barₖ + correction * λ) - target_CV
    f(λ) = φ_grid(x_barₖ .+ correction .* λ) - target_CV
    
    # Initial guess for λ
    λ = 0.0
    iter = 0
    
    # Newton-Raphson iteration with Zygote for derivative
    while iter < max_iter
        # Evaluate function
        f_val = f(λ)
        
        # Check convergence
        if abs(f_val) < tol
            return λ, 1
        end
        
        # Compute derivative using Zygote
        f_deriv = Zygote.gradient(f, λ)[1]  # Gradient returns a tuple, take the first element
        
        # Check if derivative is zero or nearly zero to avoid division issues
        if abs(f_deriv) < 1e-12
            @warn "Derivative near zero, stopping λ iteration early"
            return λ, 2
        end
        
        # Update λ
        λ_new = λ - f_val / f_deriv
        
        # Prevent overshooting by limiting step size if needed
        if abs(λ_new - λ) > 10.0  # Arbitrary large step limit
            λ_new = λ + sign(λ_new - λ) * 10.0
        end
        
        λ = λ_new
        iter += 1
    end
    
    return λ, 3
end


@inline function euler_maruyama_split_time!(sys,
    sim::CVConstrainedOverdampedLangevin,
    n_steps::Integer;
    n_threads::Integer=Threads.nthreads(),
    tol=1e-6, 
    max_iter=50,
    run_loggers=true)

    if length(sys.constraints) > 0
    @warn "CVConstrainedOverdampedLangevin is not currently compatible with constraints, " *
    "constraints will be ignored"
    end

    # Unit assertions 
    uX, ut, uT, uk, uγ, uM, uF = unit(sys.coords[1][1]), unit(sim.dt), unit(sim.T), unit(sys.k), unit(sim.γ), unit(masses(sys)[1]), sys.force_units
    @assert uX == u"nm"
    @assert ut == u"ps"
    @assert uT == u"K"
    @assert uk == u"kJ/(mol*K)"
    @assert uγ == u"ps^-1"
    @assert uM == u"g/mol"
    @assert uF == u"kJ/(nm*mol)"

    # Physical quantities
    masses_nounits = ustrip.(masses(sys))
    friction_nounits = ustrip(sim.γ)
    k_bT = ustrip(sys.k) * ustrip(sim.T)   # kJ mol^-1
    dt = ustrip(sim.dt)                    # ps

    # Coordinate initialisation
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    x_barₖ = ustrip.(sys.coords)           # nm
    num_coords = length(x_barₖ)

    # Forces and acceleration initialisation
    forces_nounits_t = ustrip_vec.(similar(sys.coords))   # kJ nm^-1 mol^-1
    forces_t = forces_nounits_t .* sys.force_units
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    accels_nounits_t = forces_nounits_t ./ 1  # we assume an identity mass matrix (in units of g mol^-1)
    @assert are_units_same((uX / ut^2), uF / uM)        

    # Noise initialisation
    noise = similar(sys.velocities)
    noise = sample_noise_vector(num_coords) # dimensionless 
    noise_scaling = sqrt(2 * k_bT * dt / friction_nounits) / (1^(1/2))  # we assume an identity mass matrix (in units of g mol^-1)
    # @assert are_units_same(uX, sqrt(uk * uT * ut / uγ) / sqrt(uM))

    # Constrained dynamics initialisation
    P_M_inv_F, gradφ = compute_Pvec_x(φ_grid=sim.φ_grid, vec=accels_nounits_t, x=x_barₖ)
    Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
    @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
    divP = compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x_barₖ)   # nm^-1
    @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))

    # Keep track of constrained CV
    constrained_CV = sim.φ_grid(x_barₖ)
    stopping_condition_counts = [0, 0, 0]
    @show progress = Progress(n_steps)
    for step_n in 1:n_steps
        forces_nounits_t .= Molly.forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n; n_threads=n_threads)
        accels_nounits_t .= forces_nounits_t ./ 1  # we assume an identity mass matrix (in units of g mol^-1)

        sample_noise_vector!(noise)

        compute_Pvec_x!(P_M_inv_F, gradφ, φ_grid=sim.φ_grid, vec=accels_nounits_t, x=x_barₖ)
        divP .= compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x_barₖ)
        compute_Pvec!(Pnoise, gradφ=gradφ, vec=noise)
        x_barₖ .+= ((P_M_inv_F) ./friction_nounits) .* dt .+ noise_scaling .* Pnoise   # no explicit division by m here because we assume an identity mass matrix (in units of g mol^-1)
        λ, stopping_condition = find_lambda(sim.φ_grid, x_barₖ, divP, k_bT, friction_nounits, dt, constrained_CV, tol=tol, max_iter=max_iter)
        stopping_condition_counts[stopping_condition] += 1
        x_barₖ .+= (k_bT * divP ./ friction_nounits) .* λ .* dt
        sys.coords .= wrap_coords.(x_barₖ * uX, (sys.boundary,))

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
        apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, current_forces=forces_nounits_t * uF)

        next!(progress)

    end

    return sys, stopping_condition_counts
end


# For benchmarking against the standard molly implementation of regular overdamped langevin dynamics
@inline function simulateOverdamped!(sys,
    sim::OverdampedLangevin,
    n_steps::Integer;
    n_threads::Integer=Threads.nthreads(),
    run_loggers=true,
    rng=Random.default_rng())
    if length(sys.constraints) > 0
        @warn "OverdampedLangevin is not currently compatible with constraints, " *
        "constraints will be ignored"
    end
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    forces_nounits_t = ustrip_vec.(similar(sys.coords))
    forces_t = forces_nounits_t .* sys.force_units
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    accels_t = forces_t ./ masses(sys)
    noise = similar(sys.velocities)

    for step_n in 1:n_steps
        forces_nounits_t .= Molly.forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n;
                            n_threads=n_threads)
        forces_t .= forces_nounits_t .* sys.force_units
        accels_t .= forces_t ./ masses(sys)

        random_velocities!(noise, sys, sim.temperature; rng=rng)
        sys.coords .+= (accels_t ./ sim.friction) .* sim.dt .+ sqrt((2 / sim.friction) * sim.dt) .* noise
        sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))

        if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
            remove_CM_motion!(sys)
        end

        neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n;
                    n_threads=n_threads)

        apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads,
        current_forces=forces_t)
    end
    
    return sys
end

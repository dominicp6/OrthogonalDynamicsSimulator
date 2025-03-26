using Unitful, Molly, Zygote, Random

function sample_noise_vector(n)
    return [@SVector randn(3) for _ in 1:n]
end

function sample_noise_vector!(data)
    for i in eachindex(data)
        data[i] = @SVector randn(3)
    end
end

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

# Shared struct for simulation parameters (excluding noise_scaling)
struct SimParams
    dt::Float64           # ps
    k_bT::Float64         # kJ mol^-1
    friction_nounits::Float64  # ps^-1
    n_threads::Int
    run_loggers::Bool
end

# Shared initialisation function
@inline function init_simulation(sys, sim::CVConstrainedOverdampedLangevin, n_threads::Integer, run_loggers::Bool)
    # Warn if constraints are present
    isempty(sys.constraints) || @warn "CVConstrainedOverdampedLangevin ignores constraints"

    # Unit assertions
    uX, ut, uT, uk, uγ, uM, uF = unit(sys.coords[1][1]), unit(sim.dt), unit(sim.T), unit(sys.k), unit(sim.γ), unit(masses(sys)[1]), sys.force_units
    @assert uX == u"nm" && ut == u"ps" && uT == u"K" && uk == u"kJ/(mol*K)" && uγ == u"ps^-1" && uM == u"g/mol" && uF == u"kJ/(nm*mol)"

    # Physical quantities
    friction_nounits = ustrip(sim.γ)
    k_bT = ustrip(sys.k) * ustrip(sim.T)
    dt = ustrip(sim.dt)

    # Coordinate initialisation
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    x = ustrip.(sys.coords)
    num_coords = length(x)

    # Forces and noise initialisation
    forces_nounits_t = ustrip_vec.(similar(sys.coords))
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    noise = sample_noise_vector(num_coords)

    # Constrained dynamics initialisation
    accels_nounits_t = forces_nounits_t  # Identity mass matrix
    P_F, gradφ = compute_Pvec_x(φ=sim.φ_grid, vec=accels_nounits_t, x=x)
    Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
    divP = compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x)

    # Unit consistency checks
    @assert are_units_same(uX / ut^2, uF / uM)
    @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
    @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))

    return (x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer,
            SimParams(dt, k_bT, friction_nounits, n_threads, run_loggers))
end

# Core update functions
@inline function compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, params::SimParams)
    forces_nounits_t .= Molly.forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n; n_threads=params.n_threads)
    accels_nounits_t .= forces_nounits_t  # Identity mass matrix
    sample_noise_vector!(noise)
end

@inline function compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
    compute_Pvec_x!(P_F, gradφ, φ=sim.φ_grid, vec=accels_nounits_t, x=x)
    divP .= compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x)
    compute_Pvec!(Pnoise, gradφ=gradφ, vec=noise)
end

@inline function finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, params::SimParams)
    sys.coords .= wrap_coords.(x * u"nm", (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=params.n_threads)
    apply_loggers!(sys, neighbors, step_n, params.run_loggers; n_threads=params.n_threads, current_forces=forces_nounits_t * u"kJ/(nm*mol)")
    return neighbors
end

@inline function find_lambda(φ_grid, x_barₖ, divP, k_bT, friction_nounits, dt, target_CV; tol=1e-6, max_iter=50)
    correction = (k_bT * divP ./ friction_nounits) .* dt
    f(λ) = φ_grid(x_barₖ .+ correction .* λ) - target_CV
    
    λ = 0.0
    for iter in 1:max_iter
        f_val = f(λ)
        if abs(f_val) < tol 
            return λ, 1
        end
        
        f_deriv = Zygote.gradient(f, λ)[1]
        if abs(f_deriv) < 1e-12 
            @warn "Derivative near zero"
            return λ, 2
        end
        
        λ_new = λ - f_val / f_deriv
        λ = λ_new
    end
    
    return λ, 3
end

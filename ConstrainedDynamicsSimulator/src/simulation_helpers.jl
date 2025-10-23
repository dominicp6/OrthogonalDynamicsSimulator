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

# @inline function init_simulation(sys, sim::CVConstrainedOverdampedLangevin, n_threads::Integer, run_loggers::Bool)
#     # Warn if constraints are present
#     if length(sys.constraints) > 0
#         @warn "CVConstrainedOverdampedLangevin is not currently compatible with constraints, " *
#               "constraints will be ignored"
#     end

#     # Unit assertions
#     uX, ut, uT, uk, uγ, uM, uF = unit(sys.coords[1][1]), unit(sim.dt), unit(sim.T), unit(sys.k), unit(sim.γ), unit(masses(sys)[1]), sys.force_units
#     @assert uX == u"nm"
#     @assert ut == u"ps"
#     @assert uT == u"K"
#     @assert uk == u"kJ/(mol*K)"
#     @assert uγ == u"ps^-1"
#     @assert uM == u"g/mol"
#     @assert uF == u"kJ/(nm*mol)"

#     # Physical quantities
#     masses_nounits = ustrip.(masses(sys))
#     friction_nounits = ustrip(sim.γ)
#     k_bT = ustrip(sys.k) * ustrip(sim.T)   # kJ mol^-1
#     dt = ustrip(sim.dt)                    # ps

#     # Coordinate initialization
#     sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
#     !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
#     neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
#     apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
#     x = ustrip.(sys.coords)                # nm
#     num_coords = length(x)

#     # Forces and acceleration initialization
#     forces_nounits_t = ustrip_vec.(similar(sys.coords))   # kJ nm^-1 mol^-1
#     forces_t = forces_nounits_t .* sys.force_units
#     forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
#     accels_nounits_t = forces_nounits_t ./ 1  # Identity mass matrix (division by 1 is a no-op)
#     @assert are_units_same((uX / ut^2), uF / uM)

#     # Noise initialization
#     noise = similar(sys.velocities)
#     noise = sample_noise_vector(num_coords)  # dimensionless
#     # Note: / (1^(1/2)) is redundant as it equals 1, but kept for fidelity to original
#     compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, 0, n_threads)
#     # @assert are_units_same(uX, sqrt(uk * uT * ut / uγ) / sqrt(uM))  # Commented out as in original

#     # Constrained dynamics initialization
#     P_F, gradφ = compute_Pvec_x(φ=sim.φ_grid, vec=accels_nounits_t, x=x)
#     Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
#     @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
#     divP = compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x)   # nm^-1
#     @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))

#     # Return all necessary variables
#     return (x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, 
#             masses_nounits, friction_nounits, k_bT, dt)
# end

@inline function init_simulation(sys, sim::CVConstrainedOverdampedLangevin, n_threads::Integer, run_loggers::Bool, compute_ergodic_integral::Bool, quantity::Union{Function, Nothing})
    # Warn if constraints are present
    if length(sys.constraints) > 0
        @warn "CVConstrainedOverdampedLangevin is not currently compatible with constraints, " *
              "constraints will be ignored"
    end

    if compute_ergodic_integral && quantity === nothing
        @error "A quantity function must be provided if you wish to compute an ergodic integral"
    end

    if !compute_ergodic_integral && quantity !== nothing
        @warn "You have indicated to not compute an ergodic integral but have provided a quantity function. Are you sure?"
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

    # Coordinate initialization
    sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
    !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
    neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
    apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
    x = ustrip.(sys.coords)                # nm
    num_coords = length(x)

    # Forces and acceleration initialization
    forces_nounits_t = ustrip_vec.(similar(sys.coords))   # kJ nm^-1 mol^-1
    forces_t = forces_nounits_t .* sys.force_units
    forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
    accels_nounits_t = forces_nounits_t ./ 1  # Identity mass matrix (division by 1 is a no-op)
    @assert are_units_same((uX / ut^2), uF / uM)

    # Noise initialization
    noise = similar(sys.velocities)
    noise = sample_noise_vector(num_coords)  # dimensionless
    # Note: / (1^(1/2)) is redundant as it equals 1, but kept for fidelity to original
    compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, 0, n_threads)
    # @assert are_units_same(uX, sqrt(uk * uT * ut / uγ) / sqrt(uM))  # Commented out as in original

    # Constrained dynamics initialization
    P_F, gradφ = compute_Pvec_x(φ=sim.φ_grid, vec=accels_nounits_t, x=x)
    Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
    @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
    laplacianφ = [0.0]
    gHg = [0.0]
    divP = compute_divP!(φ_flat=sim.φ_flat, gradφ=gradφ, laplacianφ=laplacianφ, gHg=gHg, x=x)   # nm^-1
    @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))

    # Return all necessary variables
    return (x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, laplacianφ, gHg, neighbors, forces_buffer, 
            masses_nounits, friction_nounits, k_bT, dt)
end


# Shared initialisation function
# @inline function init_simulation(sys, sim::CVConstrainedOverdampedLangevin, n_threads::Integer, run_loggers::Bool)    
#     # Warn if constraints are present
#     isempty(sys.constraints) || @warn "CVConstrainedOverdampedLangevin ignores constraints"

#     # Unit assertions
#     uX, ut, uT, uk, uγ, uM, uF = unit(sys.coords[1][1]), unit(sim.dt), unit(sim.T), unit(sys.k), unit(sim.γ), unit(masses(sys)[1]), sys.force_units
#     @assert uX == u"nm" && ut == u"ps" && uT == u"K" && uk == u"kJ/(mol*K)" && uγ == u"ps^-1" && uM == u"g/mol" && uF == u"kJ/(nm*mol)"

#     # Physical quantities
#     friction_nounits = ustrip(sim.γ)
#     k_bT = ustrip(sys.k) * ustrip(sim.T)
#     dt = ustrip(sim.dt)
#     params = SimParams(dt, k_bT, friction_nounits, n_threads, run_loggers)

#     # Coordinate initialisation
#     sys.coords .= wrap_coords.(sys.coords, (sys.boundary,))
#     !iszero(sim.remove_CM_motion) && remove_CM_motion!(sys)
#     neighbors = find_neighbors(sys, sys.neighbor_finder; n_threads=n_threads)
#     apply_loggers!(sys, neighbors, 0, run_loggers; n_threads=n_threads)
#     x = ustrip.(sys.coords)
#     num_coords = length(x)

#     # Forces and noise initialisation
#     forces_nounits_t = ustrip_vec.(similar(sys.coords))
#     forces_buffer = Molly.init_forces_buffer!(sys, forces_nounits_t, n_threads)
#     accels_nounits_t = forces_nounits_t  # Identity mass matrix
#     noise = sample_noise_vector(num_coords)
#     compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, 0, params)
    
#     # Constrained dynamics initialisation
#     P_F, gradφ = compute_Pvec_x(φ=sim.φ_grid, vec=accels_nounits_t, x=x)
#     Pnoise = compute_Pvec(gradφ=gradφ, vec=noise)
#     divP = compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x)

#     # Unit consistency checks
#     @assert are_units_same(uX / ut^2, uF / uM)
#     @assert are_units_same(uX, (uF / (uM * uγ)) * ut)
#     @assert are_units_same(uX, uk * uT * ut / (uM * uγ * uX))

#     return (x, forces_nounits_t, accels_nounits_t, noise, P_F, Pnoise, divP, gradφ, neighbors, forces_buffer, params)
# end

@inline function update_ergodic_integral!(compute_flag::Bool, quantity::Union{Function, Nothing}, args...)
    if !compute_flag
        return 0.0
    else
        return quantity(args...)
    end
end

# Core update functions
@inline function compute_forces_and_noise!(sys, forces_nounits_t, accels_nounits_t, noise, neighbors, forces_buffer, step_n, n_threads)
    forces_nounits_t .= Molly.forces_nounits!(forces_nounits_t, sys, neighbors, forces_buffer, step_n; n_threads=n_threads)
    accels_nounits_t .= forces_nounits_t  # Identity mass matrix
    sample_noise_vector!(noise)
end

# @inline function compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, sim, accels_nounits_t, noise, x)
#     compute_Pvec_x!(P_F, gradφ, φ=sim.φ_grid, vec=accels_nounits_t, x=x)
#     divP .= compute_divP(φ_flat=sim.φ_flat, gradφ=gradφ, x=x)
#     compute_Pvec!(Pnoise, gradφ=gradφ, vec=noise)
# end

@inline function compute_drift_and_diffusion_components!(P_F, Pnoise, divP, gradφ, laplacianφ, gHg, sim, accels_nounits_t, noise, x)
    compute_Pvec_x!(P_F, gradφ, φ=sim.φ_grid, vec=accels_nounits_t, x=x)
    divP .= compute_divP!(φ_flat=sim.φ_flat, gradφ=gradφ, laplacianφ=laplacianφ, gHg=gHg, x=x)
    compute_Pvec!(Pnoise, gradφ=gradφ, vec=noise)
end

@inline function finalise_step!(sys, x, neighbors, forces_nounits_t, step_n, run_loggers, n_threads)
    sys.coords .= wrap_coords.(x * u"nm", (sys.boundary,))
    neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
    apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, current_forces=forces_nounits_t * u"kJ/(nm*mol)")
    return neighbors
end

@inline function find_lambda(φ_grid, x_barₖ, divP, k_bT, friction_nounits, dt, target_CV; tol=1e-6, max_iter=50)
    correction = (k_bT * divP ./ friction_nounits) .* dt
    f(λ) = φ_grid(x_barₖ .+ correction .* λ) - target_CV
    
    λ = 0.0
    f_val = 0.0
    for iter in 1:max_iter
        f_val = f(λ)
        if abs(f_val) < tol 
            return λ, 1, abs(f_val), iter 
        end
        
        f_deriv = Zygote.gradient(f, λ)[1]
        if abs(f_deriv) < 1e-12 
            @warn "Derivative near zero"
            return λ, 2, abs(f_val), iter
        end
        
        λ_new = λ - f_val / f_deriv
        λ = λ_new
    end
    
    return λ, 3, abs(f_val), max_iter
end

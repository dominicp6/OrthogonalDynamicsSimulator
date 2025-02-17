export MT2

function MT2!(sys, sim, x0, F_x_barₖ₋₁, d, noise)
    x0 .= sys.coords + sim.dt * F_x_barₖ₋₁ / 4 
    gradφ .= ForwardDiff.gradient(sim.φ, x0)

    # Generate random variables χ for each dimension
    χ = rand([-1, 1], d)

    # Construct the Ja matrix (d x d)
    J = zeros(d, d)
    # TODO: consider multithreading this part of the code
    for a in 1:d
        for b in 1:d
            if a == b
                J[a, b] = dt * (noise[b]^2 - 1) / 2
            elseif a > b
                J[a, b] = dt * (noise[a] * noise[b] - χ[a]) / 2
            else
                J[a, b] = dt * (noise[a] * noise[b] + χ[b]) / 2
            end
        end
    end

    # Initialize result vector
    result = zeros(d)  

    # Computing the first term
    # TODO: consider multithreading this part of the code
    # TODO: assign arg1 and arg2 at beginning of loop for speed-up?
    for a in 1:d
        # Compute the arguments based on D_x0 * Ja
        PJa = compute_Pvec(gradφ, J[a, :])
        arg1 = x0 + sim.sigma * PJa
        arg2 = x0 - sim.sigma * PJa

        # Accumulate results for D_vec[a](...) corresponding to Da in the equation
        eI = zeros(eltype(x0), length(x))
        eI[a] = 1
        # TODO: Should I put a dot here?
        result += 0.5 * sigma * (compute_Pvec_x(arg1, eI) - compute_Pvec_x(arg2, eI))
    end

    # Computing the second term
    sqrt_dt = sqrt(sim.dt)
    sqrt_dt_half = sqrt(sim.dt / 2)

    Pχ = compute_Pvec(gradφ, χ)
    arg1 = x0 + sqrt_dt_half * sigma * Pχ
    arg2 = x0 - sqrt_dt_half * sigma * Pχ

    # The last term involves the entire diffusion tensor D, not individual components
    result += (sigma^2) * sqrt_dt / 2 * (compute_Pvec_x(arg1, noise) + compute_Pvec_x(arg2, noise))  

    return result
end


# function MT2(x0, sigma, dt, D, D_column, Rₖ)
#     # Dimension of the system
#     d = length(x0)

#     # Generate random variables χ for each dimension
#     χ = rand([-1, 1], d)

#     # Construct the Ja matrix (d x d)
#     J = zeros(d, d)
#     # TODO: consider multithreading this part of the code
#     for a in 1:d
#         for b in 1:d
#             if a == b
#                 J[a, b] = dt * (Rₖ[b]^2 - 1) / 2
#             elseif a > b
#                 J[a, b] = dt * (Rₖ[a] * Rₖ[b] - χ[a]) / 2
#             else
#                 J[a, b] = dt * (Rₖ[a] * Rₖ[b] + χ[b]) / 2
#             end
#         end
#     end

#     # Evaluate the diffusion tensor D at x0 (returns dxd matrix)
#     D_x0 = D(x0)  

#     # Initialize result vector
#     result = zeros(d)  

#     # Computing the first term
#     # TODO: consider multithreading this part of the code
#     for a in 1:d
#         # Compute the arguments based on D_x0 * Ja
#         arg1 = x0 + sigma * D_x0 * J[a, :]
#         arg2 = x0 - sigma * D_x0 * J[a, :]

#         # Accumulate results for D_vec[a](...) corresponding to Da in the equation
#         result += 0.5 * sigma * (D_column(arg1, a) - D_column(arg2, a))
#     end

#     # Computing the second term
#     sqrt_dt = sqrt(dt)
#     sqrt_dt_half = sqrt(dt / 2)

#     arg1 = x0 + sqrt_dt_half * sigma * D_x0 * χ
#     arg2 = x0 - sqrt_dt_half * sigma * D_x0 * χ

#     # The last term involves the entire diffusion tensor D, not individual components
#     result += (sigma^2) * sqrt_dt / 2 * (D(arg1) + D(arg2)) * Rₖ

#     return result
# end



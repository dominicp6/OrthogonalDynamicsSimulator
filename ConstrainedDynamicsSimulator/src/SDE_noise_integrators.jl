export MT2

function sample_chi_vector(d)
    return [@SVector rand([-1.0, 1.0], 3) for _ in 1:d]
end

# TODO: add typing here
function MT2!(coords, gradφ, dt, sigma, φ_grid, x0, F_x_barₖ₋₁, d, noise)
    # grid
    x0 .= coords + dt * F_x_barₖ₋₁ / 4 
    # grid
    gradφ .= clean_gradient(Zygote.gradient(φ_grid, x0)[1])

    # Generate random variables χ for each dimension
    # grid
    χ = sample_chi_vector(d)

    # Construct the Ja matrix (3d x 3d)
    # flat 
    J = zeros(3d, 3d)
    # TODO: consider multithreading this part of the code
    # TODO: or making more efficient to avoid quadratic cost!
    for a in 1:3d
        for b in 1:3d
            i_a, j_a = divrem(a - 1, 3) .+ 1  # Convert flat index to (row, col)
            i_b, j_b = divrem(b - 1, 3) .+ 1

            if a == b
                J[a, b] = dt * (noise[i_b][j_b]^2 - 1) / 2
            elseif a > b
                J[a, b] = dt * (noise[i_a][j_a] * noise[i_b][j_b] - χ[i_a][j_a]) / 2
            else
                J[a, b] = dt * (noise[i_a][j_a] * noise[i_b][j_b] + χ[i_b][j_b]) / 2
            end
        end
    end

    # Initialize result vector
    # grid
    result = similar(gradφ)

    # Computing the first term
    # TODO: consider multithreading this part of the code
    # grid
    PJa = similar(gradφ)
    arg1 = similar(gradφ)
    arg2 = similar(gradφ)
    eI = [zeros(3) for _ in 1:d]
    for a in 1:3d
        # Compute the arguments based on D_x0 * Ja
        compute_Pvec!(PJa, gradφ, unflatten(J[a, :]))
        arg1 .= x0 + sigma * PJa
        arg2 .= x0 - sigma * PJa

        # Accumulate results for D_vec[a](...) corresponding to Da in the equation
        i_a, j_a = divrem(a - 1, 3) .+ 1
        eI[i_a][j_a] = 1.0
        # TODO: Should I put a dot here?
        result += 0.5 * sigma * (compute_Pvec_x(φ_grid, arg1, eI) - compute_Pvec_x(φ_grid, arg2, eI))
        eI[i_a][j_a] = 0.0
    end

    # Computing the second term
    sqrt_dt = sqrt(dt)
    sqrt_dt_half = sqrt(dt / 2)

    Pχ = compute_Pvec(gradφ, χ)
    arg1 .= x0 + sqrt_dt_half * sigma * Pχ
    arg2 .= x0 - sqrt_dt_half * sigma * Pχ

    # The last term involves the entire diffusion tensor D, not individual components
    result += (sigma^2) * sqrt_dt / 2 * (compute_Pvec_x(φ_grid, arg1, noise) + compute_Pvec_x(φ_grid, arg2, noise))  

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



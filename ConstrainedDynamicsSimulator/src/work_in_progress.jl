        # println("iteration $step_n")
        # println("x_bar")
        # println(x_barₖ)
        # println("sys.coords")
        # println(sys.coords)

        # println("P_M_inv_F")
        # println(P_M_inv_F)
        # println("divP")
        # println(divP)

        # if !iszero(sim.remove_CM_motion) && step_n % sim.remove_CM_motion == 0
        #     remove_CM_motion!(sys)
        # end

        # if !(sim.φ_grid(x_barₖ) ≈ torsion_angle(sys.coords[5], sys.coords[7], sys.coords[9], sys.coords[15], sys.boundary))
        #     println("Assertion failed on iteration $step_n") 
        #     println("x_bar was:") 
        #     println(x_barₖ) 
        #     println("sys.coords was:")
        #     println(sys.coords)
        #     return sys
        # end 
        
        # Apply one step of the PVD2 integrator
        # gradφ .= clean_gradient(Zygote.gradient(sim.φ_grid, X)[1])

        # compute_PF!(PF, gradφ, sim.φ_grid, x_barₖ, scaled_F)
        # divP .= compute_divP(sim.φ_flat, gradφ, x_barₖ) # nm^-1
        # x_barₖ += (PF + k_bT * divP) * dt + sqrt(dt) * sigma * compute_Pvec(gradφ, noise) 
        # x_barₖ += accels_t * dt / ustrip(sim.γ) + sqrt(2 * k_bT * dt / ustrip(sim.γ)) * noise ./ ustrip.(masses(sys)).^(0.5) 
        


        # TODO: what is this line below even correct?
        # x_barₖ .= X + 0.5 * sqrt(dt) * sigma * compute_Pvec(gradφ, noise) # nm
        # compute_PgradV!(PgradV, gradφ, sim.φ_grid, x_barₖ, scaled_F)
        # divP .= compute_divP(sim.φ_flat, gradφ, x_barₖ) # nm^-1

        # F_x_barₖ .= - PgradV + k_bT * divP   # s^-1

        # TODO: needs fixing
        # X .+= dt * F_x_barₖ + MT2!(X, gradφ, dt, sigma, sim.φ_grid, x_barₖ, F_x_barₖ₋₁, num_coords, noise)  # nm
        
        # Log and update coordinates, ready for the next iteration
        # sys.coords .= wrap_coords.(x_barₖ * units_coords, (sys.boundary,))  
        # neighbors = find_neighbors(sys, sys.neighbor_finder, neighbors, step_n; n_threads=n_threads)
        # apply_loggers!(sys, neighbors, step_n, run_loggers; n_threads=n_threads, current_forces=forces_t * sys.force_units)

        # F_x_barₖ₋₁ .= F_x_barₖ  # s^-1





        # function compute_divP_efficient(;φ_flat::Function, x::AbstractVector{SVector{3,Float64}})
#     # Flatten the input
#     # TODO: collect needed here?
#     x_flat = flatten(x)  # Ensure a mutable Vector{Float64}

#     # Compute the gradient ∇φ
#     gradφ = ForwardDiff.gradient(φ_flat, x_flat)
#     norm_g_sq = dot(gradφ, gradφ)

#     # Compute H ∇φ = (1/2) ∇(||∇φ||²)
#     # Define the function ||∇φ||²
#     # TODO: consider more efficient implementation of this line
#     norm_grad_sq = x -> dot(ForwardDiff.gradient(φ_flat, x), ForwardDiff.gradient(φ_flat, x))
#     grad_norm_g_sq = ForwardDiff.gradient(norm_grad_sq, x_flat)
#     H_g = 0.5 * grad_norm_g_sq

#     # Compute the Laplacian ∇²φ = ∑ ∂²φ/∂x_i²
#     laplacian = 0.0
#     # TODO: is there a better way to compute second derivatives in Julia?
#     n = length(x_flat)
#     for i in 1:n
#         # Define φ along the i-th direction
#         e_i = zeros(n); e_i[i] = 1.0
#         φ_i = t -> φ_flat(x_flat + t * e_i)
#         # First derivative dφ/dt
#         dφ_i_dt = t -> ForwardDiff.derivative(φ_i, t)
#         # Second derivative d²φ/dt² at t=0
#         d2φ_i_dt2 = ForwardDiff.derivative(dφ_i_dt, 0.0)
#         laplacian += d2φ_i_dt2
#     end

#     # Compute ∇⋅P components
#     term1 = - (laplacian / norm_g_sq) * gradφ
#     term2 = (2 * dot(gradφ, H_g) / (norm_g_sq^2)) * gradφ
#     term3 = - (1 / norm_g_sq) * H_g
#     div_P_flat = term1 + term2 + term3

#     # Reshape back to Vector{SVector{3,Float64}}
#     return unflatten(div_P_flat)
# end



export MT2, MT2_efficient!

using .Threads

function sample_chi_vector(d)
    return [@SVector rand([-1.0, 1.0], 3) for _ in 1:d]
end

@inline function MT2_efficient!(x0, dt, sqrt_dt, φ_flat, noise, x0_flat, noise_flat, d, gradφ, prefactor1, prefactor2, prefactor3, prefactor4, χ, result, result_final, PJa, arg1, arg2, Pxi1, Pxi2, ei, J_row, Pχ, proj1, proj2)
    # Flatten inputs
    x0_flat .= flatten(x0)
    noise_flat .= flatten(noise)

    # Initial gradient and random variables
    gradφ .= clean_gradient(Zygote.gradient(φ_flat, x0_flat)[1])
    χ .= rand([-1.0, 1.0], d)

    # First term: Loop over dimensions
    @threads for a in 1:d
        # Compute J_row on-the-fly
        # TODO: check this calculation
        J_row .= (noise_flat[a] * noise_flat) / 2
        if a > 1
            J_row[1:(a-1)] .-= χ[a] / 2
        end
        if a < d
            J_row[(a+1):end] .+= χ[(a+1):end] / 2
        end
        J_row[a] -= 0.5

        # Compute projection PJa
        compute_Pvec!(PJa, gradφ=gradφ, vec=J_row)

        # Compute arguments
        arg1 .= x0_flat .+ dt * prefactor2 .* PJa
        arg2 .= x0_flat .- dt * prefactor2 .* PJa

        # Set basis vector
        ei[a] = 1.0

        # Compute projection difference and accumulate
        compute_Px_i!(Pxi1, gradφ, φ=φ_flat, ei=ei, x=arg1, i=a)[1]
        compute_Px_i!(Pxi2, gradφ, φ=φ_flat, ei=ei, x=arg2, i=a)[1]
        result .+= prefactor1 .* (Pxi1 .- Pxi2)

        # Reset basis vector
        ei[a] = 0.0
    end

    # Second term
    Pχ .= compute_Pvec(gradφ=gradφ, vec=χ)
    arg1 .= x0_flat .+ sqrt_dt .* prefactor4 .* Pχ
    arg2 .= x0_flat .- sqrt_dt .* prefactor4 .* Pχ
    proj1 .= compute_Pvec_x(φ=φ_flat, vec=noise_flat, x=arg1)[1]
    proj2 .= compute_Pvec_x(φ=φ_flat, vec=noise_flat, x=arg2)[1]
    result .+= sqrt_dt .* prefactor3 .* (proj1 .+ proj2)
    result_final .= result
    result .= zeros(d)

    return unflatten(result_final)
end

@inline function MT2(x0, dt, k_bT, gamma, φ_flat, noise)
    x0 = flatten(x0)
    noise = flatten(noise)
    d = length(noise)

    # using m = 1 gmol^-1
    prefactor1 = sqrt(k_bT / (2 * gamma)) * (1)^(-(1/2)) / (gamma)^(1/2)
    prefactor2 = sqrt(2 * k_bT) * (1)^(-(1/2))
    prefactor3 = sqrt(k_bT / (2 * gamma)) * (1)^(-(1/2)) 
    prefactor4 = sqrt(k_bT / gamma) * (1)^(-(1/2))

    gradφ = clean_gradient(Zygote.gradient(φ_flat, x0)[1])

    # Generate random variables χ for each dimension
    χ = rand([-1, 1], d)

    # Construct the J matrix
    J = zeros(d, d)
    # TODO: consider efficiency
    for a in 1:d
        for b in 1:d
            if a == b
                J[a, b] = (noise[b]^2 - 1) / 2
            elseif a > b
                J[a, b] = (noise[a] * noise[b] - χ[a]) / 2
            else
                J[a, b] = (noise[a] * noise[b] + χ[b]) / 2
            end
        end
    end

    # Initialize result vector
    result = zeros(size(x0))

    # Computing the first term
    PJa = similar(x0)
    arg1 = similar(x0)
    arg2 = similar(x0)
    eI = [0.0 for _ in 1:d]
    # TODO: consider efficiency
    for a in 1:d
        compute_Pvec!(PJa, gradφ=gradφ, vec=J[a, :])
        arg1 .= x0 .+ dt * prefactor2 .* PJa
        arg2 .= x0 .- dt * prefactor2 .* PJa

        eI[a] = 1.0
        result .+= prefactor1 .* (compute_Pvec_x(φ=φ_flat, vec=eI, x=arg1)[1] - compute_Pvec_x(φ=φ_flat, vec=eI, x=arg2)[1])
        eI[a] = 0.0
    end

    # Computing the second term
    sqrt_dt = sqrt(dt)

    Pχ = compute_Pvec(gradφ=gradφ, vec=χ)
    arg1 .= x0 .+ sqrt_dt .* prefactor4 .* Pχ
    arg2 .= x0 .- sqrt_dt .* prefactor4 .* Pχ

    # The last term involves the entire diffusion tensor, not individual components
    result += sqrt_dt .* prefactor3 .* (compute_Pvec_x(φ=φ_flat, vec=noise, x=arg1)[1] + compute_Pvec_x(φ=φ_flat, vec=noise, x=arg2)[1])  

    return unflatten(result)
end



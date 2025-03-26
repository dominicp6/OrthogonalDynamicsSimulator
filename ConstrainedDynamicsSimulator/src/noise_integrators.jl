export MT2

function sample_chi_vector(d)
    return [@SVector rand([-1.0, 1.0], 3) for _ in 1:d]
end

# TODO: add typing here
function MT2(x0, dt, k_bT, gamma, φ_flat, noise)
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



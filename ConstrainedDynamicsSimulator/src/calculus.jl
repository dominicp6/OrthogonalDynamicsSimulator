export clean_gradient, compute_divP, compute_divP_efficient, compute_Pvec, compute_Pvec_x, compute_Pvec!, compute_Px_i!

flatten(x::Vector{SVector{3,Float64}}) = reinterpret(Float64, x)
unflatten(x::Vector{Float64}) = reinterpret(SVector{3,Float64}, x)

function clean_gradient(gradient)
    return map(g -> something(g, SVector{3, Float64}(0.0, 0.0, 0.0)), gradient)
end

function compute_Pvec(;gradφ::AbstractVector, vec::AbstractVector) :: AbstractVector
    return vec .- gradφ .* dot(gradφ, vec) ./ dot(gradφ, gradφ)
end

function compute_Pvec!(Pvec::AbstractVector; gradφ::AbstractVector, vec::AbstractVector)
    Pvec .= vec .- gradφ .* dot(gradφ, vec) ./ dot(gradφ, gradφ)
end

function compute_Pvec_x(;φ::Function, vec::AbstractVector, x::AbstractVector) :: Tuple{AbstractVector, AbstractVector}
    # phi should be either phi_grid or phi_flat to match the shape of the vectors
    gradφ = clean_gradient(Zygote.gradient(φ, x)[1])
    return vec .- (gradφ .* dot(gradφ, vec)) ./ dot(gradφ, gradφ), gradφ
end

function compute_Pvec_x!(Pvec::AbstractVector, gradφ::AbstractVector; φ::Function, vec::AbstractVector, x::AbstractVector)
    # phi should be either phi_grid or phi_flat to match the shape of the vectors
    gradφ .= clean_gradient(Zygote.gradient(φ, x)[1])
    Pvec .= vec .- gradφ .* dot(gradφ, vec) ./ dot(gradφ, gradφ)
end

function compute_Px_i!(Pxi::AbstractVector, gradφ::AbstractVector; φ::Function, ei::AbstractVector, x::AbstractVector, i::Integer)
    # phi should be either phi_grid or phi_flat to match the shape of the vectors
    gradφ .= clean_gradient(Zygote.gradient(φ, x)[1])
    Pxi .= ei .- gradφ .* gradφ[i] ./ dot(gradφ, gradφ)
end

function compute_divP(;φ_flat::Function, gradφ::AbstractVector{<:Union{SVector{3, Float64}, Float64}}, x::AbstractVector{<:Union{SVector{3, Float64}, Float64}}) :: AbstractVector{<:Union{SVector{3, Float64}, Float64}}
    x_flat = flatten(x)             # Flatten x to 3N vector
    g = flatten(gradφ)              # ∇φ as 3N vector
    H = Zygote.hessian(φ_flat, x_flat)  # 3N × 3N Hessian
    laplacian = sum(diag(H))        # ∇²φ = tr(H)
    Hg = H * g                      # H ∇φ
    gHg = dot(g, Hg)                # ∇φ^T H ∇φ
    norm_g_sq = dot(g, g)           # ||∇φ||²
    div_P = - (laplacian / norm_g_sq) * g + (2 * gHg / (norm_g_sq^2)) * g - (1 / norm_g_sq) * Hg                  # Number of particles
    return unflatten(div_P)  # Unflatten to N 3-vectors
end
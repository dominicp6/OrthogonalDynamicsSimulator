export compute_PgradV, compute_divP, compute_Pvec, compute_Pvec_x

# Compute P(x) ∇V(x) = ∇V(x) - (∇φ(x)) (∇φ(x)ᵀ ∇V(x))
function compute_PgradV(φ, x::AbstractVector, gradV::AbstractVector)
    gradφ = ForwardDiff.gradient(φ, x)
    return gradV - gradφ * (dot(gradφ, gradV)), gradφ
end

# Compute div P(x) where [divP(x)]ₖ = (∂ₖφ) tr(H(φ)) + [H(φ)*∇φ(x)]ₖ.
# For high-dimensional problems, one would compute H*∇φ via a Hessian–vector product.
function compute_divP(φ, gradφ::AbstractVector, x::AbstractVector)
    H = ForwardDiff.hessian(φ, x)
    laplacian = sum(diag(H))  # tr(H)
    return gradφ * laplacian + H * gradφ
end

function compute_Pvec(gradφ::AbstractVector, vec::AbstractVector)
    return vec - gradφ * dot(gradφ, vec)
end

function compute_Pvec_x(x::AbstractVector, vec::AbstractVector)
    gradφ = ForwardDiff.gradient(φ, x)
    return vec - gradφ * dot(gradφ, vec)
end

# Compute the Hessian-vector product H(φ(x)) * v without forming the full Hessian.
# function hvp(φ, x::AbstractVector, v::AbstractVector)
#     # Promote each xᵢ to a Dual with seed vᵢ
#     x_dual = ForwardDiff.Dual.(x, v)
#     # Compute the gradient; each component is a Dual number whose derivative is the directional derivative
#     grad_dual = ForwardDiff.gradient(φ, x_dual)
#     # Extract the derivative part (a one-element tuple) from each Dual
#     return [first(ForwardDiff.partials(g)) for g in grad_dual]
# end

# # Compute div P(x)_k = (∂ₖ φ) (tr H(φ)) + [H(φ)∇φ]_k,
# # where we compute H(φ)∇φ using our Hessian-vector product.
# function compute_divP(φ, x::AbstractVector)
#     gradφ = ForwardDiff.gradient(φ, x)
#     # Laplacian: for moderate n, use full Hessian; for high-dim, consider alternative strategies.
#     H = ForwardDiff.hessian(φ, x)
#     laplacian = sum(diag(H))
#     # Compute Hessian-vector product H(φ(x))*gradφ(x) efficiently:
#     H_gradφ = hvp(φ, x, gradφ)
#     return gradφ * laplacian + H_gradφ
# end
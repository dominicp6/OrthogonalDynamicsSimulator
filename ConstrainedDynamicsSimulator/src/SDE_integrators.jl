include("SDE_noise_integrators.jl")
using LinearAlgebra
export EM, LMd, SS2, PVD2

function PVD2(x0::Vector{Float32}, Vprime::Function, D::Function, div_DDT::Function, D_column::Function, sigma::Float32, m::Integer, dt::Float32)::Matrix{Float32}
    
    # preliminary functions
    d::Int = length(x0)
    D² = (x) -> D(x)^2
    F = (x) -> D²(x) * (-Vprime(x)) + sigma^2 * div_DDT(x)/2

    # set up
    t::Float32 = 0.0
    x::Vector{Float32} = copy(x0)
    x_traj::Matrix{Float32} = zeros(d, m)
    sqrt_dt::Float32 = sqrt(dt)

    # simulate
    x_barₖ₋₁ = x 
    F_x_barₖ₋₁ = F(x_barₖ₋₁)
    for i in 1:m
        Rₖ = randn(d)
        # choosing x_tilde = x
        x_barₖ = x + 0.5 * sqrt_dt * sigma * D(x) * Rₖ
        F_x_barₖ = F(x_barₖ)
        x += dt * F_x_barₖ + MT2(x + dt * F_x_barₖ₋₁ / 4, sigma, dt, D, D_column, Rₖ)

        x_traj[:, i] .= x_barₖ

        # update the time
        t += dt

        x_barₖ₋₁ = copy(x_barₖ)
        F_x_barₖ₋₁ = copy(F_x_barₖ)
    end

    return x_traj    
end

function EM(x0::Vector{Float32}, Vprime::Function, D::Function, div_DDT::Function, D_column::Function, sigma::Float32, m::Integer, dt::Float32)::Matrix{Float32}
    # set up
    d::Int = length(x0)
    t::Float32 = 0.0
    x::Vector{Float32} = copy(x0)
    x_traj::Matrix{Float32} = zeros(d, m)
    sqrt_dt::Float32 = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        D_x = D(x)
        grad_V = Vprime(x)
        DDT_x = D_x * D_x'
        div_DDT_x = div_DDT(x)
        drift = -DDT_x * grad_V + (sigma^2) * div_DDT_x / 2 
        diffusion = sigma * D_x * randn(d)
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[:,i] .= x
        
        # update the time
        t += dt
    end
    
    return x_traj
end

function LMd(x0::Vector{Float32}, Vprime::Function, D::Function, div_DDT::Function, D_column::Function, sigma::Float32, m::Integer, dt::Float32)::Matrix{Float32}
    # Leimkuhler-Matthews method with drift correction (a.k.a. Hummer-Leimkuhler-Matthews)

    # set up
    d::Int = length(x0)
    t::Float32 = 0.0
    x::Vector{Float32} = copy(x0)
    x_traj::Matrix{Float32} = zeros(d, m)
    Rₖ::Vector{Float32} = randn(d)
    sqrt_dt::Float32 = sqrt(dt)

    # simulate
    for i in 1:m
        # compute the drift and diffusion coefficients
        D_x = D(x)
        grad_V = Vprime(x)
        DDT_x = D_x * D_x'
        div_DDT_x = div_DDT(x)
        drift = - DDT_x * grad_V + (3/4) * sigma^2 * div_DDT_x / 2
        Rₖ₊₁ = randn(d)
        diffusion = sigma * D_x * (Rₖ + Rₖ₊₁)/2 
        
        # update the configuration
        x += drift * dt + diffusion * sqrt_dt
        x_traj[:,i] .= x
        
        # update the time
        t += dt

        # update the noise increment
        Rₖ = copy(Rₖ₊₁)      
    end 
    
    return x_traj
end

# TODO: make a version of this that only uses a second-order drift integrator rather than this RK4 version
function SS2(x0::Vector{Float32}, Vprime::Function, D::Function, div_DDT::Function, D_column::Function, sigma::Float32, m::Integer, dt::Float32)::Matrix{Float32}
    # Second-order strang splitting method using an RK4 integrator for the drift terms

    # set up
    d::Int = length(x0)
    t::Float32 = 0.0
    x::Vector{Float32} = copy(x0)
    x_traj::Matrix{Float32} = zeros(d, m)

    # simulate
    for i in 1:m
        drift_term = x -> -(D(x)*D(x)') * Vprime(x) + sigma^2 * div_DDT(x) / 2

        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * drift_term(x)
        k2 = (dt / 2) * drift_term(x + 0.5 * k1)
        k3 = (dt / 2) * drift_term(x + 0.5 * k2)
        k4 = (dt / 2) * drift_term(x + k3) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        Rₖ = randn(d)

        x += MT2(x, sigma, dt, D, D_column, Rₖ)

        # Perform 1 step of RK4 integration for hat_xₖ₊₁
        k1 = (dt / 2) * drift_term(x)
        k2 = (dt / 2) * drift_term(x + 0.5 * k1)
        k3 = (dt / 2) * drift_term(x + 0.5 * k2)
        k4 = (dt / 2) * drift_term(x + k3) 
            
        # Update state using weighted average of intermediate values
        x += (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 

        x_traj[:,i] = x

        # update the time
        t += dt
    end

    return x_traj
end
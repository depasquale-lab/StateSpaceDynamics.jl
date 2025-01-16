using StateSpaceDynamics, Plots, Infiltrator

const SSD = StateSpaceDynamics

model = initialize_slds()

T = 1000
# Generate synthetic data
x, y, z = sample(model, T)


using LinearAlgebra

p = model.B[1].obs_dim
d = model.B[1].latent_dim

K = model.K
A = rand(K, K)
A ./= sum(A, dims=2) # Normalize rows to sum to 1

πₖ = rand(K)
πₖ ./= sum(πₖ) # Normalize to sum to 1

Q = Matrix(0.5 * I(d))

x0 = [0.0; 0.0]
P0 = Matrix(0.5 * I(d))

# set up the observation parameters
C = randn(p, d)
R = Matrix(0.001 * I(p))

B = [StateSpaceDynamics.LinearDynamicalSystem(
    StateSpaceDynamics.GaussianStateModel(0.95 * [cos(f) -sin(f); sin(f) cos(f)], Q, x0, P0),
    StateSpaceDynamics.GaussianObservationModel(C, R),
    d, p, fill(true, 6  )) for (i,f) in zip(1:K, [0.5, 0.5])]

modeli = SwitchingLinearDynamicalSystem(A, B, πₖ, model.K)




function mstep!(slds::SSD.SwitchingLinearDynamicalSystem,
    FS::Vector{SSD.FilterSmooth{T}}, y::Matrix{T}, FB::SSD.ForwardBackward) where {T<:Real}

    K = slds.K

    #update initial state distribution
    SSD.update_initial_state_distribution!(slds, FB)
    #update transition matrix
    SSD.update_transition_matrix!(slds, FB)

    @infiltrate

    γ = FB.γ
    hs = exp.(γ)

    # get initial parameters
    old_params = vec([SSD.stateparams(slds.B[k]) for k in 1:K])
    old_params = [old_params; vec([SSD.obsparams(slds.B[k]) for k in 1:K])]

    for k in 1:K
        # Update LDS parameters
        SSD.update_initial_state_mean!(slds.B[k], FS[k].E_z)
        SSD.update_initial_state_covariance!(slds.B[k], FS[k].E_z, FS[k].E_zz)
        SSD.update_A!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        SSD.update_Q!(slds.B[k], FS[k].E_zz, FS[k].E_zz_prev)
        SSD.update_C!(slds.B[k], FS[k].E_z, FS[k].E_zz, reshape(y, size(y)...,1), vec(hs[k,:]))
        #update_R!(slds.B[k], FS[k].E_z, FS[k].E_zz, reshape(y, size(y)...,1), vec(hs[k,:]))
    end

    new_params = vec([SSD.stateparams(slds.B[k]) for k in 1:K])
    new_params = [new_params; vec([SSD.obsparams(slds.B[k]) for k in 1:K])]

    # calculate norm of parameter changes
    norm_change = norm(new_params - old_params)
    return norm_change
end


function variational_qs!(model::Vector{SSD.GaussianObservationModel{T}}, FB::SSD.ForwardBackward, 
    y, FS::Vector{SSD.FilterSmooth{T}}) where {T<:Real}
      log_likelihoods = FB.loglikelihoods
      K = length(model)
      T_steps = size(y, 2)
  
      for k in 1:K
  
          R_chol = SSD.cholesky(Symmetric(model[k].R))
          C = model[k].C
          C_Rinv = (R_chol \ C)'
  
          for t in 1:T_steps
              yt_Rinv = (R_chol \ y[:,t])'
              log_likelihoods[k, t] = -0.5 * yt_Rinv * y[:,t] + 
                  yt_Rinv * C * FS[k].E_z[:,t,1] - 0.5 * tr(C_Rinv * C * FS[k].E_zz[:,:,t,1])
          end
          println("In variational qs")
          
          
          # Subtract max for numerical stability
          log_likelihoods[k, :] .-= maximum(log_likelihoods[k, :])
  
      end 
  
      # Convert to likelihoods, normalize, and back to log space
      likelihoods = exp.(log_likelihoods)
      normalized_probs = likelihoods ./ sum(likelihoods, dims=1)
      FB.loglikelihoods = log.(normalized_probs)
  end

function variational_expectation!(model::SwitchingLinearDynamicalSystem, y, FB::SSD.ForwardBackward, FS::Vector{SSD.FilterSmooth{T}}) where {T<:Real}
  
    tol = 1e-6
    # Get starting point for iterative E-step
    γ = FB.γ
    hs = exp.(γ)
    ml_total = 0.
  
    # Initialize to something higher than the tolerance
    ml_diff = 1
    ml_prev = -Inf
    ml_storage = []

    while ml_diff > tol
        # Reselt likelihood calculation if tolerance not reached
        ml_total = 0
        #1. compute qs from xs, which will live as log_likelihoods in FB
        variational_qs!([model.obs_model for model in model.B], FB, y, FS)

        #2. compute hs from qs, which will live as γ in FB
        SSD.forward!(model, FB)
        SSD.backward!(model, FB)
        SSD.calculate_γ!(model, FB)
        SSD.calculate_ξ!(model, FB)
        hs = exp.(FB.γ)

        #ml_total += hmm_elbo(model, FB)
        ml_total = SSD.logsumexp(FB.α[:, end])

        for k in 1:model.K
            #3. compute xs from hs
            FS[k].x_smooth, FS[k].p_smooth, inverse_offdiag, total_entropy  = SSD.smooth(model.B[k], y, vec(hs[k,:]))
            FS[k].E_z, FS[k].E_zz, FS[k].E_zz_prev = 
                SSD.sufficient_statistics(reshape(FS[k].x_smooth, size(FS[k].x_smooth)..., 1), 
                reshape(FS[k].p_smooth, size(FS[k].p_smooth)..., 1), 
                reshape(inverse_offdiag, size(inverse_offdiag)..., 1))
            # calculate elbo
            ml_total += SSD.calculate_elbo(model.B[k], FS[k].E_z, FS[k].E_zz, FS[k].E_zz_prev, 
            reshape(FS[k].p_smooth, size(FS[k].p_smooth)..., 1), reshape(y, size(y)...,1), total_entropy)
        end

        # Set ml_total to the next iterations previous ml
        ml_diff = ml_prev - ml_total
        print(ml_total)
        ml_prev = ml_total
    end  # while loop
  
    return ml_total
  
  end  # function

function fiter!(
    slds::SSD.SwitchingLinearDynamicalSystem, y::Matrix{T}; 
    max_iter::Int=1000, tol::Real=1e-12) where {T<:Real}

    # Initialize log-likelihood
    prev_ml = -T(Inf)

    # Create a vector to store the log-likelihood values
    mls = Vector{T}()
    param_diff = Vector{T}()

    SSD.sizehint!(mls, max_iter)  # Pre-allocate for efficiency

    K = slds.K
    T_step = size(y, 2)
    FB = SSD.initialize_forward_backward(slds, T_step)
    #γ = FB.γ
    #γ .= -100.
    FS = [SSD.initialize_FilterSmooth(slds.B[k], T_step) for k in 1:K]

    @infiltrate

    # Run EM
    for i in 1:max_iter
        # E-step
        ml = variational_expectation!(slds, y, FB, FS)       

        # M-step
        Δparams = mstep!(slds, FS, y, FB)
        @infiltrate

        # Update the log-likelihood vector and parameter difference
        push!(mls, ml)
        push!(param_diff, Δparams)

        # Check convergence
        if abs(ml - prev_ml) < tol
            return mls, param_diff
        end

        prev_ml = ml
    end

    return mls, param_diff, FB, FS
end


  FB = StateSpaceDynamics.initialize_forward_backward(modeli, T)
  FS = [StateSpaceDynamics.initialize_FilterSmooth(modeli.B[k], T) for k in 1:K]
  
  mls, param_diff, FB, FS = fiter!(modeli, y; max_iter=1)

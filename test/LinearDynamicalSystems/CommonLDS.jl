function setup_lds(
    lds_type::Type,
    obs_dim::Int,
    latent_dim::Int,
    ntrials::Int=1,
    fit_bool::Vector{Bool}=[true, true, true, true, true, true],
)
    # Consistent parameters across model types
    A = [cos(0.1) -sin(0.1); sin(0.1) cos(0.1)]
    Q = 0.1 * Matrix{Float64}(I(latent_dim))
    x0 = randn(latent_dim)
    P0 = Matrix{Float64}(I(latent_dim))
    if lds_type == GaussianLDS
        C = Matrix{Float64}(I(obs_dim))
        R = 0.1 * Matrix{Float64}(I(obs_dim))
        lds = GaussianLDS(;
            A=A,
            C=C,
            Q=Q,
            R=R,
            x0=x0,
            P0=P0,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            fit_bool=fit_bool,
        )
    elseif lds_type == PoissonLDS
        C = abs.(randn(obs_dim, latent_dim))
        log_d = log.(abs.(randn(obs_dim)))
        lds = PoissonLDS(;
            A=A,
            C=C,
            Q=Q,
            log_d=log_d,
            x0=x0,
            P0=P0,
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            fit_bool=fit_bool,
        )
    else
        error("Unsupported LDS type")
    end

    T = 100
    x, y = StateSpaceDynamics.sample(lds, T, ntrials)
    return lds, x, y
end

function test_lds_properties(lds, obs_dim, latent_dim)
    @test isa(lds.state_model, StateSpaceDynamics.GaussianStateModel)
    @test isa(lds, StateSpaceDynamics.LinearDynamicalSystem)
    @test size(lds.state_model.A) == (latent_dim, latent_dim)
    @test size(lds.state_model.Q) == (latent_dim, latent_dim)
    @test size(lds.state_model.x0) == (latent_dim,)
    @test size(lds.state_model.P0) == (latent_dim, latent_dim)
    @test size(lds.obs_model.C) == (obs_dim, latent_dim)

    if isa(lds, GaussianLDS)
        @test isa(lds.obs_model, StateSpaceDynamics.GaussianObservationModel)
        @test size(lds.obs_model.R) == (obs_dim, obs_dim)
    elseif isa(lds, PoissonLDS)
        @test isa(lds.obs_model, StateSpaceDynamics.PoissonObservationModel)
        @test size(lds.obs_model.log_d) == (obs_dim,)
    end
end

function test_gradient(lds, x, y)
    for i in axes(y, 1)
        f = latents -> StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x[:, :, i])
        @test isapprox(grad_numerical, grad_analytical, rtol=1e-5, atol=1e-5)
    end
end

function test_hessian(lds, x, y)
    for i in axes(y, 1)
        hess, main, super, sub = StateSpaceDynamics.Hessian(lds, y[i, 1:3, :])
        @test size(hess) == (3 * lds.latent_dim, 3 * lds.latent_dim)
        @test size(main) == (3,)
        @test size(super) == (2,)
        @test size(sub) == (2,)

        function log_likelihood(x::AbstractArray)
            return StateSpaceDynamics.loglikelihood(permutedims(x), lds, y[i, 1:3, :])
        end
        hess_numerical = ForwardDiff.hessian(log_likelihood, permutedims(x[i, 1:3, :]))
        @test isapprox(hess_numerical, hess, rtol=1e-5, atol=1e-5)
    end
end

function test_smooth(lds, x, y)
    x_smooth, p_smooth, inverseoffdiag = StateSpaceDynamics.smooth(lds, y)

    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, size(y, 2))
    @test size(inverseoffdiag) == (lds.latent_dim, lds.latent_dim, size(y, 2))

    for i in axes(y, 3)
        f = latents -> StateSpaceDynamics.loglikelihood(latents, lds, y[:, :, i])
        grad_numerical = ForwardDiff.gradient(f, x_smooth[:, :, i])
        grad_analytical = StateSpaceDynamics.Gradient(lds, y[:, :, i], x_smooth[:, :, i])
        @test isapprox(grad_numerical, grad_analytical, rtol=1e-5, atol=1e-5)
        @test norm(grad_analytical) < 1e-4
    end
end

function test_estep(lds, x, y)
    E_z, E_zz, E_zz_prev, x_smooth, p_smooth, ml_total = StateSpaceDynamics.estep(lds, y)

    @test size(E_z) == (lds.latent_dim, size(y, 2))
    @test size(E_zz) == (lds.latent_dim, lds.latent_dim, size(y, 2), size(y, 3))
    @test size(E_zz_prev) == (lds.latent_dim, lds.latent_dim, size(y, 2), size(y, 3))
    @test size(x_smooth) == size(x)
    @test size(p_smooth) == (lds.latent_dim, lds.latent_dim, size(y, 2), size(y, 3))
    @test isa(ml_total, Float64)
end

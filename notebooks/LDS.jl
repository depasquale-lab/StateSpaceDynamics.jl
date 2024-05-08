### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 53c96240-c044-11ee-2f83-e93206e06ac6
begin
	using Pkg
	Pkg.activate("/Users/ryansenne/Documents/Github/SSM")
	include("/Users/ryansenne/Documents/Github/SSM//src/SSM.jl")
	using .SSM
	using LinearAlgebra
	using Distributions
	using Random
	using Plots
end

# ╔═╡ c3f10cbe-2505-47d3-a3de-340a8b7d63de
md"""## Linear Dynamical System""" 

# ╔═╡ 2b64b709-ea92-49e8-8b47-218b4ecc56c7
md"""The Linear Dynamical System is one of the most fundamental models in machine learning and statistics for sequential data analysis. Furthermore, along with the Hidden Markov Model, the LDS with Gaussian observations is of the simplest state-space models. Like the HMM, the LDS has the following graphical model structure:

We can summarize this through a state-evolution and observation equation. We assume the following probabilitistc structure:
"""

# ╔═╡ 35cbfd54-05bb-4c22-9f9b-ae76995fd1c8
md"""### Example: A noisy spring."""

# ╔═╡ 2e5a7b3c-cebd-459c-8315-4be0d4f85596
md"""In this example we will track a spring. We assume the spring is subject to white noise fluctuations in the state evolution, and that our sensors are themselves subject to white noise error. Assume we can observe the (noisy) position of the spring, but we do not observe the velocity. The dynamics of this model can be described as follows:

$p_{t+1} = p_t + v_t + q_t$
$v_{t+1} = (1-\alpha)v_t + \beta x_t + q_t$
$q_t \sim N(0, Q)$

where $\alpha$ and $\beta$ are the damping coefficent and spring constant, respectively. We can rewrite the above model in a matrix formlation for simplicity:

$\begin{bmatrix} p_{t+1} \\ v_{t+1} \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ -\beta & (1 - \alpha)\end{bmatrix}\begin{bmatrix} p_t \\ v_t\end{bmatrix} + \begin{bmatrix} q_{t,1} \\ q_{t,2}\end{bmatrix}$ 

Since we are assuming that position is the only observed variable, the observation model can be written as:

$y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} p_t \\ v_t \end{bmatrix}+ r_t$
$r_t \sim N(0, R)$
"""

# ╔═╡ 21dc64ac-c929-4eea-9568-8fbaef3e2e44
begin
	# function to simulate data from the above model
	function simulate_LDS(T::Int, α::Float64, β::Float64, x₀::Vector{Float64}, Q::Matrix{Float64}, R::Float64)
		# pre-allocate array
		state = zeros(T, 2) # pos/velocity
		observation = zeros(T, 1) # observations
		observation[1] = x₀[1]
		# set up parameters
		A = [1 1; -β (1-α)]
		state[1, :] = x₀
		process_noise = MvNormal(zeros(2), Q)
		observation_noise = Normal(0, R)
		# simulate
		for t in 2:T
			state[t, :] = A * state[t-1, :] .+ rand(process_noise)
			observation[t, :] = state[t] .+ rand(observation_noise, 1)
		end
		return state, observation
	end

# simulate 1000 steps of this model
state_sequence, observation_sequence = simulate_LDS(100, 0.9, 0.75, [1., -1], [0.01 0; 0 0.01], 1.)
	
end

# ╔═╡ dd275f99-1d29-4dbd-a804-7e64893074b0
begin
	# plot our latent states and what we actually observed
	l = @layout [a; b; c]
	p1 = plot(state_sequence[:, 2], linewidth=:3, labels="Velocity")
	p2 = plot(state_sequence[:, 1], linewidth=:3, labels="Position")
	p3 = plot(observation_sequence, linewidth=:3, labels="Noisy Position")
	plot(p1, p2, p3, layout=l)
end

# ╔═╡ 6fa55d13-f126-40f1-8b34-4221a70d36b3
md"""From what we can see, the position follows the general trends of the true state evolution. But, we can certainly do better. Furthermore, we have no understanding of the velocity of the spring moment-to-moment (we only have it because we simulated it)! We can first try the Kalman Filter to see if we can a.) get a better estimate of the position and b.) estimate the springs velocity."""

# ╔═╡ 7fb9f5a9-6cf6-4f09-8932-f7daefd937b2
md"""### The Kalman Filter"""

# ╔═╡ 885348ce-e907-47fd-a625-d9d1773d7025
md"""At the risk of being incredibly pedantic, we should make a quick distinction: the model describing the dynamics of our spring is a *Linear Dynamical System* (LDS), **not** a Kalman Filter. While these terms are often used interchangeably, they represent different concepts. The LDS model characterizes the underlying behavior of the spring's dynamics. On the other hand, the Kalman Filter is a specific Bayesian filtering algorithm used for state estimation within LDS models. In essence, the Kalman Filter functions as the Bayesian *filtering* mechanism for the Linear Dynamical System. 
\
\
Now that we have made that point, we can describe what a Kalman Filter actually does in probabilistic terms. In general, a probabilistic state-space model describes two sets of observations, the latent or "hidden" state, and the observations. The two processes are distributed as follows:

$x_t \sim p(x_t|x_{t-1})$
$y_t \sim p(y_t|x_t)$

where $x$ is the hidden state and $y$ is the observations. Now of course, while this model feels intuitively very simple, we are faced with the challenge posed to us by the hidden states, as their name suggest, they are hidden from us! Thus we are faced with computing the marginal posterior distribution of our state process $x$ at soem timepoint $t$ with the knowledge of all of the observations up until that point, or otherwise stated:

$p(x_t|y_{1:t})$

For any general state-space model, we can predict the most-likely next state, using the Chapman-Kolmogorov equation:

$p(x_t|y_{1:t-1}) = \int p(x_{t}|x_{t-1}) p(x_{t-1}|y_{1:t-1}) dx_{t-1}$

Of course, this integral is almost always intractable to compute, but we need not worry about that for now. Once we have calculated this value, we now need to update our beliefs about the state of $x_t$ once we have observed the next piece of evidence $y_t$. Using Baye's theorem we get that:

$p(x_t|y_{1:T}) \propto p(y_t|x_t)p(x_t|y_{1:t-1})$

It is worth examining this relation. What this means is that Bayesian Filtets, (the Kalman Filter included!) tell us how to update our beliefs about the state of our hidden variables based on a single new piece of evidence, $y_t$. Thus, the Kalman Filter, really is just a special case of computing this marginal conditional posterior distribution using a recursive algorithm. 
"""

# ╔═╡ 3ac3ee86-3426-4cf4-ba02-29d692348e9a
begin
	# set up parameters, we currently are assuming we know these, in practice we may know only a few, if any at all
	β = 0.75
	α = 0.9
	A = [1 1; -β (1 - α)]
	H = reshape([1., 0.], 1, 2)
	Q = 0.01 * I(2)
	R = [1]
	x₀ = [1, -1]
	P₀ = 0.01 * I(2)

	# set up a LDS object
	spring_LDS = SSM.LDS(A, H, nothing, Q, R, x₀, P₀, nothing, 1, 2, fill(false, 7))

	# now we can filter our observations
	x_filtered, p_filtered = SSM.KalmanFilter(spring_LDS, observation_sequence)
end

# ╔═╡ a28a94b6-e160-45e8-9762-80108aac176a
begin
	# generate confidence intervals for each t in T
	filter_std_errors = [sqrt.(diag(p_filtered[t, :, :])) for t in 1:100]

	pos_bounds = [1.96 * std_err[1] for std_err in filter_std_errors]
	vel_bounds = [1.96 * std_err[2] for std_err in filter_std_errors]

	# plot the results
	l2 = @layout [a; b]
	filter_1 = plot(state_sequence[:, 1], label="True Position", linewidth=:3)
	plot!(x_filtered[:, 1], ribbon=pos_bounds, label="Filtered Position", linewidth=:3)
	filter_2 = plot(state_sequence[:, 2], label="True Velocity", linewidth=:3)
	plot!(x_filtered[:, 2], ribbon=vel_bounds, label="Filtered Velocity", linewidth=:3)
	plot(filter_1, filter_2, layout=l2)
end

# ╔═╡ 43f18089-1ccb-4ecb-903e-3f252e7d6691
md"""From the above plot we can make a few observations. The position estimate is far less noisy than the measurement we obtained from the sensors and has far less error. This is unsurprising as we had a well developed model for how dynamic changes of position in the world actually *works*. Secondly, although we had no estimate of the velocity prior, we now have a seemingly reasonable estimate. We can however notice the failings. There are places where the 95% estimate of the state do not include the true value. With a less noisy sensor this may not be the case, but luckily we still have another trick up our sleeves: the Rauch-Tung-Striebel (RTS) or "Kalman" Smoother."""

# ╔═╡ 7edbb64b-ca39-4377-bdad-730876e35b27
md"""### RTS Smoother"""

# ╔═╡ 4ea51161-6402-46dc-9222-b687a59cf424
md"""The RTS smoother is the closed-form Bayesian Smoothing algorithm for the Linear Dynamical System with Gaussian observations. As opposed to filtering which aims to compute:

$p(x_t|y_{1:t})$

we want to compute the posterior distribution with respect to *all* of the data:

$p(x_t|y_{1:T})$
"""

# ╔═╡ 43b64fbd-62ad-42a4-9905-1062eeec4001
begin
	# we already set up parameters from the previous example, the usage is very similar to the Kalman Filter.
	x_smooth, p_smooth = SSM.KalmanSmoother(spring_LDS, observation_sequence, "RTS") 
end

# ╔═╡ 93a7e46f-a8e3-4bba-9e01-30b194b653f6
begin
	# generate confidence intervals for each t in T
	smoother_std_errors = [sqrt.(diag(p_smooth[t, :, :])) for t in 1:100]

	pos_bounds_smooth = [1.96 * std_err[1] for std_err in smoother_std_errors]
	vel_bounds_smooth = [1.96 * std_err[2] for std_err in smoother_std_errors]

	# plot the results
	smooth_1 = plot(state_sequence[:, 1], label="True Position", linewidth=:3)
	plot!(x_smooth[:, 1], ribbon=pos_bounds_smooth, label="Smoothed Position", linewidth=:3)
	smooth_2 = plot(state_sequence[:, 2], label="True Velocity", linewidth=:3)
	plot!(x_smooth[:, 2], ribbon=vel_bounds_smooth, label="Smoothed Velocity", linewidth=:3)
	plot(smooth_1, smooth_2, layout=l2)
end

# ╔═╡ 00800593-d432-420e-9e15-fd40fd7de7ee


# ╔═╡ bdaa735a-7490-4ddd-931b-f01e1382e740


# ╔═╡ 1a173cad-acec-4358-8147-8119ce6b7449
begin

	# set up a new LDS with no parameters (outside of the dimensionality) so that we can perform EM
	modelless_spring = SSM.LDS(;obs_dim=1, latent_dim=2)
	# perform EM
	SSM.KalmanFilterEM!(modelless_spring, observation_sequence)
end

# ╔═╡ Cell order:
# ╠═53c96240-c044-11ee-2f83-e93206e06ac6
# ╟─c3f10cbe-2505-47d3-a3de-340a8b7d63de
# ╠═2b64b709-ea92-49e8-8b47-218b4ecc56c7
# ╟─35cbfd54-05bb-4c22-9f9b-ae76995fd1c8
# ╟─2e5a7b3c-cebd-459c-8315-4be0d4f85596
# ╠═21dc64ac-c929-4eea-9568-8fbaef3e2e44
# ╠═dd275f99-1d29-4dbd-a804-7e64893074b0
# ╟─6fa55d13-f126-40f1-8b34-4221a70d36b3
# ╟─7fb9f5a9-6cf6-4f09-8932-f7daefd937b2
# ╠═885348ce-e907-47fd-a625-d9d1773d7025
# ╠═3ac3ee86-3426-4cf4-ba02-29d692348e9a
# ╠═a28a94b6-e160-45e8-9762-80108aac176a
# ╠═43f18089-1ccb-4ecb-903e-3f252e7d6691
# ╟─7edbb64b-ca39-4377-bdad-730876e35b27
# ╠═4ea51161-6402-46dc-9222-b687a59cf424
# ╠═43b64fbd-62ad-42a4-9905-1062eeec4001
# ╠═93a7e46f-a8e3-4bba-9e01-30b194b653f6
# ╠═00800593-d432-420e-9e15-fd40fd7de7ee
# ╠═bdaa735a-7490-4ddd-931b-f01e1382e740
# ╠═1a173cad-acec-4358-8147-8119ce6b7449

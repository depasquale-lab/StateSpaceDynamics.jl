# What is a Linear Dynamical System?

A **Linear Dynamical System (LDS)** is a mathematical model used to describe how a system evolves over time. These systems are a subset of **state-space models**, where the hidden state dynamics are continuous. What makes these models *linear* is that the latent dynamics evolve according to a linear function of the previous state. The observations, however, can be related to the hidden state through a nonlinear link function.

At its core, an LDS defines:

- **A state transition function**: how the internal state evolves from one time step to the next.
- **An observation function**: how the internal state generates the observed data.

# The Gaussian Linear Dynamical System

The **Gaussian Linear Dynamical System** — typically just referred to as an LDS — is a specific type of linear dynamical system where both the state transition and observation functions are linear, and all noise is Gaussian.

The generative model is given by:

$$
x_t \sim \mathcal{N}(A x_{t-1}, Q) \\
y_t \sim \mathcal{N}(C x_t, R)
$$

Where:

- $x_t$ is the hidden state at time $t$
- $y_t$ is the observed data at time $t$
- $A$ is the state transition matrix
- $C$ is the observation matrix
- $Q$ is the process noise covariance
- $R$ is the observation noise covariance

This can equivalently be written in equation form:

$$
x_t = A x_{t-1} + \epsilon_t \\
y_t = C x_t + \eta_t
$$

Where:

- $\epsilon_t \sim \mathcal{N}(0, Q)$ is the process noise
- $\eta_t \sim \mathcal{N}(0, R)$ is the observation noise

# The Poisson Linear Dynamical System

The **Poisson Linear Dynamical System** is a variant of the LDS where the observations are modeled as counts. This is useful in useful in fields like neuroscience where we are often interested in modeling spike count data. To relate the spiking data to the Gaussian latent variable, we use a nonlinear link function, specifically the exponential function. Thus our generative model is given by: 

```math
x_t \sim \mathcal{N}(A x_{t-1}, Q) \\
y_t \sim \text{Poisson}(\text{exp}(Cx_t + b))

```

Where:

- $b$ is a bias term

# Inference in Linear Dynamical Systems
In StateSpaceDynamics.jl, we directly maximize the complete-data log-likelihood function with respect to the latent states given the data and the parameters of the model. In other words, the **maximum a priori** (MAP) estimate of the latent state path is:

```math
\underset{x}{\text{argmax}}  \left[ \log p(x_0) + \sum_{t=2}^T \log p(x_t \mid x_{t-1}) + \sum_{t=1}^T \log p(y_t \mid x_t) \right]
```

This MAP estimation approach has the same computational complexity as traditional Kalman filtering and smoothing — $ \mathcal{O}(T) $ — but is significantly more flexible. Notably, it can handle **nonlinear observations** and **non-Gaussian noise** while still yielding **exact MAP estimates**, unlike approximate techniques such as the Extended Kalman Filter (EKF) or Unscented Kalman Filter (UKF).

## Newton's Method for Latent State Optimization

To find the MAP trajectory, we iteratively optimize the latent states using Newton’s method. The update equation at each iteration is:

```math
x^{(i+1)} = x^{(i)} - \left[ \nabla^2 \mathcal{L}(x^{(i)}) \right]^{-1} \nabla \mathcal{L}(x^{(i)})
```

Where:

- $\mathcal{L}(x)$ is the complete-data log-likelihood:

```math
\mathcal{L}(x) = \log p(x_0) + \sum_{t=2}^T \log p(x_t \mid x_{t-1}) + \sum_{t=1}^T \log p(y_t \mid x_t)
```

- $\nabla \mathcal{L}(x)$ is the gradient of the full log-likelihood with respect to all latent states,
- $\nabla^2 \mathcal{L}(x)$ is the Hessian of the full log-likelihood.

This update is performed over the entire latent state sequence $ x_{1:T} $, and repeated until convergence.

For **Gaussian models**, $ \mathcal{L}(x) $ is quadratic and Newton's method converges in a single step — recovering the exact Kalman smoother solution. But, for **non-Gaussian models**, the Hessian is not constant and the optimization is more complex. However, the MAP estimate can still be computed efficiently using the same approach as the optimization problem is still convex.


# Laplace Approximation of Posterior for Non-Conjugate Observation Models

In the case of non-Gaussian observations, we can use a Laplace approximation to compute the posterior distribution of the latent states. Notably, in the case of Gaussian Observations (which is conjugate with the Gaussian state model), the posterior is also Gaussian, and is the exact posterior. However, for non-Gaussian observations, we can approximate the posterior using a Gaussian distribution centered at the MAP estimate of the latent states. This approximation is given by:

```math 

p(x \mid y) \approx \mathcal{N}(x^{(*)}, -\left[ \nabla^2 \mathcal{L}(x^{(*)}) \right]^{-1})

```

Where:

- $ x^{(*)}$ is the MAP estimate of the latent states
- $\nabla^2 \mathcal{L}(x^{(*)})$ is the Hessian of the log-likelihood at the MAP estimate.

Despite, the requirement of inverting a Hessian of diomension (d x T) x (d x T), this is still computationally efficient, as the Markov structure of the model, renders the Hessian block-tridiagonal, and thus the inversion is not intractable.

# Learning in Linear Dynamical Systems

Given the latent structure of state-space models, we must rely on either the Expectation-Maximization (EM) or Variational Inference (VI) approaches to learn the parameters of the model. StateSpaceDynamics.jl supports both EM and VI. For LDS models, we can use Laplace EM, where we approximate the posterior of the latent state path using the Laplace approximation as outlined above. Using these approximate posteriors (or exact ones in the Gaussian case), we can apply closed-form updates for the model parameters.

```@autodocs
Modules = [StateSpaceDynamics]
Pages   = ["LinearDynamicalSystems.jl"]
```

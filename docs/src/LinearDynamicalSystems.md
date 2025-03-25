# LDS Models

## Mathematical Background

In the most general form, a linear dynamical system (LDS) is a state-space model with linear dynamics. This can be expressed as follows:

```math
\begin{align*}
x_{t+1} &= f(x_t)\\
y_t &= g(x_t)
\end{align*}
```

where $x_t$ is the latent state at time $t$, $y_t$ is the observed data at time $t$, $f$ is the state transition function, $g$ is the observation function. In the linear case, $f$ and $g$ are linear functions. The way we have written the above expression, we haven't made an explicit statement about the noise distribution--and this is on purpose. While the classic Linear-Gaussian Dynamical System (i.e., the Kalman Filter/Smoother), is often the immedate assumption, there's nothing mathematically preventing us from assuming other noise.

## The Gaussian Linear Dynamical System

The Gaussian Linear Dynamical System (GLDS) is the canonical state-space model. It is often defined two equivalent ways:

```math
X_t ~ \mathcal{N}(AX_{t-1}, Q)
Y_t ~ \mathcal{N}(CX_t, R)
```

where A is the state transition matrix, Q is the process noise covariance, C is the observation matrix, and R is the observation noise covariance. Equivalently, we can write it in equation form:

```math
X_t = AX_{t-1} + \epsilon_t
Y_t = CX_t + \eta_t
```

where $\epsilon_t \sim \mathcal{N}(0, Q)$ and $\eta_t \sim \mathcal{N}(0, R)$.


```@autodocs
Modules = [StateSpaceDynamics]
Pages   = ["LinearDynamicalSystems.jl"]
```
# LDS Models

## Mathematical Background

In the most general form, a linear dynamical system (LDS) is a state-space model with linear dynamics. This can be expressed as follows:

```math
\begin{align*}
x_{t+1} &= f(x_t)\\
y_t &= g(x_t)
\end{align*}
```

where $x_t$ is the latent state at time $t$, $y_t$ is the observed data at time $t$, $f$ is the state transition function, $g$ is the observation function. In the linear case, $f$ and $g$ are linear functions. The way we have written the above expression, we haven't made an explicit statement about the noise distribution--and this is on purpose. While the classic Linear-Gaussian Dynamical System (i.e., the Kalman Filter/Smoother), is often the immedate assumption, there's nothing mathematically preventing us from assuming othe rnoise 

## The Filtering Problem

In LDS models, one of the three major problems is filtering. Specifically, we are interested in estiamting the latent state $x_t$ given the observed data $y_1, \ldots, y_t$. Otherwise stated, we want to solve the following integral:


## The Smoothing Problem

## The Prediction Problem


When the state and observation noise are Gaussian, the LDS is a Gaussian LDS, often refered to as the Kalman filter/smoother. This model can be described as follows:

```math
\begin{align*}
x_{t+1} &\sim \mathcal{N}(A x_t + b, Q)\\
y_t &\sim \mathcal{N}(C x_t + d, R) 
\end{align*}
```

where $A$ is the state transition matrix, $C$ is the observation matrix, $b$ and $d$ are bias terms, and $Q$ and $R$ are the state and observation noise covariance matrices, respectively.



```@autodocs
Modules = [StateSpaceDynamics]
Pages   = ["LinearDynamicalSystems.jl"]
```
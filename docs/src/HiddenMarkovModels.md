# What is a Hidden Markov Model?

A **Hidden Markov Model (HMM)** is a graphical model that describes how systems change over time. When modeling a time series with $T$ observations using an HMM, we assume that the observed data $y_{1:T}$ depends on hidden states $x_{1:T}$ that are not observed. Specifically, an HMM is a type of **state-space model** in which the hidden states are discrete.

The three components of an HMM are as follows:

- **An initial state distribution ($\pi$):** which hidden states we are likely to start in.
- **A transition matrix ($A$):** how the hidden states evolve over time.
- **An emission model:** how the hidden states generate the observed data.

The generative model is given by:

```math
\begin{align}
    x_1 &\sim \text{Cat}(\pi) \\
    x_t &\mid x_{t-1} \sim \text{Cat}(A_{x_{t-1}, :}) \\
    y_t &\mid x_t \sim p(y_t \mid \theta_{x_t})
\end{align}
```


Where:

- $x_t$ is the hidden (discrete) state at time $t$
- $y_t$ is the observed data at time $t$
- $\pi$ is the initial state distribution
- $A$ is the state transition matrix
- $\theta_{x_t}$ are the parameters of the emission distribution for state $x_t$

The emission model can take many forms: Gaussian, Poisson, Bernoulli, categorical, etc... In the case of a Gaussian emission distribution, this becomes:

```math
y_t \mid (x_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)
```

Where:

- $\mu_k$ is the mean of the emission distribution for state $k$
- $\Sigma_k$ is the covariance of the emission distribution for state $k$

# What is a Generalized Linear Model - Hidden Markov Model

A **Hidden Markov Model - Generalized Linear Model (GLM-HMM)** - also known as **Switching Regression Model** - is an extension to classic HMMs where the emission models are state-dependent GLMs that link an observed input to an observed output. This formulation allows each hidden state to define its own regression relationship between inputs and outputs, enabling the model to capture complex, state-dependent dynamics in the data. Currently, StateSpaceDynamics.jl support Gaussian, Bernoulli, Poisson, and Autoregressive GLMs as emission models.

The generative model is as follows:

```math
\begin{align}
    x_1 &\sim \text{Cat}(\pi) \\
    x_t &\mid x_{t-1} \sim \text{Cat}(A_{x_{t-1}, :}) \\
    y_t &\mid x_t, u_t \sim p(y_t \mid \theta_{x_t}, u_t)
\end{align}
```

Where:

- $x_t$ is the hidden (discrete) state at time $t$
- $y_t$ is the observed output at time $t$
- $u_t$ is the observed input (covariate) at time $t$
- $\theta_{x_t}$ are the parameters of the GLM emission model for state $x_t$

### Example Emission Models

For example, if the emission is a Gaussian GLM:

```math
y_t \mid (x_t = k), u_t \sim \mathcal{N}(\mu_k + \beta_k^\top u_t, \sigma_k^2)
```

Where:

- $\beta_k$ are the regression weights for state $k$
- $\sigma_k^2$ is the state-dependent variance
- $\mu_k$ is the state-dependent bias

If the emission is Bernoulli (for binary outputs):

```math
y_t \mid (x_t = k), u_t \sim \text{Bernoulli} \left( \sigma \left( \mu_k + \beta_k^\top u_t \right) \right)
```

Where:

- $\beta_k$ are the regression weights for state $k$
- $\sigma(\cdot)$ is the logistic sigmoid function for binary outputs
- $\mu_k$ is the state-dependent bias

# Learning in an HMM

`StateSpaceDynamics.jl` implements Expectation-Maximization (EM) for parameter learning in both HMMs and GLM-HMMs. EM is an iterative method for finding maximum likelihood estimates of the parameters in graphical models with hidden variables. 

### Expectation Step (E-step)
In the **expectation step (E-step)**, we calculate the posterior distribution of the latent states given the current parameters of the model:

```math
p(X \mid Y, \theta_{\text{old}})
```

We use dynamic programming to efficiently calculate this posterior using the **forward** and **backward** recursions for HMMs. This posterior is then used to construct the expectation of the complete data log-likelihood, also known as the **Q-function**:

```math
Q(\theta, \theta_{\text{old}}) = \sum_X p(X \mid Y, \theta_{\text{old}}) \ln p(Y, X \mid \theta)
```

### Maximization Step (M-step)
In the **maximization step (M-step)**, we maximize this expectation with respect to the parameters $\theta$. Specifically:

- For the initial state distribution and the transition matrix, we use **analytical updates** for the parameters, derived using Lagrange multipliers.
- For emission models in the case of HMMs, we also implement **analytical updates**.
- If the emission model is a GLM, we use `Optim.jl` to **numerically optimize** the objective function.

# Inference in an HMM

For state inference in Hidden Markov Models (HMMs), we implement two common algorithms:

### Forward-Backward Algorithm
The **Forward-Backward** algorithm is used to compute the **posterior state probabilities** at each time step. Given the observed data, it calculates the probability of being in each possible hidden state at each time step, marginalizing over all possible state sequences.

### Viterbi Algorithm
The **Viterbi** algorithm is used for **best state sequence labeling**. It finds the most likely sequence of hidden states given the observed data. This is done by dynamically computing the highest probability path through the state space, which maximizes the likelihood of the observed sequence.

# Reference

For a complete mathematical formulation of the relevant HMM and HMM-GLM learning and inference algorithms, we recommend **Pattern Recognition and Machine Learning, Chapter 13** by **Christopher Bishop**.

```@autodocs
Modules = [StateSpaceDynamics]
Pages   = ["HiddenMarkovModels.jl"]
```

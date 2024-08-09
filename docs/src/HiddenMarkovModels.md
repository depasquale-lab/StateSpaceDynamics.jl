# Hidden Markov Models
```@meta
CollapsedDocStrings = true
```

A Hidden Markov Model (HMM) is a time series model with unobservable latent variables that follow a Markov process. Condititional distributions (when conditioned on the latents) are often chosen to be common distributions (e.g. Gaussian).


```@docs
HiddenMarkovModel
sample(model::HiddenMarkovModel, data...; n::Int)
loglikelihood(model::HiddenMarkovModel, data...)
weighted_initialization(model::HiddenMarkovModel, data...)
fit!(model::HiddenMarkovModel, data...; max_iters::Int=100, tol::Float64=1e-6)
class_probabilities(model::HiddenMarkovModel, data...)
viterbi(model::HiddenMarkovModel, data...)
```
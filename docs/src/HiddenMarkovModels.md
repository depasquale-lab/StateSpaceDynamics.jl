# Hidden Markov Models
```@meta
CollapsedDocStrings = true
```

A Hidden Markov Model (HMM) is a time series model with unobservable latent variables that follow a Markov process. Condititional distributions (when conditioned on the latents) are often chosen to be common distributions (e.g. Gaussian).

```@autodocs
Modules = [StateSpaceDynamics]
Pages   = ["HiddenMarkovModels.jl", "HMMConstructors.jl"]
```
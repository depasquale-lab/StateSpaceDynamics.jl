---
title: 'StateSpaceDynamics.jl: A Julia package for probabilistic state space models (SSMs)'
tags:
  - State space models
  - Julia
authors:
  - name: Ryan Senne
    orcid: 0000-0003-3776-4576
    affiliation: "1,2"
  - name: Zachary Loschinskey
    orcid: 0009-0005-2831-2641
    affiliation: "1"
  - name: Carson Loughridge
    affiliation: "1"
  - name: James Fourie
    affiliation: "1"
  - name: Brian D. DePasquale
    orcid: 0000-0003-4508-0537
    affiliation: "1,2"

affiliations:
 - name: Department of Biomedical Engineering, Boston University
   index: 1
 - name: Graduate Program for Neuroscience, Boston University
   index: 2
date: 5 August 2024

bibliography: paper.bib
---

# Summary

State-space models (SSMs) are powerful tools for modeling time series data that naturally arise across a variety of domains, including neuroscience, finance, and engineering. The unifying principle of these models is that they assume that an observation sequence, $Y_1, Y_2,...,Y_T$, is generated through an underlying hidden latent sequence, $X_1, X_2,...,X_T$. This general framework encompasses two of the most popular models for time series analysis: the Hidden Markov Model (HMM) and the (Gaussian) Linear Dynamical System (LDS, i.e., the Kalman filter). Thus, SSMs provide a probabilistic framework for describing the temporal evolution of many phenomena, and their generality naturally leads to widely applicable use cases. We introduce `StateSpaceDynamics.jl`, an open source, modular package designed to be fast, readable, and self contained for the express purpose of fitting a plurality of SSMs, easily in Julia. [@SSDjl2024]

# Statement of need

Advancements in systems neuroscience have enabled the collection of massive, multivariate, and complex time series datasets, where simultaneous recordings from hundreds to thousands of neurons are increasingly common. Interpreting these high-dimensional recordings presents a significant challenge. Recent modeling approaches suggest that neural activities can be effectively characterized by a set of latent factors evolving over a low-rank manifold. Consequently, there is a growing need for models that combine dimensionality reduction with temporal dynamics, for which state-space models (SSMs) provide a natural framework.

While advanced SSM implementations exist in Python—such as the SSM package [@PySSM2022] and Dynamax [@ChangUnknown-wn], the Julia programming language lacks an equivalent library that meets the needs of modern neuroscientists. Existing Julia offerings, like StateSpaceModels.jl [@SSMjl2024], are limited to Gaussian observation models and rely on analytical calculation of the marginal log-likelihood function. This fundamental limitation precludes the analysis of non-Gaussian observations, which are common in neuroscience where spike counts often follow Poisson or other discrete distributions. Furthermore, the requirement for analytical computation of the marginal likelihood integral:

\begin{equation}
p(y_{1:T}|\theta) = \int_{x_{1:T}} p(y_{1:T}|x_{1:T}, \theta) p(x_{1:T}|\theta) dx_{1:T}
\end{equation}

restricts the development of learning algorithms for non-conjugate observation models. To address these limitations, we have developed StateSpaceDynamics.jl, which provides a flexible framework for fitting a variety of SSMs, including non-Gaussian observation models, while maintaining computational efficiency.

# Package design

To address these limitations, we have developed `StateSpaceDynamics.jl`, which employs a previously advocated approach of directly maximizing the complete-data log-likelihood with respect to the hidden state path for Linear Dynamical System models [@Paninski2010-ns]. By leveraging the block-tridiagonal structure of the Hessian matrix, this method allows for the exact computation of the Kalman smoother in O(T) time [@Paninski2010-ns]. Furthermore, it facilitates the generalization of the Rauch–Tung–Striebel (RTS) smoother to accommodate other observation noise models (e.g., Poisson and Bernoulli), requiring only the computation of the gradient and Hessian of the new model to obtain an *exact* maximum a posteriori (MAP) path [@NIPS2011_7143d7fb].

Furthermore, given the analytical Hessian matrices are available, once can make use of this to perform an approximate EM algorithm by generating an LaPlace approxiamtion of the posterior distirubtion of the latent states. One can easily make use of fast inversion algorithms of the negative Hessian (i.e., Fisher Information Matrix), which are block-triadiagonal [@Rybicki1990-ky]. From here one can compute the approximate second moments of the posterior distribution i.e., $\text{Cov}(X_t, X_t)$ and $\text{Cov}(X_t, X_{t-1})$, and use the analytical updates of the canonical LDS [@Paninski2010-ns; @Bishop2006-kv]. This approach becomes exact EM in the case of Gaussian observations.

Lastly, `StateSpaceDynamics.jl` provides implementations od discrete state space models i.e., Hidden Markov Models, and the ability to fit these models using the EM algorithm. While this is not the primary development target of the package, these models are necessary for the development of hierarchichal models, e.g., the switching LDS (SLDS) and the recurrent switching LDS (rSLDS) [@Linderman2016-xe; @Murphy1998-bk]. However, the recent development of HiddenMarkovModels.jl, may make this feature redundant, and our future work my entail directly interfacing with this package [@HMMjl2024]. Nonetheless, we provide a suite of HMM models popular in neuroscience including the classic Gaussian HMM and GLM-HMMs.

By providing these features, `StateSpaceDynamics.jl` fills a critical gap in the Julia ecosystem, offering modern computational neuroscientists the tools necessary to model complex neural data with state-space models that incorporate both dimensionality reduction and temporal dynamics.

# Example

![Model architecture](model.png)\

Consider the Poisson Linear Dynamical System (PLDS):

\begin{align}
X_t &\sim \mathcal{N}(AX_{t-1}, Q) \\
Y_t &\sim \text{Poisson}(f(CX_t + b)\Delta t) \\
X_0 &\sim \mathcal{N}(\mu_0, \Sigma_0)
\end{align}

# Availability

``StateSpaceDynamics.jl`` is publicly available under the [GNU license](https://github.com/depasquale-lab/StateSpaceDynamics.jl/blob/main/LICENSE) at <https://github.com/depasquale-lab/StateSpaceDynamics.jl>.

# Conclusion

# Author contributions

RS did XXX. CL did XXX. ZL did XXX. BD oversaw development, provided mentorship, and secured funding.

# Acknowledgements

This work was supported by the Biomedical Engineering Department at Boston University.

# References
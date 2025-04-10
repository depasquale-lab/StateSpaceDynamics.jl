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

State-space models (SSMs) are powerful tools for modeling time series data that naturally arise in a variety of domains, including neuroscience, finance, and engineering. The unifying principle of these models is they assume an observation sequence, $Y_1, Y_2,...,Y_T$, is generated through an underlying latent sequence, $X_1, X_2,...,X_T$. This framework encompasses two popular models for time series analysis: the hidden Markov model (HMM) and the (Gaussian) linear dynamical system (LDS, i.e., the Kalman filter). Thus, SSMs provide a probabilistic framework for describing the temporal evolution of many phenomena, and their generality naturally leads to a variety of use cases. We introduce `StateSpaceDynamics.jl` [@SSDjl2024], an open source, modular package designed to be fast, readable, and self contained for the purpose of easily fitting a plurality of SSMs in the Julia language.

# Statement of need

Advances in neuroscience have enabled the collection of massive, multivariate, and complex time series datasets, where simultaneous observations from hundreds to thousands of neurons are increasingly common. Interpreting these high-dimensional datasets presents significant challenges. Recent modeling approaches suggest that neural activity can be characterized by a set of latent factors evolving across a low-dimensional manifold. Consequently, there is a growing need for models that combine dimensionality reduction with temporal dynamics, for which state-space models provide a natural framework.

While state space model implementations exist in Python, such as the SSM package [@PySSM2022] and Dynamax [@ChangUnknown-wn], the Julia programming language lacks an equivalent that meets the needs of modern neuroscientists. Existing Julia offerings, like `StateSpaceModels.jl` [@SSMjl2024], can accomdodate continuous-state SSMs (e.g., LDS) but are limited to Gaussian observation models and rely on analytical calculation of the marginal log-likelihood. This latter limitation precludes model inferenece and parameter learning for non-conjugate observations which are common in neuroscience, where neural activity follow Poisson or other discrete distributions. Packages for performing inference and learning using sampling based methods exist in Julia (such as `Turing.jl` [@Turing2018]) but are computationally inefficeint compared to tailored approaches based on Expectation-Maximization (EM). For discrete SSMs, an existing Julia offering, `HiddenMarkovModels.jl` [@HMMjl2024], is efficient and scalable but not intentionally designed with the functionality for mixing models that contain both discrete and continuous latent variable, such as the switching linear dynamical system model (SLDS) [@Linderman2016-xe; @slds] increasingly used in neuroscience.

# Package design

 To address these limitations, we developed `StateSpaceDynamics.jl`, which provides a flexible framework for fitting a variety of SSMs--including non-Gaussian observation models and models that mix discrete and continous latents--while maintaining computational efficiency. 

For continuous latent variable models, (e.g., LDS) `StateSpaceDynamics.jl` employs a previously advocated approach of directly maximizing the complete-data log-likelihood with respect to the hidden state path [@Paninski2010-ns]. By leveraging the block-tridiagonal structure of the Hessian matrix, this method allows for the exact computation of the Kalman smoother in O(T) time [@Paninski2010-ns]. Furthermore, it facilitates the generalization of the Rauch–Tung–Striebel (RTS) smoother to accommodate other observation models (e.g., Poisson and Bernoulli), requiring only the computation of the gradient and Hessian of the new model to obtain an *exact* maximum a posteriori (MAP) path [@NIPS2011_7143d7fb].

Using analytically computable Hessians, `StateSpaceDynamics.jl` performs approximate EM for non-Gaussian models via Laplace approximation of the latent posterior. Speed is maintained by using fast inversion algorithms of the negative Hessian (i.e., Fisher Information Matrix), which are block-triadiagonal [@Rybicki1990-ky]. From here `StateSpaceDynamics.jl` computes the approximate second moments of the posterior i.e., $\text{Cov}(X_t, X_t)$ and $\text{Cov}(X_t, X_{t-1})$, and uses the analytical updates of the canonical LDS [@Paninski2010-ns; @Bishop2006-kv]. This approach becomes exact EM for Gaussian observations.

Lastly, `StateSpaceDynamics.jl` provides implementations of discrete state space models i.e., hidden Markov models, and the ability to fit these models using EM. While this is not the primary development target of the package, these models are necessary for the development of hierarchichal models that mix discrete and continuous latents, e.g., the switching LDS (SLDS) and the recurrent switching LDS (rSLDS) [@Linderman2016-xe; @Murphy1998-bk; @slds] which have become immensely popular in neuroscience and require similarly tailored computational routines for efficient inference and learning. To illustrate the functionality of `StateSpaceDynamics.jl` for this model class, we include an implementation of the SLDS fit via stuctured variational EM (vEM) [@slds]. The development of `HiddenMarkovModels.jl`, may make our approach to discrete model learning redundant, and future work may entail directly interfacing with this package [@HMMjl2024]. Nonetheless, we provide a suite of HMM models popular in neuroscience including the classic Gaussian HMM and a variety of input-output HMMs [@NIPS1994_8065d07d], commonly referred to as generalized linear model-HMMs (GLM-HMMs) [@Ashwood2022] in neuroscience.

By providing these features, `StateSpaceDynamics.jl` fills a critical gap in the Julia ecosystem, offering modern computational neuroscientists the tools to model complex neural data with state-space models that incorporate both dimensionality reduction and temporal dynamics.

# Benchmarks

To evaluate the performance of `StateSpaceDynamics.jl`, we conducted two benchmarking studies focusing on fitting a Gaussian LDS and a Bernoulli GLM-HMM. For the Gaussian LDS benchmark, we compared our package against two alternatives: the NumPy-based Kalman filter-smoother package `pykalman` and the more recent JAX-based `Dynamax`. We intentionally excluded `StateSpaceModels.jl` from our comparison as its scope is geared towards structured time-series models.

For our Gaussian LDS experiments, we constructed a synthetic dataset as follows. The state transition matrix $A$ was generated as a random $n$-dimensional rotation matrix, while the observation matrix $C$ was created as a random $m \times n$ matrix. Both the state noise covariance $Q$ and observation noise covariance $R$ were set to identity matrices. To ensure a fair comparison, all packages were initialized using identical random parameters, after which we executed the EM algorithm for 100 iterations. We conducted these benchmarks using `PythonCall.jl` and `BenchmarkTools.jl` to ensure accurate timing measurements.

To thoroughly assess performance across different scales, we tested three sequence lengths ($T = 100, 500, 1000$) and explored various dimensionality combinations, with state dimensions $n = 2, 4, 8$ and observation dimensions $m = 2, 4, 8$.

\begin{figure}
\includegraphics[width=\textwidth]{benchmark_plot.pdf}
\end{figure}

For the subsequent benchmarking experiments, we compared `StateSpaceDyamics.jl`, `HiddenMarkovModels.jl`, and `Dynamax` in their abilities to fit a Bernoulli HMM-GLM. We completed the benchmarking on the same combinations of sequence lengths, state dimensions, and observation dimensions as in the Gaussian LDS benchmarking. Synthetic datasets were generated for each of these combinations by sampling from a randomly generated Bernoulli HMM-GLM using `StateSpaceDynamics.jl`. All packages were initialized to the same random parameters and the EM algorithm was run for 200 iterations. We disabled the convergence criteria to ensure each package completed all 200 iterations of EM. These benchmarks were conducted using `PythonCall.jl` and `BenchmarkTools.jl` to ensure timing accuracy.

\begin{figure}
\includegraphics[width=\textwidth]{SwitchingBernoulli_Benchmark_Plot.pdf}
\end{figure}

In our benchmarking, we find that in the LDS case, both variants of StateSpaceDynamics.jl are substantially faster than pykalman across all sequence lengths and dimension configurations. More generally, StateSpaceDynamics.jl outperforms Dynamax at shorter sequence lengths across various dimensional settings. However, Dynamax exhibits superior scaling behavior. This difference is likely due to our use of direct maximization of the complete-data log-likelihood. As the sequence length increases, the associated Hessian matrix grows in size and memory requirements, which can negatively impact scaling. That said, this part of our package has not yet been optimized, and future releases will likely address these inefficiencies. We believe that the added generality of our method justifies the current tradeoff.

In our GLM-HMM benchmarks, HiddenMarkovModels.jl consistently performs best across a range of sequence lengths and dimensional configurations. As previously noted, our implementation of these models serves primarily to provide a seamless interface with our continuous state-space models and is not yet optimized to the same extent as HiddenMarkovModels.jl. These benchmarks underscore the potential for a future release that leverages direct interoperability with that package. Furthermore, similar to HiddenMarkovModels.jl, our package generally outperforms Dynamax. However, this trend does not hold for larger state dimensions, again suggesting that there is room for improving our scaling performance.

Taken together, these benchmarks demonstrate the competitiveness of StateSpaceDynamics.jl for fitting state-space models.

# Availability

``StateSpaceDynamics.jl`` is publicly available under the [GNU license](https://github.com/depasquale-lab/StateSpaceDynamics.jl/blob/main/LICENSE) at <https://github.com/depasquale-lab/StateSpaceDynamics.jl>.

# Conclusion

`StateSpaceDynamics.jl` fills an existing gap in the Julia ecosystem for general state-space modeling that exist in Python. Importantly, our package's approach is simple enough that other candidate state-space models can be easily implemented. Further, this work provides a foundation for future development of more advanced state-space models, such as the rSLDS, which are essential for modeling complex neural data. We expect that this package will be of interest to computational neuroscientists and other researchers working with high-dimensional time series data and we are currenrtly using its functionality in three separate projects.

# Author contributions

RS (Ryan Senne) was the primary developer of `StateSpaceDynamics.jl`, implementing the core algorithms, designing the package architecture, and writing the manuscript. ZL (Zachary Loschinskey) was the secondary developer, whose contributions include optimizing and extending HMM/HMM-GLM functionality, implementing core multi-trial EM algorithms, and assisting with sLDS development. CL (Carson Loughridge) and JF (James Fourie) contributed to package development, including implementation of key features, testing, and documentation. BDD (Brian D. DePasquale) conceived the project, provided theoretical guidance and technical oversight throughout development, secured funding, and supervised the work. All authors reviewed and approved the final manuscript.

# Acknowledgements

This work was supported by the Biomedical Engineering Department at Boston University.

# References
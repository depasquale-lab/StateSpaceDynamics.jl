


	# change this to your system path
	using Pkg

	# In the package manager, use "dev path\to\local\package\under\development"
	using SSM

	using LinearAlgebra
	using Distributions
	using Random
	using Plots
	using CSV
	using DataFrames
end



## Example

Given some normal distributed clusters of data:

```julia
data = CSV.read("data.csv", DataFrame)
```

![gaussian_data](assets/gmm_data_plot.png)


We then fit a model:
```julia
k = 2
data_dim = 2
gmmModel = SSM.GaussianMixtureModel(k, data_dim)

SSM.fit!(gmmModel, data)
```

Overlaying the trained model reveals a good fit:

![gaussian_model](assets/gmm_model_plot.png)

# now lets fit the model!
SSM.fit!(gmmEstimate, data)


# DISPLAY MODEL PLOT HERE



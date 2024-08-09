# LDS Models

Overview tbc...

## LDS Constructor
```@docs
LDS
``` 

## KalmanFilter
```@docs
KalmanFilter(l::LDS, y::AbstractArray)
``` 

## RTSSmoother
```@docs
RTSSmoother(l::LDS, y::AbstractArray)
``` 

## DirectSmoother
```@docs
DirectSmoother(l::LDS, y::AbstractArray, tol::Float64=1e-6)
``` 

## KalmanSmoother
```@docs
KalmanSmoother(l::LDS, y::AbstractArray, method::String="RTS")
``` 

## KalmanFilterEM!
```@docs
KalmanFilterEM!(l::LDS, y::AbstractArray, max_iter::Int=1000, tol::Float64=1e-6)
``` 

## Loglikelihood
```@docs
loglikelihood(X::AbstractArray, l::LDS, y::AbstractArray)
marginal_loglikelihood(l::LDS, v::AbstractArray, j::AbstractArray)
``` 


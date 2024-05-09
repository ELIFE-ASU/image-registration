using Base.Threads, Turing, Distributions, StatsBase, Plots, StatsPlots, MCMCChains, DataFrames, CSV, ProgressMeter

Turing.setprogress!(false)

import Logging

@model AR(x, N) = begin
    α ~ Normal(0, 1)
    β ~ Uniform(-1, 1)
    for t in 2:N
        μ = α + β * x[t-1]
        x[t] ~ Normal(μ, 1)
    end
end

struct ARModel
    α::Float64
    β::Float64
end

function predict(model::ARModel, x₀, N)
    p = zeros(Float64, N + 1)
    p[1] = x₀
    𝒩 = Normal(0, 1)
    for t in 1:N
        p[t + 1] = model.α + model.β*p[t] + rand(𝒩)
    end
    p
end

function fit(data::AbstractVector{Float64}, args...; kwargs...)
    N = length(data)
    logger = Logging.SimpleLogger(stderr, Logging.Error)
    chain = Logging.with_logger(logger) do
        sample(AR(data, N), args...; kwargs...)
    end
    α, β = mean(chain[:α]), mean(chain[:β])
    return ARModel(α, β)
end

function fit(filename::AbstractString; thread=true)
    df = DataFrame(CSV.File(filename))

    T = length(unique(df.timestep))

    data = mapslices(zscore, reshape(df.c + df.r, T, nrow(df) ÷ T), dims=1)
    train = data[1:floor(Int, 0.9 * T), :]

    models = Array{ARModel}(undef, size(data,2))
    if thread
        Threads.@threads :static for i in 1:size(data, 2)
            models[i] = fit(train[:, i], NUTS(), 100000)
        end
    else
        @showprogress for i in 1:size(data, 2)
            models[i] = fit(train[:, i], NUTS(), 100000)
        end
    end

    models
end 

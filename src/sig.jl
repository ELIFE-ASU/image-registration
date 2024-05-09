using Distributions, Imogen, Random

struct EmpericalDist{T <: Number}
    gt::T
    count::Int
    values::Vector{T}
    p::Float64
    se::Float64

    function EmpericalDist(gt::T, values::AbstractVector{T}) where {T <: Number}
        N = length(values)
        p = count(gt .≤ values) / N
        se = sqrt((p * (1-p))/N)
        new{T}(gt, N, values, p, se)
    end
end

function sig(::Type{EmpericalDist}, func, firstarg, args...; nperm=1000, kwargs...)
    gt = func(firstarg, args...; kwargs...)
    values = zeros(typeof(gt), nperm + 1)
    values[1] = gt
    src = copy(firstarg)
    for i in 1:nperm
        values[i + 1] = func(shuffle!(src), args...; kwargs...)
    end
    EmpericalDist(gt, values)
end

pvalue(dist::EmpericalDist) = dist.p
pvalue(dist::EmpericalDist, gt) = (1 + count(gt .≤ dist.values)) / (dist.count + 1)

stder(dist::EmpericalDist) = dist.se
stder(dist::EmpericalDist, gt) = let p = pvalue(dist, gt)
    sqrt(p * (1 - p) / (dist.count + 1))
end

struct AnalyticDist{T <: Number}
    gt::T
    N::Int
    dof::Float64
    dist::Chisq
    p::Float64
    function AnalyticDist(gt::T, N::Int, dof::Number) where {T <: Number}
        dist = Chisq(dof)
        p = 1 - cdf(dist, 2 * N * gt)
        new{T}(gt, N, dof, dist, p)
    end
end

function sig(::Type{AnalyticDist}, func, firstarg, args...; kwargs...)
    gt = func(firstarg, args...; kwargs...)
    N = length(firstarg)
    dof = (maximum(firstarg) - 1) * prod(map(a -> maximum(a) - 1, args))

    AnalyticDist(gt, N, dof);
end

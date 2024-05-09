@everywhere begin
    using Distributed, Imogen, LinearAlgebra, Random, SharedArrays
    
    abstract type PermTest end
    abstract type ExtremePermTest <: PermTest end
    abstract type ExtremePositivePermTest <: PermTest end
    
    struct Sig
        value::Float64
        p::Float64
        nperm::Int
    end
    Base.isless(x::Sig, y::Sig; α=0.05) = !(x.p < α && y.p ≥ α) && ((x.p ≥ α && y.p < α) || isless(x.value, y.value))
    
    fails(::Type{ExtremePermTest}, gt::Real, perm::Real) = Int(abs(gt) ≤ abs(perm))
    fails(::Type{ExtremePositivePermTest}, gt::Real, perm::Real) = Int(gt ≤ perm)

    function sig_mutualinfo(P::Type{<:PermTest}, xs::AbstractVector, ys::AbstractVector; α=0.05, nperm=100, kwargs...)
        N = size(xs, 1)
        gt = mutualinfo(xs, ys; kwargs...)
        count = 1
        for _ in 1:nperm
            count += fails(P, gt, mutualinfo(xs, ys[randperm(N)]; kwargs...))
        end
        p = count / (nperm + 1)
        Sig(gt, p, nperm)
    end
    function sig_mutualinfo(xs::AbstractVector, ys::AbstractVector; kwargs...)
        sig_mutualinfo(ExtremePositivePermTest, xs, ys; kwargs...)
    end

    function sig_transferentropy(P::Type{<:PermTest}, xs::AbstractVector, ys::AbstractVector, k::Int, τ::Int, delay::Int; α=0.05, nperm=100, kwargs...)
        N = size(xs, 1)
        gt = transferentropy(xs, ys; k=k, τ=τ, delay=delay, kwargs...)
        count = 1
        for _ in 1:nperm
            count += fails(P, gt, transferentropy(xs, ys[randperm(N)]; k=k, τ=τ, delay=delay, kwargs...))
        end
        p = count / (nperm + 1)
        Sig(gt, p, nperm)
    end
    function sig_transferentropy(P::Type{<:PermTest}, xs::AbstractVector, ys::AbstractVector, k::Int, τ::Int, delay::AbstractVector; kwargs...)
        findmax([sig_transferentropy(ExtremePositivePermTest, xs, ys, k, τ, d; kwargs...) for d in delay])
    end
    function sig_transferentropy(xs::AbstractVector, ys::AbstractVector, k::Int, τ::Int, delay; kwargs...)
        sig_transferentropy(ExtremePositivePermTest, xs, ys, k, τ, delay; kwargs...)
    end
end

function mi_matrix(P::Type{<:PermTest}, series; nperm=100, kwargs...)
    N = size(series,2)
    mi = Array{Sig}(undef, N, N)
    @sync for i in 1:N, j in i:N
        if i == j
            mi[i,j] = Sig(Imogen.entropy(series[:,i]), 0.0, 1)
        else
            @async mi[i,j] = @fetch sig_mutualinfo(P, series[:,i], series[:,j]; nperm=nperm, kwargs...)
        end
    end
    mi
end
mi_matrix(series; kwargs...) = mi_matrix(ExtremePositivePermTest, series; kwargs...)

function estimate_embedding(series::AbstractVector, ks, τs)
    maxai = -1.0
    maxk = 0
    maxτ = 0
    for k in ks
        for τ in τs
            ai = activeinfo(series; k=k, τ=τ)
            if ai > maxai
                maxai = ai
                maxk = k
                maxτ = τ
            end
        end
    end
    maxk = (maxk == 0) ? 1 : maxk
    maxτ = (maxτ == 0) ? 1 : maxτ
    
    maxk, maxτ
end

function estimate_embedding(series::AbstractMatrix, ks, τs)
    k = zeros(Int, size(series, 2))
    τ = zeros(Int, size(series, 2))
    for i in 1:size(series, 2)
        k[i], τ[i] = estimate_embedding((@view series[:, i]), ks, τs)
    end
    k, τ
end
estimate_embedding(series::AbstractMatrix) = estimate_embedding(series, 1:floor(Int, log2(size(series, 1)))-1, 1:(size(series, 1) ÷ 10))


function te_matrix(P::Type{<:PermTest}, series, ks, τs, delays; nperm=100, kwargs...)
    N = size(series, 2)
    te = Array{Sig}(undef, N, N)
    d = zeros(Int64, N, N)
    @sync for i in 1:N, j in 1:N
        @async te[i,j], d[i,j] = @fetch sig_transferentropy(P, series[:, i], series[:, j], ks[j], τs[j], delays; nperm=nperm, kwargs...)
    end
    te, d
end
te_matrix(series, ks, τs, delays; kwargs...) = te_matrix(ExtremePositivePermTest, series, ks, τs, delays; kwargs...)

distance(com) = let N = size(com, 2)
    [norm(com[:, i] - com[:, j]) for i in 1:N, j in 1:N]
end

function findnn(n::Int; θ=0.2, lo=1, hi=10)
    k = 1
    while sqrt((k+1)/n) ≤ θ
        k += 1
    end
    clamp(k, lo, hi)
end
findnn(A::AbstractArray, dim=1; kwargs...) = findnn(size(A, dim); kwargs...)

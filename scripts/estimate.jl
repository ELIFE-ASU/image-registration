using Base.Iterators, Base.Threads, CSV, DataFrames, Distances, Distributions, Imogen, Random, ProgressMeter

function sig(func, gt, firstarg, args...; nperm=100, α=0.1, kwargs...)
    count = 1
    src = copy(firstarg)
    for _ in 1:nperm
        count += (gt ≤ func(shuffle!(src), args...; kwargs...))
    end
    p = count / (nperm + 1)
    (p < α) ? gt : zero(gt)
end

function refine!(::typeof(transferentropy), te, C; kwargs...)
    ncells = size(C, 2)
    indices = findall(te .!= zero(eltype(te)))
    p = Progress(length(indices))
    @threads for idx in indices
        i, j = idx.I
        #  others = [(@view C[:,k]) for k in setdiff(1:ncells, [i,j])]
        #  te[i,j] = @views sig(transferentropy, te[i,j], C[:,i], C[:,j], others...; kwargs...)
        #  others = [(@view C[:,k]) for k in setdiff(1:ncells, [i,j])]
        te[i,j] = @views sig(transferentropy, te[i,j], C[:,i], C[:,j]; kwargs...)
        next!(p)
    end
    te
end

function refine!(::typeof(activeinfo), ai, C; kwargs...)
    ncells = size(C, 2)
    indices = findall(ai .!= zero(eltype(ai)))
    p = Progress(length(indices))
    @threads for idx in indices
        τ, k, i = idx.I
        ai[τ,k,i] = @views sig(activeinfo, ai[τ,k,i], C[:,i]; k=k, τ=τ, kwargs...)
        next!(p)
    end
    ai
end

function optimalparams(ai)
    map(1:size(ai,3)) do i
        A = Matrix(ai[:,:,i])
        params = findall(A .≈ maximum(A))
        if isempty(params)
            (τ=1, k=1)
        else
            _, i = findmin(map(idx -> *(idx.I...), params))
            (τ, k) = params[i].I
            (; τ, k)
        end
    end
end

function aiphase(C; θ=1e-8, krange=1:floor(Int, log(size(C,1))), τrange=1:(size(C,1) ÷ 10), kwargs...)
    ncells = size(C, 2)

    @info "Initial active information pass"
    ai = zeros(length(τrange), length(krange), ncells)
    indices = collect(product(τrange, krange, 1:ncells))
    p = Progress(length(indices))
    @threads for (τ, k, i) in indices
        ai[τ,k,i] = activeinfo(C[:,i]; k=k, τ=τ)
        if ai[τ,k,i] < θ
            ai[τ,k,i] = zero(ai[τ,k,i])
        end
        next!(p)
    end

    nperm, α = 100, 0.1
    @info "First refinement" nperm α
    refine!(activeinfo, ai, C; nperm=nperm, α=α, kwargs...)

    nperm, α = 1000, 0.05
    @info "Final refinement" nperm α
    refine!(activeinfo, ai, C; nperm=nperm, α=α, kwargs...)

    @info "Identifying optimal parameters"
    params = optimalparams(ai)
    finalai = zeros(ncells)
    for (i, (τ, k)) in enumerate(params)
        finalai[i] = ai[τ,k,i]
    end

    finalai, params
end

function tephase(C, params; θ=1e-8, kwargs...)
    ncells = size(C, 2)

    @info "Initial transfer entropy pass"

    te = zeros(ncells, ncells)
    indices = collect(product(1:ncells, 1:ncells))
    p = Progress(length(indices))
    @threads for (i, j) in indices
        if i != j
            #  others = [(@view C[:,k]) for k in setdiff(1:ncells, [i,j])]
            #  te[i,j] = @views transferentropy(C[:,i], C[:,j], others...; params[j]..., kwargs...)
            te[i,j] = @views transferentropy(C[:,i], C[:,j]; params[j]..., kwargs...)
            if te[i,j] < θ
                te[i,j] = zero(te[i,j])
            end
        end
        next!(p)
    end

    nperm, α = 100, 0.1
    @info "First refinement" nperm α
    refine!(transferentropy, te, C; nperm=nperm, α=α, kwargs...)

    nperm, α = 1000, 0.05
    @info "Final refinement" nperm α
    refine!(transferentropy, te, C; nperm=nperm, α=α, kwargs...)

    te
end

function main(dir; kwargs...)
    series = DataFrame(CSV.File(joinpath(dir, "series_trimmed.csv")))
    noise = Normal(0.0, 1e-8)
    series.c += rand(noise, nrow(series))
    ncells = length(unique(series.cellid))

    C = Array(transpose(reshape(series.c, ncells, nrow(series) ÷ ncells)))

    ai, params = aiphase(C)
    te = tephase(C, params; kwargs...)

    ai, te, params
end

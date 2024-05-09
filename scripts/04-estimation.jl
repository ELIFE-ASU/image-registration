using CSV, ClusterManagers, DataFrames, DelimitedFiles, Distributed, Distributions, DrWatson
addprocs(SlurmManager(160))
 
@everywhere begin
    using Pkg
    Pkg.activate(".")
end

include(srcdir("information.jl"))

@info "Packages Loaded"

const T1 = 33
const T2 = 33
const minperm = 100_000
const tenperm = 10_000

const datasets = vcat(readdir.(joinpath.(readdir(datadir(), join=true), "analysis"), join=true)...)

for dataset in datasets
    phase1dir = joinpath(dataset, "phase_1")
    phase2dir = joinpath(dataset, "phase_2")

    phase1_series = CSV.File(joinpath(phase1dir, "series_trimmed.csv")) |> DataFrame
    phase1_com = CSV.File(joinpath(phase1dir, "com_trimmed.csv")) |> DataFrame
    phase1_com[!,:phase] .= 1
    phase1_image = load(joinpath(phase2dir, "video.tif"))
    phase1_δ = transpose(Array(phase1_com[:,[:x,:y]])) |> distance

    phase2_series = CSV.File(joinpath(phase2dir, "series_trimmed.csv")) |> DataFrame
    phase2_com = CSV.File(joinpath(phase2dir, "com_trimmed.csv")) |> DataFrame
    phase2_com[!,:phase] .= 2
    phase2_image = load(joinpath(phase2dir, "video.tif"))
    phase2_δ = transpose(Array(phase2_com[:,[:x,:y]])) |> distance

    phase1_c = hcat([g.c for g in groupby(phase1_series, :cellid)]...)
    phase2_c = hcat([g.c for g in groupby(phase2_series, :cellid)]...)

    writedlm(joinpath(phase1dir, "series_noised.dat"), phase1_c)
    writedlm(joinpath(phase2dir, "series_noised.dat"), phase2_c)

    𝒩 = Normal(0, 1e-8)
    phase1_c .+= rand(𝒩, size(phase1_c)...)
    phase2_c .+= rand(𝒩, size(phase2_c)...)

    phase1_c = phase1_c[end+1-T1:end,:]
    phase2_c = phase2_c[1:T2,:]

    nn1, nn2 = findnn(phase1_c), findnn(phase2_c)

    @info "Processing" dataset phase1dir size(phase1_c) phase2dir size(phase2_c) nn1 nn2

    @info "Computing Mutual Information"

    @time phase1_mi_matrix = mi_matrix(phase1_c; nperm=minperm, nn=nn1)
    @time phase2_mi_matrix = mi_matrix(phase2_c; nperm=minperm, nn=nn2)
    
    mi = DataFrame(phase=Int[],
                   source=Int[], target=Int[],
                   source_x=Float64[], source_y=Float64[], target_x=Float64[], target_y=Float64[],
                   distance=Float64[],
                   value=Float64[], p=Float64[], nperm=Int[])
    for i in 1:size(phase1_mi_matrix, 1), j in i:size(phase1_mi_matrix, 2)
        source, source_x, source_y = phase1_com[i, [:cellid, :x, :y]]
        target, target_x, target_y = phase1_com[j, [:cellid, :x, :y]]
        δ = phase1_δ[i, j]
        sig = phase1_mi_matrix[i, j]
        push!(mi, [1, source, target, source_x, source_y, target_x, target_y, δ, sig.value, sig.p, sig.nperm])
    end
    for i in 1:size(phase2_mi_matrix, 1), j in i:size(phase2_mi_matrix, 2)
        source, source_x, source_y = phase2_com[i, [:cellid, :x, :y]]
        target, target_x, target_y = phase2_com[j, [:cellid, :x, :y]]
        δ = phase2_δ[i, j]
        sig = phase2_mi_matrix[i, j]
        push!(mi, [2, source, target, source_x, source_y, target_x, target_y, δ, sig.value, sig.p, sig.nperm])
    end
    
    let params = @dict T1 T2 minperm nn1 nn2
        fname = joinpath(dataset, savename("mi", params, "csv"))
        CSV.write(fname, mi)
    end
    
    @info "Computing Active Information"

    phase1_ks, phase1_τs = estimate_embedding(phase1_c)
    phase2_ks, phase2_τs = estimate_embedding(phase2_c)
    
    ai = DataFrame(phase=Int[],
                   source=Int[],
                   source_x=Float64[], source_y=Float64[],
                   k=Int[], τ=Int[],
                   value=Float64[])
    for i in 1:size(phase1_c, 2)
        source, source_x, source_y = phase1_com[i, [:cellid, :x, :y]]
        k = phase1_ks[i]
        τ = phase1_τs[i]
        value = activeinfo(phase1_c[:, i]; k, τ)
        push!(ai, [1, source, source_x, source_y, k, τ, value])
    end
    for i in 1:size(phase2_c, 2)
        source, source_x, source_y = phase2_com[i, [:cellid, :x, :y]]
        k = phase2_ks[i]
        τ = phase2_τs[i]
        value = activeinfo(phase2_c[:, i]; k, τ)
        push!(ai, [2, source, source_x, source_y, k, τ, value])
    end
    
    let params = @dict T1 T2
        fname = joinpath(dataset, savename("ai", params, "csv"))
        CSV.write(fname, ai)
    end
    
    @info "Computing Transfer Entropy"

    @time phase1_te_matrix, phase1_delay_matrix = te_matrix(phase1_c, phase1_ks, phase1_τs, 1:10; nperm=tenperm, nn=nn1)
    @time phase2_te_matrix, phase2_delay_matrix = te_matrix(phase2_c, phase2_ks, phase2_τs, 1:10; nperm=tenperm, nn=nn2)
    
    te = DataFrame(phase=Int[],
                   source=Int[], target=Int[],
                   source_x=Float64[], source_y=Float64[], target_x=Float64[], target_y=Float64[],
                   distance=Float64[],
                   k=Int[], τ=Int[], delay=Int[],
                   value=Float64[], p=Float64[], nperm=Int[])
    for i in 1:size(phase1_te_matrix, 1), j in 1:size(phase1_te_matrix, 2)
        source, source_x, source_y = phase1_com[i, [:cellid, :x, :y]]
        target, target_x, target_y = phase1_com[j, [:cellid, :x, :y]]
        δ = phase1_δ[i, j]
        sig = phase1_te_matrix[i, j]
        k = phase1_ks[j]
        τ = phase1_τs[j]
        delay = phase1_delay_matrix[i, j]
        push!(te, [1, source, target, source_x, source_y, target_x, target_y, δ, k, τ, delay, sig.value, sig.p, sig.nperm])
    end
    for i in 1:size(phase2_te_matrix, 1), j in 1:size(phase2_te_matrix, 2)
        source, source_x, source_y = phase2_com[i, [:cellid, :x, :y]]
        target, target_x, target_y = phase2_com[j, [:cellid, :x, :y]]
        δ = phase2_δ[i, j]
        sig = phase2_te_matrix[i, j]
        k = phase2_ks[j]
        τ = phase2_τs[j]
        delay = phase2_delay_matrix[i, j]
        push!(te, [2, source, target, source_x, source_y, target_x, target_y, δ, k, τ, delay, sig.value, sig.p, sig.nperm])
    end
    
    let params = @dict T1 T2 tenperm nn1 nn2
        fname = joinpath(dataset, savename("te", params, "csv"))
        CSV.write(fname, te)
    end
end

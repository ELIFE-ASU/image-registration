using CSV, DataFrames, DrWatson

const datasets = vcat(readdir.(joinpath.(readdir(datadir(), join=true), "analysis"), join=true)...)

const T = 33

for dataset in datasets
    phase1dir = joinpath(dataset, "phase_1")
    phase2dir = joinpath(dataset, "phase_2")

    phase1_series = CSV.File(joinpath(phase1dir, "series_trimmed.csv")) |> DataFrame
    phase2_series = CSV.File(joinpath(phase2dir, "series_trimmed.csv")) |> DataFrame

    phase1_c = hcat([g.c for g in groupby(phase1_series, :cellid)]...)
    phase2_c = hcat([g.c for g in groupby(phase2_series, :cellid)]...)

    phase1_c = phase1_c[end+1-T:end, :]
    phase2_c = phase2_c[1:T, :]

    @info dataset phase1_ncells=length(groupby(phase1_series, :cellid)) phase1=size(phase1_c) phase2=size(phase2_c)
end

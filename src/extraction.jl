using Base.Iterators, PyCall, ImageShow, ImageAxes

hnc = pyimport("hnccorr")
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

function segment(name, framepath; kwargs...)
    nframes = length(readdir(framepath))
    config = hnc.HNCcorrConfig(; kwargs...)
    h = hnc.HNCcorr.from_config(config)
    movie = hnc.Movie.from_tiff_images(name, framepath, nframes, false, 1)
    h.segment(movie)
    cells = h.segmentations_to_list()
    map(x -> map(y -> CartesianIndex(y .+ (1, 1)), first(values(x))), cells)
end

function moore()
    iter = product(-1:1, -1:1)
    sanszero = Iterators.filter(x -> !all(iszero.(x)), iter)
    Iterators.map(CartesianIndex, sanszero)
end

moore(pixel) = map(o -> pixel + o, moore())

function boundary(cell)
    Iterators.filter(cell) do pixel
        !all(n -> n in cell, moore(pixel))
    end
end

function marksegments(frame, cells)
    frame = RGB{N0f16}.(frame)
    for cell in cells, pixel in boundary(cell)
        c = frame[pixel]
        frame[pixel] = RGB{N0f16}(1., 0., 0.)
    end
    frame
end

function signal(frames, cells)
    T = size(frames, timedim(frames))
    timeseries = zeros(Float64, T, length(cells))
    for (i, cell) in enumerate(cells)
        for pixel in cell
            timeseries[:,i] += frames[pixel.I...,:]
        end
        timeseries[:,i] ./= length(cell)
    end
    timeseries
end

function celltable(cells::Vector{Vector{CartesianIndex{2}}})
    data = NamedTuple{(:id, :pixel, :x, :y), NTuple{4, Int}}[]
    for (id, cell) in enumerate(cells)
        for (pid, pixel) in enumerate(cell)
            push!(data, (; id, pixel=pid, x=pixel[1], y=pixel[2]))
        end
    end
    DataTable(data)
end

function seriestable(series::Matrix)
    ids = repeat(1:size(series,2), inner=size(series,1))
    timesteps = repeat(1:size(series,1), outer=size(series,2))
    values = vec(series)
    data = map(x -> (; id=x[1], timestep=x[2], value=x[3]), zip(ids, timesteps, values))
    DataTable(data)
end

function movingwindow(series::Vector; w=length(series)÷40)
    [mean(series[i:i+w]) for i in 1:length(series)-w]
end
function movingwindow(series::Matrix; kwargs...)
    mapslices(xs -> movingwindow(xs; kwargs...), series; dims=1)
end

function baselinesubtract(series; kwargs...)
    mw = movingwindow(series; kwargs...)
    series[1:size(mw,1),:] .- mw
end

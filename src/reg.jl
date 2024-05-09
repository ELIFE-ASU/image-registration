using ANTsRegistration
using Base.Threads
using CoordinateTransformations
using ImageAxes
using ImageTransformations
using Images
using Images.ImageFiltering
using LinearAlgebra
using MultivariateStats
using Printf
using ProgressMeter
using Rotations
using StaticArrays
using Statistics
using VideoIO

import ANTsRegistration: AbstractTransformation

"""
Load video frames from a video file or from a directory of images. The color
space of the resulting video can be specified with the `color` argument.
"""
function loadframes(path; color=Gray)
    frames = if isfile(path)
        VideoIO.load(path)
    else
        load.(readdir(path, join=true))
    end
    AxisArray(color.(cat(frames...; dims=3)), :x, :y, :time)
end

function saveframes(path::AbstractString, frames; progress=true, ext="tiff")
    N = size(frames, timedim(frames))
    progress && @info "Writing frames to $path"
    fnames = String[]
    meter = Progress(N; enabled=progress)
    for k in 1:N
        fname = joinpath(path, (@sprintf "%05d" k) * ".$ext")
        save(fname, frames[:,:,k])
        push!(fnames, fname)
        next!(meter)
    end
    fnames
end

Images.gray(img::AxisArray{<:AbstractGray}) = gray.(img)
Images.gray(img::AxisArray{<:Color}) = gray(Gray.(img))
Images.gray(T::Type, img) = map(T ∘ gray, img)
Images.gray(T::Type{<:Integer}, img) = map(p -> round(T, 255 * gray(p)), img)

"""
Compute the center of mass of the image.
"""
function com(img::AxisArray{T, 2}; k=2.0, stat=median, kwargs...) where {T <: Gray}
    N, M = size(img)
    θ = k * stat(img)

    totalmass = 0
    com = SVector{2,Float64}(0, 0)
    for j in 1:M, i in 1:N
        mass = Float64(gray(img[i, j] ≥ θ))
        com = com + SVector(i*mass, j*mass)
        totalmass += mass
    end
    com ./ totalmass
end

function evenrange(rng, min, max)
    if first(rng) != min
        first(rng)-1:last(rng)
    elseif last(rng) != max
        first(rng):last(rng)-1
    else
        first(rng)+1:last(rng)
    end
end

"""
Save a video (matrix of frames) to a video file.
"""
function video(path, frames, args...; progress=true, codec_name="libx264", encoder_options=(crf=23, preset=0), kwargs...)
    meter = Progress(size(frames,3); enabled=progress)
    open_video_out(path, frames[:,:,1]; codec_name, encoder_options, kwargs...) do writer
        for i in 1:size(frames, 3)
            write(writer, frames[:,:,i])
            next!(meter)
        end
    end
end

struct CropBox
    rows::StepRange{Int,Int}
    cols::StepRange{Int,Int}
end

function CropBox(frames; progress=true, k=2., σ=(100,100), border=20, makeeven=true, kwargs...)
    progress && @info "Cropping"
    meter = Progress(size(frames, timedim(frames)); enabled=progress)
    masks = similar(frames);
    @threads for k in 1:size(frames, timedim(frames))
        blur = gray(imfilter((@view frames[:,:,k]), Kernel.gaussian(σ)))
        θ = k * mean(blur)
        masks[:,:,k] = blur .≥ θ
        next!(meter)
    end

    mask = reduce((x,y) -> x || y, masks; dims=3, init=false)

    rs = vec(any(mask; dims=(2,3)))
    cs = vec(any(mask; dims=(1,3)))

    rows = max(1, findfirst(rs) - border):min(size(mask,1), findlast(rs) + border)
    cols = max(1, findfirst(cs) - border):min(size(mask,2), findlast(cs) + border)

    if makeeven
        isodd(length(rows)) && (rows = evenrange(rows, 1, size(mask, 1)))
        isodd(length(cols)) && (cols = evenrange(cols, 1, size(mask, 1)))
    end

    CropBox(rows, cols)
end

crop(box::CropBox, frames; kwargs...) = frames[box.rows, box.cols, :]
crop(frames; kwargs...) = crop(CropBox(frames; kwargs...), frames)

function majoraxis(frame::AxisArray{T}; k=2., σ=(2,2), stat=median, mask=false, kwargs...) where {T<:Colorant}
    blur = gray(imfilter(frame, Kernel.gaussian(σ)))
    θ = k * stat(blur)
    if mask
        frame[:,:] = T.(blur .≥ θ)
    end
    A = map(x -> [x.I...], findall(blur .≥ θ))
    if isempty(A)
        error("The image is probably too homogeneous, consider reducing k=$k")
    end
    B = reshape(vcat(A...), 2, length(A))
    model = fit(PCA, B)
    model.proj[:,1]
end

function Base.angle(frame::AxisArray{<:Colorant}; kwargs...)
    θ = angle(Complex(majoraxis(frame; kwargs...)...))
    θ ≤ zero(θ) && (θ += π)
    θ
end

function backgroundsubtract!(frames::AxisArray{T,3}; k=2., σ=(15,15), σₘ=(5,5), stat=median, progress=true, kwargs...) where {T <: Colorant}
    progress && @info "Background Subtraction"
    meter = Progress(size(frames, 3); enabled=progress)
    kernel = Kernel.gaussian(σ)
    blur = similar(frames[:,:,1])
    @threads for i in 1:size(frames, 3)
        frame = @view frames.data[:,:,i]
        imfilter!(blur, frame, kernel)
        θ = k * stat(blur)
        mask = imfilter(blur .≤ θ, Kernel.gaussian(σₘ))
        frame[:,:] = clamp!(frame - mask, zero(T), one(T))
        next!(meter)
    end
    frames
end

"""
Performs a pseudo-flat field correction on the provided video. First the frames
of the video are averaged to yield a representative image. Then a gaussian blur
is applied with a kernel width comparable to the size of the frame. The blurred
image is then subtracted from each frame of the video, clamping the values
between 0 and 1.
"""
function flatfield!(frames::AxisArray{T,3}; progress=true, kwargs...) where {T <: Colorant}
    progress && @info "Pseudo-Flat Field Correction"
    meter = Progress(size(frames, 3); enabled=progress)
    μ = reshape(mean(frames; dims=timedim(frames)), size_spatial(frames)...)
    blur = imfilter(μ, Kernel.gaussian(size_spatial(frames)))
    @threads for k in 1:size(frames, 3)
        frame = @view frames[:,:,k]
        frame[:,:] = clamp!(frame - blur, zero(T), one(T))
        next!(meter)
    end
    frames
end

"""
Regularize the orientation of the image. For each frame, the center of mass of
the image is computed and a translation is applied to center it. Then a PCA is
performed to estimate the major axis of the image, and a rotation is applied to
standardize the orientation. This is a very rough approximation of a rigid
registration.
"""
function regularize!(frames::AxisArray{T, 3}; progress=true, addmarker=false, kwargs...) where {T <: Colorant}
    progress && @info "Regularizing Orientation"
    meter = Progress(size(frames,3); enabled=progress)
    @threads for k in 1:size(frames,3)
        frame = @view frames[:,:,k]

        q = round.(Int, center(frame))
        r = round.(Int, com(frame; kwargs...))
        Δ = r - q

        ϕ = Translation(Δ...)

        frame[:,:] = warp(frame, ϕ, axes(frame))

        θ = angle(frame; kwargs...)

        if addmarker
            r = max(500, 0.25 * min(size(frame)...))
            x, y = round(Int, q[1] + r*cos(θ)), round(Int, q[2] + r*sin(θ))

            frame[x-5:x+5, y-5:y+5] .= Gray{N0f16}(1.)
        end

        ψ = recenter(RotMatrix(θ), q)

        frame[:,:] = warp(frame, ψ, axes(frame))

        next!(meter)
    end
    frames
end

function preprocess(frames; progress=true, regularize=false, cropvid=false, ffsubtract=false, bgsubtract=false, kwargs...)
    dst = deepcopy(frames)
    regularize && regularize!(dst; progress, kwargs...)
    ffsubtract && flatfield!(dst; progress, kwargs...)
    bgsubtract && backgroundsubtract!(dst; progress, kwargs...)
    if !bgsubtract && cropvid
        progress && @info "Background subtracting to facilitate cropping; the subtraction will not be present in the final result"
        precrop = deepcopy(dst)
        backgroundsubtract!(precrop; progress, kwargs...)
        box = CropBox(precrop; progress, kwargs...)
        crop(box, dst; progress, kwargs...)
    elseif cropvid
        crop(dst; progress, kwargs...)
    else
        dst
    end
end
preprocess(; kwargs...) = frames -> preprocess(frames; kwargs...)

function correctmotion(frames::AbstractArray{T,3}, stages::AbstractTransformation...; color=T, progress=true, kwargs...) where {T <: Colorant}
    frames = preprocess(frames; progress, kwargs...)
    progress && @info "Correcting Motion"
    mktempdir() do dir
        moving = joinpath(dir, "moving.nrrd")
        output = (joinpath(dir, "registered"), joinpath(dir, "registered.nrrd"))
        fixed = frames[:,:,1]
        if isempty(stages)
            stages = [Stage(fixed, Global("Rigid"), MI())]
        end
        save(moving, frames)
        motioncorr(output, fixed, moving, stages; verbose=false)

        registered = load(last(output))
        progress && @info "Registered image" pixelrange=extrema(registered)
        AxisArray(color.(registered ./ maximum(registered)), axisnames(frames)...)
    end
end

function correctmotion(stages::AbstractTransformation...; kwargs...)
    frames -> correctmotion(frames, stages...; kwargs...)
end

function correctorload(dir, frames; force=false, kwargs...)
    if !force && isdir(dir) && !isempty(dir)
        @info "Loading registered frames from $dir"
        loadframes(dir; color=Gray{N0f16})
    else
        correctmotion(frames; kwargs...)
    end
end
correctorload(dir; kwargs...) = frames -> correctorload(dir, frames; kwargs...)

using Compose, FileIO, Images, Printf, LinearAlgebra, Statistics

function info_graph(frame, com, info, overlays=[]; node_color="red", node_α=0.3, node_size=0.01, edge_color="orange", edge_α=0.1, edge_weight=0.1, heads=true)
    φ = 0.5 * (1 - sqrt(5))
    aspect = Compose.default_graphic_width / Compose.default_graphic_height
    io = IOBuffer()
    save(Stream(format"PNG", io), frame)
    framedata = take!(io);
    scale_factors = reverse(size(frame))
    scaled_com = transpose([com.y com.x]) ./ scale_factors
    
    df = filter(r -> r.source != r.target, info)
    
    min, max = extrema(df.value)
    lines = map(eachrow(df)) do row
        # start, stop = scaled_com[:, row.source], scaled_com[:, row.target]
        start = [row.source_y, row.source_x] ./ scale_factors
        stop = [row.target_y, row.target_x] ./scale_factors
        δ = node_size * (stop - start) / norm(stop - start)
        [start + δ, stop - δ]
    end
    arrows = map(eachrow(df)) do row
        # start, stop = scaled_com[:, row.source], scaled_com[:, row.target]
        start = [row.source_y, row.source_x] ./ scale_factors
        stop = [row.target_y, row.target_x] ./scale_factors
        δ = (stop - start) / norm(stop - start)
        u = [-δ[2], aspect * δ[1]]
        [stop - node_size*δ, stop - 2node_size*δ + node_size*u/4φ, stop - 2node_size*δ - node_size*u/4φ]
    end
    compose(context(),
        overlays...,
        (context(), stroke(edge_color), strokeopacity(edge_α), fill(edge_color), fillopacity(edge_α), linewidth(edge_weight),
            heads ? (context(), polygon(arrows)) : context(),
            (context(), line(lines))
        ),
        (context(),
            circle(scaled_com[1,:], scaled_com[2,:], fill(node_size, size(scaled_com,2))),
            fill(node_color), fillopacity(node_α), stroke(node_color), strokeopacity(clamp(2node_α, 0.0, 1.0))),
        (context(), bitmap(["image/png"], [framedata], [0], [0], [1], [1]))
    )
end

svg2png(svgfile; pngfile=splitext(svgfile)[1] * ".png") = (run(`inkscape -z -e $pngfile $svgfile`); pngfile)

# puncture = [(context(), star(0.52, 0.82, 0.015, 5), fill("blue"))]

function filter_graph(outdir, outname, frame, com, info, col; remove=false, overlays=[], text_pos=(hright, vbottom), font_color="white", font_size=20pt, kwargs...)
    if !ispath(joinpath(outdir, outname))
        mkpath(joinpath(outdir, outname))
    end
    
    for (i, group) in enumerate(groupby(info, col; sort=true))
        graphic = compose(context(),
            (context(), Compose.text(0.9, 0.9, string(col) * "= $(group[1,col])", text_pos...), fill(font_color), fontsize(font_size)),
            info_graph(frame, com, group, overlays; kwargs...))
        svgfile = joinpath(outdir, outname, (@sprintf "%03d.svg" i))
        save(svgfile, graphic)
        svg2png(svgfile)
        rm(svgfile)
    end
    
    let infiles=joinpath(outdir, outname, "%03d.png"), outfile=joinpath(outdir, "$outname.gif")
        run(`ffmpeg -y -framerate 1 -i $infiles $outfile`)
    end

    if remove
        rm(joinpath(outdir, outname); recursive=true)
    end
end

module Visuals
export show_obs_graph

using ..ObserverGraph
using ..Propagation

using GraphPlot
using Colors
using Graphs

function show_obs_graph(og::ObsGraph, loc_data::LocData; should_label_idx::Bool=false, source_pred::Int=1)
    V = nv(og.graph)
    node_colors = [colorant"white" for _ in 1:V]
    for i in 1:V
        if i in og.observer_set
            value = get(loc_data.obs_data, i, 0)
            green_intensity = (value - loc_data.t_start) / (maximum(values(loc_data.obs_data)) - loc_data.t_start)
            node_colors[i] = RGB(0, (green_intensity == NaN) ? 1 : 1 - green_intensity, 0)
        end
    end

    node_colors[loc_data.source] = colorant"red"
    node_colors[source_pred] = colorant"gold"

    gplot(og.graph,
        nodefillc=node_colors,
        nodestrokec=colorant"black",
        nodestrokelw=1,
        nodelabel=should_label_idx ? (1:V) : nothing,  # Node annotations
        nodelabelc=colorant"grey",
        # EDGELINEWIDTH=0.01,
        edgestrokec=colorant"grey",
        # nodelabelsize=2,
        # NODESIZE=0.03
    )
end
end
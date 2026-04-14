import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os 

with open("/gs/gsfs0/shared-lab/duong-lab/VLM_graphrag/artifacts/05_create_cooccurrence_graph/26237587/cooccurrence_graph.json") as f:
    graph = json.load(f)

prevalence = graph["label_prevalence"]
edges_raw  = graph["edges"]

G = nx.Graph()
G.add_nodes_from(prevalence.keys())

seen = set()
for src, targets in edges_raw.items():
    for tgt, w in targets:
        key = tuple(sorted([src, tgt]))
        if key not in seen and w > 0.05:   # threshold weak edges
            seen.add(key)
            G.add_edge(src, tgt, weight=w)

pos = nx.circular_layout(G)

def node_color(p):
    if p > 0.15:  return "#1D9E75"
    if p >= 0.03: return "#7F77DD"
    return "#D85A30"

node_sizes  = [prevalence[n] * 8000 for n in G.nodes()]
node_colors = [node_color(prevalence[n]) for n in G.nodes()]
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
edge_widths  = [w * 8 for w in edge_weights]
edge_alphas  = [min(1.0, 0.3 + w * 2) for w in edge_weights]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")
ax.axis("off")

for (u, v), lw, alpha in zip(G.edges(), edge_widths, edge_alphas):
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    ax.plot(x, y, color="#888780", linewidth=lw, alpha=alpha, zorder=1)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colors, ax=ax)

# Short labels for cramped nodes
label_map = {n: n.replace("Enlarged Cardiomediastinum", "Enl. Cardiomediastinum")
               .replace("Pleural Effusion", "Pl. Effusion")
               .replace("Support Devices", "Support Dev.")
             for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=label_map,
                        font_size=9, font_weight="bold", ax=ax)

legend = [
    mpatches.Patch(color="#1D9E75", label="Common (>15%)"),
    mpatches.Patch(color="#7F77DD", label="Moderate (3–15%)"),
    mpatches.Patch(color="#D85A30", label="Rare (<3%)"),
]
ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.8)
ax.set_title("Label co-occurrence graph (NPMI > 0.05)", fontsize=12, pad=12)

plt.tight_layout()
os.makedirs("/gs/gsfs0/shared-lab/duong-lab/VLM_graphrag/artifacts/graph", exist_ok=True)
plt.savefig("/gs/gsfs0/shared-lab/duong-lab/VLM_graphrag/artifacts/graph/cooccurrence_graph.pdf", bbox_inches="tight", dpi=300)
plt.savefig("/gs/gsfs0/shared-lab/duong-lab/VLM_graphrag/artifacts/graph/cooccurrence_graph.png", bbox_inches="tight", dpi=300)
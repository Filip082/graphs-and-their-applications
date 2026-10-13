"""Draw simple graphs with vertices evenly spaced on a circle."""

import matplotlib.pyplot as plt
import networkx as nx


def draw_circular(graph: nx.Graph, output: str | None = None) -> None:
  """Show the drawing in a window, or save it to `output` (format from the extension, e.g. .png, .svg)."""
  fig, ax = plt.subplots(figsize=(6, 6))
  nx.draw_circular(
    graph,
    ax=ax,
    with_labels=True,
    node_color="white",
    edgecolors="black",
    node_size=500,
  )
  ax.set_aspect("equal")
  ax.set_title(f"n = {graph.number_of_nodes()}, m = {graph.number_of_edges()}")
  if output:
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
  else:
    plt.show()

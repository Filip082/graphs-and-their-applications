"""Random simple graphs G(n, l) and G(n, p), with vertices numbered 1..n."""

import networkx as nx


def _check_n(n: int) -> None:
  if n < 1:
    raise ValueError("n must be at least 1")


def random_gnl(n: int, l: int, seed: int | None = None) -> nx.Graph:
  """Uniformly random graph on n vertices with exactly l edges."""
  _check_n(n)
  max_edges = n * (n - 1) // 2
  # networkx silently returns the complete graph when l > max_edges
  if not 0 <= l <= max_edges:
    raise ValueError(f"l must be in 0..{max_edges} for n = {n}")
  return nx.convert_node_labels_to_integers(nx.gnm_random_graph(n, l, seed=seed), first_label=1)


def random_gnp(n: int, p: float, seed: int | None = None) -> nx.Graph:
  """Random graph on n vertices where each of the n(n-1)/2 edges appears independently with probability p."""
  _check_n(n)
  # networkx silently treats p < 0 as 0 and p > 1 as 1
  if not 0 <= p <= 1:
    raise ValueError("p must be in [0, 1]")
  return nx.convert_node_labels_to_integers(nx.gnp_random_graph(n, p, seed=seed), first_label=1)

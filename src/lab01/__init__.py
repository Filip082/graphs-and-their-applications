from pathlib import Path

import networkx as nx

from lab01.encoding import AdjacencyList, AdjacencyMatrix, IncidenceMatrix
from lab01.random_graphs import random_gnl, random_gnp
from lab01.visualization import draw_circular

EXAMPLES = Path(__file__).parent
OUTPUT = Path("out")

def show(name: str, graph: nx.Graph) -> None:
  print(f"=== {name} ===")
  print("adjacency matrix:")
  print(AdjacencyMatrix.from_networkx(graph).matrix)
  print("incidence matrix:")
  print(IncidenceMatrix.from_networkx(graph).matrix)
  print("adjacency list:")
  for v, vs in enumerate(AdjacencyList.from_networkx(graph).neighbours):
    print(f"{v + 1}. {' '.join(str(u + 1) for u in vs)}")
  output = OUTPUT / f"{name}.png"
  draw_circular(graph, output)
  print(f"drawing saved to {output}\n")

def main():
  OUTPUT.mkdir(exist_ok=True)
  show("example", AdjacencyMatrix.from_file(EXAMPLES / "example.txt").to_networkx())
  show("example2", IncidenceMatrix.from_file(EXAMPLES / "example2.txt").to_networkx())
  show("example3", AdjacencyList.from_file(EXAMPLES / "example3.txt").to_networkx())
  show("random_gnl", random_gnl(8, 10))
  show("random_gnp", random_gnp(8, 0.4))

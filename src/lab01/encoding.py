import networkx as nx
import numpy as np

# nx.Graph vertices are numbered 1..n (as in random_graphs), classes store them 0-based

class AdjacencyMatrix:
  def __init__(self, data) -> None:
    self.matrix = np.array(data)

  @classmethod
  def from_file(cls, filename: str) -> AdjacencyMatrix:
    loaded = cls(np.loadtxt(filename, dtype=int))
    shape = loaded.matrix.shape
    if not len(shape) == 2 or not shape[0] == shape[1]:
      raise ValueError("adjacency matrix must be square")
    return loaded

  @classmethod
  def from_networkx(cls, graph: nx.Graph) -> AdjacencyMatrix:
    return cls(nx.to_numpy_array(graph, nodelist=sorted(graph.nodes), dtype=int))

  def to_networkx(self) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(1, self.matrix.shape[0] + 1))
    graph.add_edges_from((i + 1, j + 1) for i, j in zip(*np.nonzero(np.triu(self.matrix))))
    return graph

  def to_incidence_matrix(self) -> IncidenceMatrix:
    vertices = self.matrix.shape[0]
    temp = np.zeros(shape=(vertices, self.matrix.sum() // 2), dtype=int)
    index = 0
    for i in range(vertices):
      for j in range(i + 1, vertices):
        if self.matrix[i][j] == 1:
          temp[i, index] = 1
          temp[j, index] = 1
          index += 1
    return IncidenceMatrix(temp)

  def to_adjacency_list(self) -> AdjacencyList:
    vertices = self.matrix.shape[0]
    temp = [[] for _ in range(vertices)]
    for i in range(vertices):
      for j in range(i):
        if self.matrix[i][j] == 1:
          temp[i].append(j)
          temp[j].append(i)
    return AdjacencyList(temp)

class IncidenceMatrix:
  def __init__(self, data) -> None:
    self.matrix = np.array(data)

  @classmethod
  def from_file(cls, filename: str) -> IncidenceMatrix:
    loaded = cls(np.loadtxt(filename, dtype=int))
    if not ((loaded.matrix == 1) | (loaded.matrix == 0)).all() or not (loaded.matrix.sum(axis=0) == 2).all():
      raise ValueError("incidence matrix must have two ones in every column")
    return loaded

  @classmethod
  def from_networkx(cls, graph: nx.Graph) -> IncidenceMatrix:
    return cls(nx.incidence_matrix(graph, nodelist=sorted(graph.nodes)).toarray().astype(int))

  def to_networkx(self) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(1, self.matrix.shape[0] + 1))
    for e in self.matrix.T:
      u, v = np.flatnonzero(e)
      graph.add_edge(u + 1, v + 1)
    return graph

  def to_adjacency_matrix(self) -> AdjacencyMatrix:
    vertices = self.matrix.shape[0]
    temp = np.zeros(shape=(vertices, vertices), dtype=int)
    for e in self.matrix.T:
      u, v = np.flatnonzero(e)
      temp[u, v] = 1
      temp[v, u] = 1
    return AdjacencyMatrix(temp)

  def to_adjacency_list(self) -> AdjacencyList:
    return self.to_adjacency_matrix().to_adjacency_list()

class AdjacencyList:
  def __init__(self, data) -> None:
    self.neighbours = data

  @classmethod
  def from_file(cls, filename: str) -> AdjacencyList:
    # file format is 1-based: "v. n1 n2 ...", stored 0-based
    with open(filename) as file:
      lines = [line for line in file if line.strip()]
    neighbours = [[] for _ in lines]
    for line in lines:
      vertex, _, rest = line.partition(".")
      v = int(vertex) - 1
      if not 0 <= v < len(lines):
        raise ValueError(f"vertex {v + 1} out of range 1..{len(lines)}")
      neighbours[v] = [int(u) - 1 for u in rest.split()]
    for v, vs in enumerate(neighbours):
      for u in vs:
        if not 0 <= u < len(lines) or u == v:
          raise ValueError(f"invalid neighbour {u + 1} of vertex {v + 1}")
        if v not in neighbours[u]:
          raise ValueError(f"edge {v + 1}-{u + 1} missing in vertex {u + 1}")
    return cls(neighbours)

  @classmethod
  def from_networkx(cls, graph: nx.Graph) -> AdjacencyList:
    index = {v: i for i, v in enumerate(sorted(graph.nodes))}
    return cls([sorted(index[u] for u in graph[v]) for v in index])

  def to_networkx(self) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(1, len(self.neighbours) + 1))
    graph.add_edges_from((v + 1, u + 1) for v, vs in enumerate(self.neighbours) for u in vs)
    return graph

  def to_adjacency_matrix(self) -> AdjacencyMatrix:
    vertices = len(self.neighbours)
    temp = np.zeros(shape=(vertices, vertices), dtype=int)
    for v, vs in enumerate(self.neighbours):
      for u in vs:
        temp[v, u] = 1
        temp[u, v] = 1
    return AdjacencyMatrix(temp)

  def to_incidence_matrix(self) -> IncidenceMatrix:
    return self.to_adjacency_matrix().to_incidence_matrix()

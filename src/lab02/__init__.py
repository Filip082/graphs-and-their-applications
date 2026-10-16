

from pathlib import Path
from lab01.visualization import draw_circular
from lab02.check_sequence import AdjacencyMatrixlab02, read_sequence_from_file

EXAMPLES = Path(__file__).parent
OUTPUT = Path("out")

def check_sequence(name,sequence):
    print("---------")
    print(f"sequence: {sequence}")
    if not AdjacencyMatrixlab02.check_if_sequence_is_graphical(sequence):
        print("The sequence is not graphical")
        return
    adjacency_matrix = AdjacencyMatrixlab02.construct_from_sequence(sequence)
    print(adjacency_matrix.matrix)
    draw_circular(adjacency_matrix.to_networkx(), OUTPUT / f"{name}")
    print("---------")

def main():
    OUTPUT.mkdir(exist_ok=True)
    print("Zad 1")
    check_sequence("lab01_1", read_sequence_from_file(EXAMPLES / "example1.txt"))
    check_sequence("lab01_2", read_sequence_from_file(EXAMPLES / "example2.txt"))
    check_sequence("lab01_3", read_sequence_from_file(EXAMPLES / "example3.txt"))
    print("Zad 2")
    matrix = AdjacencyMatrixlab02.construct_from_sequence(read_sequence_from_file(EXAMPLES / "example1.txt"))
    print("Before randomization")
    print(matrix.matrix)
    matrix.randomize(3)
    print("After 3 randomization")
    print(matrix.matrix)
    draw_circular(matrix.to_networkx(), OUTPUT / "lab02_2")

if __name__ == "__main__":
    main()
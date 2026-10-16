from lab01.encoding import AdjacencyMatrix
import numpy as np

class AdjacencyMatrixlab02(AdjacencyMatrix):

    @classmethod
    def construct_from_sequence(cls, sequence : np.Array):
        if not cls.check_if_sequence_is_graphical(sequence):
            raise ValueError("The sequence is not graphical")
        sequence = np.sort(sequence, descending=True)
        new_matrix = cls(np.zeros((len(sequence), len(sequence)), dtype=int))
        help_sequence = np.zeros(len(sequence), dtype=int)
        for i in range(len(sequence)):
            if sequence[i] > help_sequence[i]:
                steps = sequence[i] - help_sequence[i]
                j = i + 1
                while steps > 0:
                    if help_sequence[j] < sequence[j]:
                        new_matrix.matrix[i][j] = 1
                        new_matrix.matrix[j][i] = 1
                        help_sequence[j] += 1
                        steps -= 1
                    j += 1
        return new_matrix


    @classmethod
    def check_if_sequence_is_graphical(cls, sequence : np.Array):
        test_sequence = np.sort(sequence, descending=True)
        if test_sequence[0] >= len(test_sequence):
            return False
        while test_sequence.size > 0 and test_sequence[0] != 0:
            element = test_sequence[0]
            for i in range(1, element + 1):
                test_sequence[i] -= 1
                if test_sequence[i] < 0:
                    return False
            test_sequence[0] = 0
            test_sequence = np.sort(test_sequence, descending=True)
        return True

    def randomize(self, amount_of_swaps : int):
        limit = 20
        incidence_matric = self.to_incidence_matrix()
        if incidence_matric.matrix.shape[1] < 2:
            return
        for i in range(limit):
            first_edge = np.random.randint(0, incidence_matric.matrix.shape[1])
            second_edge = np.random.randint(0, incidence_matric.matrix.shape[1])
            while first_edge == second_edge:
                second_edge = np.random.randint(0, incidence_matric.matrix.shape[1])
            first_edge_A, first_edge_B = np.where(incidence_matric.matrix[:, first_edge] == 1)[0]
            second_edge_C, second_edge_D = np.where(incidence_matric.matrix[:, second_edge] == 1)[0]
            if first_edge_A == second_edge_C or first_edge_A == second_edge_D or first_edge_B == second_edge_C or first_edge_B == second_edge_D:
                continue
            first_edge_to_create = np.zeros(incidence_matric.matrix.shape[0], dtype=int)
            first_edge_to_create[first_edge_A] = 1
            first_edge_to_create[second_edge_C] = 1
            first_edge_to_create = first_edge_to_create.reshape(-1,1)
            if np.any(np.all(incidence_matric.matrix == first_edge_to_create, axis=0)):
                continue
            second_edge_to_create = np.zeros(incidence_matric.matrix.shape[0], dtype=int)
            second_edge_to_create[first_edge_B] = 1
            second_edge_to_create[second_edge_D] = 1
            second_edge_to_create  = second_edge_to_create.reshape(-1,1)
            if np.any(np.all(incidence_matric.matrix == second_edge_to_create, axis=0)):
                continue
            incidence_matric.matrix[:, first_edge] = first_edge_to_create.flatten()
            incidence_matric.matrix[:, second_edge] = second_edge_to_create.flatten()
            amount_of_swaps -= 1
            if amount_of_swaps == 0:
                break
        self.matrix = incidence_matric.to_adjacency_matrix().matrix
            

def read_sequence_from_file(filename: str) -> np.Array:
    with open(filename, 'r') as f:
        sequence = [int(x) for x in f.read().split()]
    return np.array(sequence)
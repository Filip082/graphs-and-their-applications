from typing import List

def change_from_adjacency_matrix_to_incidence_matrix(matrix : List[List[int]]) -> List[List[int]]:
    edges = sum(sum(row) for row in matrix) // 2
    incidence_matrix = [[0 for _ in range(edges)] for _ in range(len(matrix))]
    index = 0
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] == 1:
                incidence_matrix[i][index] = 1
                incidence_matrix[j][index] = 1
                index += 1
    return incidence_matrix

def change_from_incidence_matrix_to_adjacency_matrix(matrix : List[List[int]]) -> List[List[int]]:
    adjacency_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for j in range(len(matrix[0])):
        x,y = -1,-1
        for i in range(len(matrix)):
            if matrix[i][j] == 1:
                if x == -1:
                    x = i
                else:
                    y = i
        adjacency_matrix[x][y] = 1
        adjacency_matrix[y][x] = 1
    return adjacency_matrix

def change_from_adjacency_list_to_adjacency_matrix(adjacency_list : List[List[int]]) -> List[List[int]]:
    adjacency_matrix = [[0 for _ in range(len(adjacency_list))] for _ in range(len(adjacency_list))]
    for i in range(len(adjacency_list)):
        for j in adjacency_list[i]:
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
    return adjacency_matrix

def change_from_adjacency_matrix_to_adjacency_list(matrix : List[List[int]]) -> List[List[int]]:
    adjacency_list = [[] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(i):
            if matrix[i][j] == 1:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
    return adjacency_list

def change_from_incidence_matrix_to_adjacency_list(matrix : List[List[int]]) -> List[List[int]]:
    return change_from_adjacency_matrix_to_adjacency_list(change_from_incidence_matrix_to_adjacency_matrix(matrix))

def change_from_adjacency_list_to_incidence_matrix(adjacency_list : List[List[int]]) -> List[List[int]]:
    return change_from_adjacency_matrix_to_incidence_matrix(change_from_adjacency_list_to_adjacency_matrix(adjacency_list))
import lab1.test_data as test_data
import lab1.conversion as conversion


print("Adjacency List:")
for row in test_data.adjacency_list:
    print(row)
print("After conversion to adjacency matrix:")
for row in conversion.change_from_adjacency_list_to_adjacency_matrix(test_data.adjacency_list):
    print(row)
print()

print("Adjacency Matrix:")
for row in test_data.adjacency_matrix:
    print(row)
print("After conversion to adjacency list:")
for row in conversion.change_from_adjacency_matrix_to_adjacency_list(test_data.adjacency_matrix):
    print(row)
print()

print("Incidence Matrix:")
for row in test_data.incidence_matrix:
    print(row)
print("After conversion to adjacency matrix:")
for row in conversion.change_from_incidence_matrix_to_adjacency_matrix(test_data.incidence_matrix):
    print(row)
print()

print("Adjacency Matrix:")
for row in test_data.adjacency_matrix:
    print(row)
print("After conversion to incidence matrix:")
for row in conversion.change_from_adjacency_matrix_to_incidence_matrix(test_data.adjacency_matrix):
    print(row)
print()

print("Incidence Matrix:")
for row in test_data.incidence_matrix:
    print(row)
print("After conversion to adjacency list:")
for row in conversion.change_from_incidence_matrix_to_adjacency_list(test_data.incidence_matrix):
    print(row)
print()

print("Adjacency List:")
for row in test_data.adjacency_list:
    print(row)
print("After conversion to incidence matrix:")
for row in conversion.change_from_adjacency_list_to_incidence_matrix(test_data.adjacency_list):
    print(row)
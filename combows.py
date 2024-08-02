from itertools import combinations

def find_combinations(values, k):
    return list(combinations(values, k))

# Example usage:
values = [1, 2, 3, 4]
k = 2
combinations_list = find_combinations(values, k)
print(combinations_list)
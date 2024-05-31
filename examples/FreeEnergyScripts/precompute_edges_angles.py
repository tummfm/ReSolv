import numpy as np

mol = [[1,2,3], [1,4,3],...., [3,2,6]]

def return_max_edges(mol):
    overall_count = 0
    for i in range(len(mol)):
        temp_count = 0
        for j in range(len(mol)):
            dist = np.linalg.norm(mol[i] - mol[j])
            # check cutoff
            if dist < 0.5 and i!=j:
                temp_count += 1
        overall_count = max(overall_count, temp_count)
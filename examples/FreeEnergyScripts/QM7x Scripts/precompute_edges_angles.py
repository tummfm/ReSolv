import numpy as np
import itertools
import time

start = time.process_time()

load_energies = np.load("QM7x_DB/atom_energies_QM7x.npy")
all_size = load_energies.shape[0]
pad_species = np.load("QM7x_DB/atom_numbers_QM7x.npy")[:all_size]
pad_pos = np.load("QM7x_DB/atom_positions_QM7x.npy")[:all_size]
pad_pos *= 0.1

data = [pad_pos[index][:np.count_nonzero(pad_species[index])] for index in np.arange(0, len(pad_pos))]
compute = True

def return_max_edges_and_angles(mol):
    edge_count = 0
    angle_count = 0
    for i in range(len(mol)):
        temp_count = 0
        nbr_index=[]
        for j in range(len(mol)):
            dist = np.linalg.norm(mol[i] - mol[j])
            # check cutoff
            if dist < 0.5 and i!=j:
                temp_count += 1
                nbr_index.append(j)
        edge_count += temp_count                                    #max(edge_count, temp_count)
        angle = len(list((itertools.combinations(nbr_index,2))))
        # print(list((itertools.combinations(nbr_index,2))))
        # print(angle)
        angle_count += angle        #max(angle_count,angle)
    return edge_count,angle_count*2
if compute:
    max_edge = 0
    max_angle = 0
    counter = 0
    for mol in data:
        edge_count, angle_count = return_max_edges_and_angles(mol)
        if counter == 0:
            max_angle = angle_count
            max_edge = edge_count
            counter += 1
            continue
        if angle_count > max_angle:
            max_angle = angle_count
        if edge_count > max_edge:
            max_edge = edge_count
        counter += 1
        print(counter)

    duration = (time.process_time()-start)/60
    print(f'Maximum number of Edges: {max_edge}')
    print(f'Maximum number of Angles: {max_angle}')
    print(f'Took {duration} mins')
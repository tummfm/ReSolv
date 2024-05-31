import os

visible_device = ""
os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)

import numpy as onp
import chemtrain.amber_utils_qm7x as au

def correct_atomic_number():
    numbers = onp.load('QM7x_DB/shuffled_atom_numbers_QM7x.npy', allow_pickle=True)
    ids = onp.load('QM7x_DB/shuffled_atom_molecule_QM7x.npy', allow_pickle=True)
    positions = onp.load('QM7x_DB/shuffled_atom_positions_QM7x.npy', allow_pickle=True)

    count = 0
    for id in ids:
        #print(count)
        if id == '2066':
            count += 1
            continue
        ## Species as list of stringed integers from top file
        species = au.amber_prmtop_load(f"PRMTOPfiles/mol_{int(id)}.prmtop")['ATOMIC_NUMBER']
        ## UNPACK QM7X SPECIES
        qm7x_species = numbers[count]
        qm7x_pos = positions[count]
        unpacked_species = []
        unpacked_pos = []
        counter = 0
        for index in qm7x_species:
            if index != 0:
                unpacked_species.append(str(index))
                unpacked_pos.append(qm7x_pos[counter])
            counter += 1
        ## CHECK IF THEY ARE IN THE SAME ORDER
        if unpacked_species != species:
            #CORRECT THE ORDER TO MATCH THE TOP FILE
            new_index_list = []
            for item in species:
                index = unpacked_species.index(item)
                new_index_list.append(index)
           # new_qm7x_species = [unpacked_species[index] for index in new_index_list]    ##  (unpadded)
            new_qm7x_pos = [unpacked_pos[index] for index in new_index_list]
            # if new_qm7x_species != species:
            #     print('NOT EQUAL : ID' , id)

            # TODO - 1. New position array with corrected indices.
            # TODO - 2. New species array with corrected indices.
            # TODO . 3. New force array with corrected indices.


        count += 1

if __name__ == '__main__':
    correct_atomic_number()
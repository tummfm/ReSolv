import anton_scripts.anton_findData as fd
import jax.numpy as jnp
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_data(index_list, no_confs, exclusion_list=None, data_type=None):
    FE_values = []
    calc = []
    errors = []
    ## Loading FreeSolv data
    freesolv_dict = fd.get_FreeSolv_data()

    ## Selecting molecules of N-H-O-C-Cl-S composition
    relevant_molecules_dict = fd.get_relevant_molecules_2(freesolv_dict)

    ## Forming dictionary of relevant molecules with experimental Hydration free energy values + errors
    solvation_dictionary = {}
    for mol in relevant_molecules_dict.keys():
        solvation_dictionary[relevant_molecules_dict[mol]['smiles']] = {'expSolvFreeEnergy': relevant_molecules_dict[mol]['expt'], # free energy units in kcal/mol
                                                           'expUncertainty': relevant_molecules_dict[mol]['d_expt'], 'MOL' : mol , 'calc': relevant_molecules_dict[mol]['calc'], 'd_calc': relevant_molecules_dict[mol]['d_calc']}
    ## Creating list of SMILES
    smile_list = []
    MOL_list = []
    for smile in solvation_dictionary.keys():
        smile_list.append(smile)
        MOL_list.append(solvation_dictionary[smile]['MOL'])

    ## Creating data list to pass to network
    data = []
    smiles = []
    c = 0
    for k in range(len(smile_list)):
        ## Selecting subset for efficiency
        if data_type == 'train':
            if k not in index_list:
                continue
        elif data_type == 'test':
            if k in index_list:
                continue
        elif data_type == 'all':
            pass

        if exclusion_list is not None:
            if k in exclusion_list:
                continue

        # if k in [352,457,471]:     #(this means Mol 353 and 458 are bad i.e. cannot generate stable trajs)
        #     continue

        # Just something I tried and seemed to work
        if k in (147, 294):
            no_confs = 3
        else:
            no_confs = 1

        ## Embedding molecule for 3d cartesian coords
        mol = Chem.MolFromSmiles(smile_list[k])
        molecule = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(molecule, numConfs=no_confs)
        for cid in cids: AllChem.MMFFOptimizeMolecule(molecule, confId=cid)

        species = []
        mass = []
        r_init = []

        for i, atom in enumerate(molecule.GetAtoms()):

            if atom.GetSymbol() == 'H':
                mass.append(1.00784)
                species.append(1)
            if atom.GetSymbol() == 'C':
                mass.append(12.011)
                species.append(6)
            if atom.GetSymbol() == 'N':
                mass.append(14.007)
                species.append(7)
            if atom.GetSymbol() == 'O':
                mass.append(15.999)
                species.append(8)
            if atom.GetSymbol() == 'S':
                mass.append(32.06)
                species.append(16)
            if atom.GetSymbol() == 'Cl':
                mass.append(35.45)
                species.append(17)
        for n in range(no_confs):
            conf_pos = []
            for i, atom in enumerate(molecule.GetAtoms()):
                positions = molecule.GetConformer(n).GetAtomPosition(i)
                conf_pos.append([positions.x,positions.y,positions.z])
            r_init.append(conf_pos)

        if no_confs > 1:
            r_init = jnp.array(r_init) + 500
        else:
            r_init = r_init[0]
            r_init = jnp.array(r_init) + 500
        # if no_confs larger than 1 only use one configuration
        if no_confs > 1:
            r_init = r_init[1]

        smiles.append(smile_list[k])
        mass = jnp.array(mass)
        species = jnp.array(species)
        calc.append(solvation_dictionary[smile_list[k]]['calc'])
        FE_values.append(solvation_dictionary[smile_list[k]]['expSolvFreeEnergy'])
        errors.append(solvation_dictionary[smile_list[k]]['expUncertainty'])
        target_dict = {'free_energy_difference' : solvation_dictionary[smile_list[k]]['expSolvFreeEnergy']}
        data.append([r_init, mass, species, target_dict,smile_list[k],f'mol_{k+1}_AC'])
        c+=1
    return data, FE_values, smiles, calc, errors


def generate_position(smile, no_confs):
    mol = Chem.MolFromSmiles(smile)
    molecule = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(molecule, numConfs=no_confs)
    for cid in cids: AllChem.MMFFOptimizeMolecule(molecule, confId=cid)
    molecule.GetConformer()

    r_init = []

    for n in range(no_confs):
        conf_pos = []
        for i, atom in enumerate(molecule.GetAtoms()):
            positions = molecule.GetConformer(n).GetAtomPosition(i)
            conf_pos.append([positions.x, positions.y, positions.z])
        r_init.append(conf_pos)

    if no_confs > 1:
        r_init = jnp.array(r_init) + 500
    else:
        r_init = r_init[0]
        r_init = jnp.array(r_init) + 500
    return r_init

def get_404_list():
    _404 = [4, 23, 24, 34, 52, 55, 56, 65, 93, 112, 119, 131, 132, 137, 138, 143, 147, 149, 162, 168, 169, 184, 185,
            189, 193, 202, 231, 242, 245, 270, 272, 274, 276, 278, 279, 295, 322, 326, 327, 329, 331, 335, 392, 393,
            415, 424,
            425, 426, 1, 3, 6, 13, 22, 48, 60, 66, 77, 82, 83, 87, 105, 133, 134, 139, 148, 177, 178, 190, 206, 209,
            224, 235,
            236, 247, 252, 255, 265, 283, 284, 291, 310, 333, 334, 347, 355, 387, 394, 400, 405, 407, 409, 418, 21, 103,
            156, 176,
            188, 260, 275, 319, 342, 352, 359, 364, 19, 113, 165, 210, 246, 244, 9, 11, 12, 18, 20, 25, 26, 31, 33, 35,
            36,
            38, 39, 41, 43, 44, 47, 49, 50, 51, 57, 58, 62, 63, 64, 67, 69, 72, 73, 74, 75, 86, 90, 91, 92, 94, 96, 97,
            98, 100,
            101, 107, 108, 111, 116, 117, 120, 122, 124, 125, 126, 129, 136, 142, 146, 150, 152, 157, 158, 160, 161,
            163, 166, 174,
            175, 182, 186, 187, 191, 195, 198, 205, 211, 212, 214, 216, 218, 221, 223, 226, 227, 230, 232, 233, 238,
            239, 243, 248, 249, 250, 251, 254, 259, 262, 264, 266, 271, 285, 286, 288, 290, 297, 301, 304, 305, 307,
            309, 312, 313, 315,
            320, 324, 325, 328, 330, 336, 337, 338, 341, 345, 346, 351, 354, 358, 360, 362, 363, 368, 369, 370, 374,
            377, 378, 380, 381, 383, 385, 388, 391, 396, 8, 42, 53, 59, 61, 71, 79, 81, 95, 99, 102, 106, 110, 118, 121,
            123, 141,
            144, 153, 154, 170, 180, 194, 197, 199, 201, 225, 228, 240, 253, 258, 263, 268, 273, 294, 298, 303, 314,
            321,
            349, 356, 375, 382, 398, 401, 406, 427, 453, 0, 17, 30, 32, 37, 54, 80, 84, 128, 155, 213, 217, 219, 256,
            267, 282,
            293, 311, 28, 167, 296, 366, 29, 171, 350, 384, 14, 277, 413, 2, 5, 7, 10, 15, 16, 27, 40, 45, 46, 68, 70,
            76, 78, 88,
            89, 104, 109, 114, 115, 127, 130, 135, 140, 145, 151, 159, 164, 172, 173, 179, 181, 183, 192, 196, 200, 203,
            204, 207, 208, 215, 220, 222, 229, 234, 237, 241, 261, 269, 280, 281, 287, 289, 292, 300, 302, 306, 308,
            316, 317,
            318, 323, 332, 339, 343, 344, 348, 353, 361, 365, 371, 373, 376, 379, 386, 390, 395, 397]

    return _404

def get_train_dataset():
    train_dataset = [4, 23, 24, 34, 52, 55, 56, 65, 93, 112, 119, 131, 132, 137, 138, 143, 147, 149, 162, 168, 169, 184, 185, 189, 193, 202,
                     231, 242, 245, 270, 272, 274, 276, 278, 279, 295, 322, 326, 327, 329, 331, 335, 392, 393, 415, 424, 425, 426, 1, 3, 6,
                     13, 22, 48, 60, 66, 77, 82, 83, 87, 105, 133, 134, 139, 148, 177, 178, 190, 206, 209, 224, 235, 236, 247, 252, 255, 265,
                     283, 284, 291, 310, 333, 334, 347, 355, 387, 394, 400, 405, 407, 409, 418, 21, 103, 156, 176, 188, 260, 275, 319, 342, 352,
                     359, 364, 19, 113, 165, 210, 246, 244, 9, 11, 12, 18, 20, 25, 26, 31, 33, 35, 36, 38, 39, 41, 43, 44, 47, 49, 50, 51, 57,
                     58, 62, 63, 64, 67, 69, 72, 73, 74, 75, 86, 90, 91, 92, 94, 96, 97, 98, 100, 101, 107, 108, 111, 116, 117, 120, 122, 124,
                     125, 126, 129, 136, 142, 146, 150, 152, 157, 158, 160, 161, 163, 166, 174, 175, 182, 186, 187, 191, 195, 198, 205, 211, 212,
                     214, 216, 218, 221, 223, 226, 227, 230, 232, 233, 238, 239, 243, 248, 249, 250, 251, 254, 259, 262, 264, 266, 271, 285, 286,
                     288, 290, 297, 301, 304, 305, 307, 309, 312, 313, 315, 320, 324, 325, 328, 330, 336, 337, 338, 341, 345, 346, 351, 354, 358,
                     360, 362, 363, 368, 369, 370, 374, 377, 378, 380, 381, 383, 385, 388, 391, 396, 8, 42, 53, 59, 61, 71, 79, 81, 95, 99, 102,
                     106, 110, 118, 121, 123, 141, 144, 153, 154, 170, 180, 194, 197, 199, 201, 225, 228, 240, 253, 258, 263, 268, 273, 294, 298,
                     303, 314, 321, 349, 356, 375, 382, 398, 401, 406, 427, 453, 0, 17, 30, 32, 37, 54, 80, 84, 128, 155, 213, 217, 219, 256, 267,
                     282, 293, 311, 28, 167, 296, 366, 29, 171, 350, 384, 14, 277, 413, 2, 5, 7, 10, 15, 16, 27, 40, 45, 46, 68, 70, 76, 78, 88,
                     89, 104, 109, 114, 115, 127, 130, 135, 140, 145, 151, 159, 164, 172, 173, 179, 181, 183, 192, 196, 200, 203, 204, 207, 208,
                     215, 220, 222, 229, 234, 237, 241, 261, 269, 280, 281, 287, 289, 292, 300, 302, 306, 308, 316, 317, 318, 323, 332, 339, 343,
                     344, 348, 353, 361, 365, 371, 373, 376, 379, 386, 390, 395, 397, 85, 372, 420, 422, 447, 448, 466, 471, 481, 490, 494]

    return train_dataset


def get_389_train_dataset():
    train_dataset_389 = [4, 24, 25, 35, 53, 56, 58, 68, 74, 100, 122, 130, 143, 144, 151, 152, 158, 162, 164, 174, 176,
                         181, 187, 188, 205, 206,
                         211, 216, 225, 255, 266, 269, 297, 299, 301, 303, 305, 306, 322, 350, 355, 356, 358, 360, 364,
                         1, 3, 6, 13, 18, 23, 49,
                         63, 69, 81, 88, 89, 93, 106, 114, 145, 147, 150, 153, 163, 196, 197, 212, 229, 233, 248, 259,
                         260, 272, 277, 280, 290,
                         310, 311, 318, 337, 362, 363, 366, 377, 385, 22, 112, 173, 195, 210, 285, 302, 347, 372, 382,
                         390, 20, 123, 184, 234,
                         268, 9, 11, 12, 19, 21, 26, 27, 32, 34, 36, 37, 39, 40, 42, 44, 45, 48, 50, 51, 52, 57, 59, 60,
                         61, 65, 66, 67, 70, 72,
                         76, 77, 78, 79, 86, 92, 97, 98, 99, 102, 104, 105, 107, 109, 110, 116, 117, 120, 121, 127, 128,
                         131, 133, 135, 136, 137,
                         140, 141, 146, 149, 156, 157, 161, 165, 168, 169, 175, 177, 179, 180, 182, 185, 193, 194, 203,
                         208, 209, 213, 214, 218,
                         221, 228, 235, 236, 238, 240, 242, 245, 247, 250, 251, 254, 256, 257, 262, 263, 267, 270, 273,
                         274, 275, 276, 279, 284,
                         287, 289, 291, 292, 298, 312, 313, 315, 317, 324, 328, 331, 332, 334, 336, 339, 340, 341, 343,
                         348, 352, 353, 354, 357,
                         359, 365, 367, 368, 371, 375, 376, 381, 384, 388, 8, 43, 54, 62, 64, 75, 83, 84, 87, 95, 101,
                         103, 108, 111, 115, 119,
                         129, 132, 134, 155, 159, 170, 171, 189, 201, 217, 220, 222, 224, 249, 252, 264, 278, 283, 288,
                         295, 300, 321, 325, 330,
                         342, 349, 379, 386, 407, 414, 0, 17, 31, 33, 38, 55, 85, 90, 139, 172, 237, 241, 243, 281, 293,
                         294, 29, 186, 323, 30,
                         166, 190, 198, 207, 14, 304, 91, 2, 5, 7, 10, 15, 16, 28, 41, 46, 47, 71, 73, 80, 82, 94, 96,
                         113, 118, 124, 125, 138,
                         142, 148, 154, 160, 167, 178, 183, 191, 192, 199, 200, 202, 204, 215, 219, 223, 226, 227, 230,
                         231, 232, 239, 244, 246,
                         253, 258, 261, 265, 286, 296, 307, 308, 314, 316, 319, 327, 329, 333, 335, 344, 345, 346, 351,
                         361, 369, 373, 374, 378,
                         383, 392, 396, 380, 397, 404, 465, 513]
    return train_dataset_389
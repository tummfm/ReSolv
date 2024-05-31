import os
import multiprocessing
import signal

os.environ['CUDA_VISIBLE_DEVICES'] = str()

import numpy as onp
import jax.numpy as jnp
import datetime

from chemtrain.sparse_graph import get_max_edges_max_triplets


# Choose setup
load_dataset_name = '../../../../FreeEnergySolubility/examples/FreeEnergyScripts/ANI1xDB/'
model = "ANI-1x"
all_size = 2    # 4956005
# all_size = 100000

# Load data
energies = onp.load(load_dataset_name+'energies.npy')[100000:100000+all_size]
pad_pos = onp.load(load_dataset_name+'pad_pos.npy')[100000:100000+all_size]
pad_species = onp.load(load_dataset_name + 'mask_species.npy')[100000:100000+all_size]

# # Shuffle the data
# shuffle_indices = onp.arange(energies.shape[0])
# # onp.random.shuffle(shuffle_indices)
# pad_pos = pad_pos[shuffle_indices]
# pad_species = pad_species[shuffle_indices]

# TODO: Change species dtype to int32 from uint8 as current code checks for i32 || i64 else stops. Add uin8 there.
pad_species = onp.array(pad_species, dtype='int32')
pad_pos *= 0.1

# Take padded positions.
#  1. Add to non zero values 50nm
#  2. Set padded pos at x=0.6, 1.2, 1.8, .. [nm], y = z = 0. -> Energy contribution is zero

for i, pos in enumerate(pad_pos):
    pad_pos[i][:onp.count_nonzero(pad_species[i])] += onp.array([50., 50., 50.])

for i, pos in enumerate(pad_pos):
    x_spacing = 0
    for j in range(pad_species[i].shape[0] - onp.count_nonzero(pad_species[i])):
        x_spacing += 0.6
        pad_pos[i][onp.count_nonzero(pad_species[i]) + j][0] = x_spacing
print("Done")

# Remove padding, as padding for some reason causes parallel execution to stuck at sample 39543 and others.
# Sequentially it didn't.
pad_pos = [pad_pos[i][:onp.count_nonzero(pad_species[i])] for i in range(len(pad_pos))]

# Create 100nm^3 box
box = jnp.eye(3)*100

# compute max edges and angles to make training faster
max_edges = None
max_angles = None

def max_edges_and_angles_for_subset(pad_pos_subset, box, i, data_size):
    max_edges, max_angles = get_max_edges_max_triplets(r_cutoff=0.5, position_data=pad_pos_subset, box=box)
    result = onp.array([max_edges, max_angles])
    # print("result: ", result)
    onp.save('MaxEdgeAngleResults/datasize_'+str(data_size)+'_proc_'+str(i)+'.npy', result)
    return 0


# Need to do multiprocessing over batches of maybe 10000, for > 100.000 computation gets stuck for some reason.
sequential = True
parallel = False
parallel_and_sequential = False

if parallel_and_sequential:

    split_len = 256
    split = [i for i in range(0, len(pad_pos), split_len)]
    pad_pos_all = [pad_pos[i:i + split_len] for i in split]


    def handler(signum, frame):
        print("Parallel Execution timed out. Switch to sequential.")
        for k in range(len(processes)):
            processes[k].terminate()

        raise Exception
    signal.signal(signal.SIGALRM, handler)

    # Read in all max edges and angles and return overall max
    max_edge_overall = 0
    max_triplet_overall = 0
    for count, pad_pos in enumerate(pad_pos_all):

        signal.alarm(40)

        try:  # in parallel
            print("Starting with parallel")
            time1 = datetime.datetime.now()
            # Set parameters defining length and implicitly also number of CPUs to be used
            num_processors = 64
            # total_len = pad_pos.shape[0]
            # process_len = int(onp.ceil(pad_pos.shape[0] / num_processors))
            total_len = len(pad_pos)
            process_len = int(onp.ceil(len(pad_pos) / num_processors))

            # Create data lists with correct size for each processor
            data_num_list = [i for i in range(0, total_len, process_len)]

            # Avoid out of index error.
            if len(data_num_list) < num_processors:
                num_processors = len(data_num_list)

            pad_pos_parallel = [pad_pos[i:i + process_len] for i in data_num_list]

            # Start multiple processors defined by num_processors and assign correct data to processor.
            processes = []
            for i in range(num_processors):
                p = multiprocessing.Process(target=max_edges_and_angles_for_subset,
                                            args=(pad_pos_parallel[i], box, i, total_len))
                p.start()
                processes.append(p)

            # Wait for processes before 'Done' is printed. Without following two lines 'Done' would be printed right away
            # and not wait for processes.
            for process in processes:
                process.join()

            for i in range(num_processors):
                temp_edge_triplet = onp.load(
                    'MaxEdgeAngleResults/datasize_' + str(total_len) + '_proc_' + str(i) + '.npy')
                max_edge_overall = max([max_edge_overall, temp_edge_triplet[0]])
                max_triplet_overall = max([max_triplet_overall, temp_edge_triplet[1]])
                # print("Max temp edge: ", temp_edge_triplet[0])
                # print("Max temp triplet: ", temp_edge_triplet[1])
                os.remove('MaxEdgeAngleResults/datasize_' + str(total_len) + '_proc_' + str(i) + '.npy')
            print("Max edge {} / {}: {}".format(count+1, len(pad_pos_all), max_edge_overall))
            print("Max triplets {} / {}: {}".format(count+1, len(pad_pos_all), max_triplet_overall))
            max_edge_triplet = onp.array([max_edge_overall, max_triplet_overall])
            print("Ending with parallel")
            time2 = datetime.datetime.now()
            print("Time it took: ", time2 - time1)

        except:  # sequential
            print("Starting with sequential")
            max_edges, max_triplets = get_max_edges_max_triplets(r_cutoff=0.5, position_data=pad_pos, box=box)
            max_edge_overall = max([max_edge_overall, max_edges])
            max_triplet_overall = max([max_triplet_overall, max_triplets])
            print("Sequential - Max edge {} / {}: {}".format(count+1, len(pad_pos_all), max_edge_overall))
            print("Sequential - Max triplets {} / {}: {}".format(count+1, len(pad_pos_all), max_triplet_overall))
            print("Ending with sequential")
    print("For {} datasize, max edges: {}".format(all_size, max_edge_overall))
    print("For {} datasize, max triplets: {}".format(all_size, max_triplet_overall))
    final_max = onp.array([max_edge_overall, max_triplet_overall])
    onp.save('MaxEdgeAngleResults/max_edges_triplets_datasize_' + str(all_size) + '.npy', final_max)


if parallel:
    split_len = 256
    split = [i for i in range(0, len(pad_pos), split_len)]
    # split = [i for i in range(0, pad_pos.shape[0], split_len)]
    pad_pos_all = [pad_pos[i:i + split_len] for i in split]

    for count, pad_pos in enumerate(pad_pos_all):
        time1 = datetime.datetime.now()
        print("Start parallel loop")
        # Set parameters defining length and implicitly also number of CPUs to be used
        num_processors = 64
        # total_len = pad_pos.shape[0]
        # process_len = int(onp.ceil(pad_pos.shape[0] / num_processors))
        total_len = len(pad_pos)
        process_len = int(onp.ceil(len(pad_pos) / num_processors))

        # Create data lists with correct size for each processor
        data_num_list = [i for i in range(0, total_len, process_len)]
        pad_pos = [pad_pos[i:i + process_len] for i in data_num_list]

        # Start multiple processors defined by num_processors and assign correct data to processor.
        processes = []
        for i in range(num_processors):
            p = multiprocessing.Process(target=max_edges_and_angles_for_subset, args=(pad_pos[i], box, i, total_len))
            p.start()
            processes.append(p)

        # Wait for processes before 'Done' is printed. Without following two lines 'Done' would be printed right away
        # and not wait for processes.
        print("Join the processes.")
        for process in processes:
            process.join()

        print("Done joining processes.")
        # Read in all max edges and angles and return overall max
        max_edge_overall = 0
        max_triplet_overall = 0
        for i in range(num_processors):
            temp_edge_triplet = onp.load('MaxEdgeAngleResults/datasize_' + str(total_len) + '_proc_' + str(i) + '.npy')
            max_edge_overall = max([max_edge_overall, temp_edge_triplet[0]])
            max_triplet_overall = max([max_triplet_overall, temp_edge_triplet[1]])
            # print("Max temp edge: ", temp_edge_triplet[0])
            # print("Max temp triplet: ", temp_edge_triplet[1])
            os.remove('MaxEdgeAngleResults/datasize_' + str(total_len) + '_proc_' + str(i) + '.npy')
        max_edge_triplet = onp.array([max_edge_overall, max_triplet_overall])
        onp.save('MaxEdgeAngleResults/max_edges_triplets_datasize_' + str(total_len) + '_numprocs_' + str(num_processors) +'_num_subsample_' + str(count) + '.npy', max_edge_triplet)
        print('{} / {} max edges: {}, max triplets: {}'.format(count+1, len(pad_pos_all), max_edge_overall,
                                                              max_triplet_overall))
        time2 = datetime.datetime.now()
        print("Time for parallel execution in [s]: ", time2.second - time1.second)
        # print("Overall max edge: ", max_edge_overall)
        # print("Overall max triplets: ", max_triplet_overall)

    # Iterate over all saved sub max edges, triplets to get overall
    final_max_edge = 0
    final_max_triplets = 0
    for count in range(len(pad_pos_all)):
        temp_max = onp.load('MaxEdgeAngleResults/max_edges_triplets_datasize_' + str(total_len) + '_numprocs_' + str(num_processors) +'_num_subsample_' + str(count) + '.npy')
        final_max_edge = max([final_max_edge, temp_max[0]])
        final_max_triplets = max([final_max_triplets, temp_max[1]])
        os.remove('MaxEdgeAngleResults/max_edges_triplets_datasize_' + str(total_len) + '_numprocs_' + str(num_processors) +'_num_subsample_' + str(count) + '.npy')
    print("Final max edges: ", final_max_edge)
    print("Final max triplets: ", final_max_triplets)

    final_max = onp.array([final_max_edge, final_max_triplets])
    onp.save('MaxEdgeAngleResults/max_edges_triplets_datasize_' + str(total_len) + '_numprocs_' + str(num_processors) + '.npy', final_max)

if sequential:
    print("Switched to sequential")
    time1 = datetime.datetime.now()
    max_edges, max_angles = get_max_edges_max_triplets(r_cutoff=0.5, position_data=pad_pos, box=box)
    time2 = datetime.datetime.now()
    print("Time it took: ", time2.second - time1.second)
    result = onp.array([max_edges, max_angles])
    print(result)
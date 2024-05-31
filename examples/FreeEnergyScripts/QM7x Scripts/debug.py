import os
import sys

if len(sys.argv) > 1:
    visible_device = str(sys.argv[1])
else:
    visible_device = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.80'


# Avoid error in jax 0.4.25
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"


import jax
import jax.numpy as jnp
import numpy as onp
from jax import lax
import subprocess
import GPUtil

def get_number(input):
    return input

@jax.jit
def transform_array(input):
    # Create a mask for zeros and ones
    a = get_number(input)
    b = a < 2.0
    b = jnp.where(a < 2.0, True, False)
    return b


def get_gpu_stats():
    try:
     result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
     gpu_stats = result.stdout.strip().split('\n')
     return gpu_stats[1]
    except subprocess.CalledProcessError as e:
     print(f"Error: {e}")


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

list_404 = get_404_list()
list_404_sorted = onp.sort(list_404)
print(list_404_sorted)

# memory_used = []
# print(transform_array(0.0))
# memory_used.append(get_gpu_stats())
# print(transform_array(1.0))
# memory_used.append(get_gpu_stats())
# print(transform_array(2.0))
# memory_used.append(get_gpu_stats())
# print("Memory used: ", memory_used)

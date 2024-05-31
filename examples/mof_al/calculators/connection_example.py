from multiprocessing.connection import Client

import json

if "c" not in locals():
    c = Client("./mofq-socket", authkey=b'test')
    c.send('{ "cmd" : "get_data", "data_id" : "*",'  # ["ZUXPOZ_clean"],'
           '"properties" : ["id", "struc_numbers", "struc_positions",'
           ' "struc_cell", "partial_charges"] }')
    dataset = json.loads(c.recv())
else:
    raise RuntimeError('Could not open Client.')

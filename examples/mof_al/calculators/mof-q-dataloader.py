import os
from datetime import datetime
import shlex

import logging
logger = logging.getLogger()
logger.setLevel("DEBUG")

import json
import json.decoder

import multiprocessing as mp
from multiprocessing.connection import Listener
from dataclasses import dataclass, field

import numpy as np
import ase
import ase.io

def parse_simple_cif2ase(path):
    cdict = {}
    with open(path) as cif:
        loop_no = 0
        inloop = False
        loop_dict = {}
        for l in cif:
            l = l.strip()
            if l.startswith("#"):
                continue
            elif l.startswith("loop_"):
                if inloop:
                    cdict[f"loop_{loop_no}"] = loop_dict
                else:
                    inloop = True
                loop_no += 1
                loop_dict = {}
            elif inloop:
                if l.startswith("_"):
                    loop_dict[l] = []
                else:
                    for k, v in zip(loop_dict.keys(), shlex.split(l)):
                        loop_dict[k].append(v)
            elif l.startswith("_"):
                k, v = shlex.split(l)
                cdict[k[1:]] = v
            else:
                continue
        if loop_dict:
            cdict[f"loop_{loop_no}"] = loop_dict
    if cdict["symmetry_space_group_name_H-M"] != "P 1":
        logging.debug("SG "+str(cdict)+str(path))
        raise NotImplementedError("SG other than P 1 not implemented")
    if cdict["loop_1"]["_symmetry_equiv_pos_as_xyz"][0] != "x, y, z":
        logging.debug("posnot "+str(cdict)+str(path))
        raise NotImplementedError("only equiv symmetry positions implemented")
    a, b, c = [float(cdict[v]) for v in ("cell_length_a", "cell_length_b", "cell_length_c")]
    al, be, y = [float(cdict[v]) for v in ("cell_angle_alpha", "cell_angle_beta", "cell_angle_gamma")]
    positions = np.array([
        [float(c) for c in cdict["loop_2"]["_atom_site_fract_x"]],
        [float(c) for c in cdict["loop_2"]["_atom_site_fract_y"]],
        [float(c) for c in cdict["loop_2"]["_atom_site_fract_z"]]
    ]).T
    s = ase.Atoms(
        symbols = cdict["loop_2"]["_atom_site_type_symbol"],
        scaled_positions=positions,
        cell = [a, b, c, al, be, y],
        charges = [float(c) for c in cdict["loop_2"]["_atom_site_charge"]]
        )
    return s

@dataclass
class StrucMeta:
    id : str = None
    structure : ase.Atoms = None
    properties : dict = field(default_factory=dict)
    calc_date : int = 0 # set to 

    def __getattr__(self, name):
        if name == "struc_numbers":
            return self.structure.numbers.tolist()
        if name == "struc_positions":
            return self.structure.positions.tolist()
        elif name == "struc_cell" :
            return self.structure.cell.tolist()
        elif name in self.properties:
            return self.properties[name]
        else:
            return None

def calc_coremof_data(structure, q, properties=""):
    # TODO: implement calculation
    time.sleep(100)
    q.put(str(datetime.now()))

def load_coremof_data(dbpath):
    # TODO:
    # first load all data (including the CSD things...)
    # then load the
    # FOR NOW:
    # load the charge-assigned-database
    all_structures = {}
    charged_cifs = os.path.join(dbpath, "filtered_10x", "MOFs_cifs_data_ddec_charges")  # TODO set directory here
    
    if os.path.exists(charged_cifs):
        all_cifs = filter(lambda fn: fn.endswith("_DDEC.cif"), os.listdir(charged_cifs))
        names = map(lambda fn: fn.replace("_DDEC.cif", ""), os.listdir(charged_cifs))
        for _id, fn in zip(names, all_cifs):
            s = parse_simple_cif2ase(os.path.join(charged_cifs, fn))
            all_structures[_id] = StrucMeta(
                id = _id,
                structure = s,
                properties = dict(
                    partial_charges = s.get_initial_charges().tolist()
                ),
                calc_date = -1
            )
    else:
        raise FileNotFoundError(f"the DDEC-charge-assigned DB should be found in {charged_cifs}")

    return all_structures
    

# this is crude, probably you'd need something like Twisted
class DataServer():
    def __init__(self, dbpath, calcdir=None,
                 dataload_func=load_coremof_data,
                 socketpath="mofq-socket", authkey=b"test"):
        self.socketpath = socketpath
        self.authkey = authkey
        self.dbpath = dbpath
        self._datafunc = dataload_func
        logging.info(f"Starting to load data in {self.dbpath} (might take a while)")
        self._datapoints : Dict[str, StrucMeta] = self._datafunc(self.dbpath)

        self.successful_calcs = []
        self.active_calcs = []
        self.calc_results = mp.Queue()

    def handle_get_data(self, req):
        try:
            which = req["data_id"]
            if which == "*":
                req_data = self._datapoints.values()
            elif isinstance(which, list):
                req_data = [self._datapoints[_id] for _id in which]
            else:
                raise NotImplementedError
            properties = req["properties"]
            output_props = dict()
            for p in properties:
                output_props[p] = list(getattr(e, p) for e in req_data)
            resp = {"status" : "OK",
                    "properties" : output_props}
        except Exception as e:
            logging.debug(str("couldn't find property: ")+str(e))
            resp = { "status" : "ERROR",
                     "msg" : "couldn't get data: id or property not known" }
        return resp

    def handle_req_calc(self, req):
        try:
            which = req["data_id"]
            properties = reg["properties"]
        except:
            resp = { "status" : "ERROR"}
        return resp

    def handle_calc_status(self, req):
        pass

    def handle_request(self, req):
        if req["cmd"] == "get_data":
            resp = self.handle_get_data(req)
        elif req["cmd"] == "req_calc":
            resp = self.handle_req_calc(req)
        elif req["cmd"] == "calc_status":
            resp = self.handle_calc_status(req)
        else:
            resp = { "status" : 0 }
        return resp

        
    def serve_data(self):
        ## setup the calculation manager
        # TODO
        ## setup the interface
        logging.info(f"Starting to serve_data from {self.dbpath}")
        listener = Listener(self.socketpath, "AF_UNIX", authkey=self.authkey)
        conn = None
        while True:
            ## check the running calculations
            # TODO: implement
            ## handle outside requests
            try:
                if conn is None:
                    conn = listener.accept()
                elif conn.closed:
                    conn = listener.accept()
                # get some request
                message_raw = conn.recv()
                # parse the request
                message_parsed = json.loads(message_raw)
                if not isinstance(message_parsed, dict) and not \
                   hasattr(message_parsed, "cmd"):
                    raise ConnectionError
                if message_parsed["cmd"] == "close":
                    raise ConnectionResetError
                elif message_parsed["cmd"] == "stop":
                    logging.info("... STOPPING ...")
                    break
                else:
                    resp = self.handle_request(message_parsed)
                    response = json.dumps(resp)
                    conn.send(response)
            except (ConnectionError, json.decoder.JSONDecodeError,) as e:
                logging.info("RESTARTING: bad request")
                logging.debug(e)
                resp = { "status" : "FATAL" }
                response = json.dumps(resp)
                conn.send(response)
                conn.close()
                conn = None
            except (AttributeError, EOFError, mp.context.AuthenticationError) as e:
                logging.info("RESTARTING: Received invalid request or client dropped")
                logging.info(e)
                if isinstance(e, EOFError):
                    conn.close()
                conn = None
            except KeyboardInterrupt:
                logging.info("EXITING...")
                listener.close()
                break
            except Exception as e:
                logging.info("CRASHED: ", e)
                listener.close()
                break


if True and __name__ == "__main__":
    pass
    # TODO: better argparsing
    s = DataServer("../databases/coremof-q/coremof-2019")
    s.serve_data()
    # 
    #s_self = parse_simple_cif2ase("ZUXPOZ_clean_DDEC.cif")
    #s_ase = ase.io.read("ZUXPOZ_clean_DDEC.cif")    
    #print(np.all(s_self.positions - s_ase.positions < 1e-14),
    #      np.all( s_self.cell - s_ase.cell < 1e-15))

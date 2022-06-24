import ase.io
import pickle
import sys
import time
import numpy as np
#geoms = ase.io.read("./data/OE62.db")
#geoms = ase.io.read("generated_molecules.db",":100")
geoms=ase.io.read("./models/model1/generated/generated_molecules_11.db",":100")
sys.path.append('./GSchNetOE62')
from GSchNetOE62 import utility_functions
from utility_functions import print_atom_bond_ring_stats
from utility_classes import Molecule
even = 0
odd = 0
index = 0
#predictions = np.load('predictions.npz')
molecule_to_keep = []
#filtered_predictions = []
#import os, psutil
#process = psutil.Process(os.getpid())
#print(len(predictions['eigenvalues_pbe0'][0]))
time.sleep(5)
for geom in geoms:
    #os = geom.get_positions()
    atypes = geom.get_atomic_numbers()
    if(sum(atypes)%2 == 0):
        even +=1
        molecule_to_keep.append(geom)
        #filtered_predictions.append(predictions['eigenvalues_pbe0'][index])
    else:
        odd +=1
    index +=1
    print(index)
    #print(len(filtered_predictions))
    #print(len(filtered_predictions[-1]))
    #print(process.memory_info().rss)  # in bytes
print(molecule_to_keep)
#print(len(molecule_to_keep))
#print(even)
#filtered_predictions = predictions['eigenvalues_pbe0'][molecule_to_keep]
#print(len(filtered_predictions))
#np.savez_compressed('filtered_predictions2.npz',eigenvalues_pbe0 = filtered_predictions) 
ase.io.write("FullGenerated.db", molecule_to_keep)
#ase.io.write("generated_molecules_no_rad.db", molecule_to_keep)


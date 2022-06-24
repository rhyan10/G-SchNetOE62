import numpy as np
import statistics
import ase.io
import ase
import ase.io.xyz
import argparse
import subprocess
import ase.io
import pickle
import sys
import shutil
import time
sys.path.append('../G-SchNetOE62')
import utility_functions
from utility_functions import print_atom_bond_ring_stats
from ase import neighborlist
from ase.build import molecule
import numpy as np
import os
from scipy import sparse
from analysis import MoleculeAnalysis
from analysis import SchNetHAnalysis
import matplotlib.pyplot as plt
size=22
params={'legend.fontsize': 'large',
       'figure.figsize': (7,5),
        'axes.labelsize':size,
        'axes.titlesize':size,
        'ytick.labelsize':size,
        'xtick.labelsize':size,
        'axes.titlepad':10
       }
plt.rcParams.update(params)
#fig=plt.figure()



if __name__ == "__main__":
    bond_length_pairs = [['C','C'],['C','O'],['C','H']]
    bond_angle_trios = [['C','C','C'],['C','O','C'],['C','C','O']]
    
    number_of_loops = 20
    generated_database_size = 200000
    datapath = "./data/"
    database_size = 13635
    for i in range(7,number_of_loops):
        if i!=7:
            train = subprocess.Popen(['python ../G-SchNetOE62/gschnet_script.py train gschnet '+datapath+' ./models/iteration'+str(i)+'/ --pretrained_path ./models/iteration'+str(i-1)+'/  --dataset_name template_data --split '+str(round(database_size*0.8))+' '+str(round(database_size*0.1))+' --cuda --draw_random_samples 10 --batch_size 1 --max_epochs 5000'],shell=True)
            train.wait()
            generate = subprocess.Popen(['python ../G-SchNetOE62/gschnet_script.py generate gschnet ./models/iteration'+str(i)+'/ '+str(generated_database_size)+' --cuda'],shell=True)
            generate.wait()
            filter_ = subprocess.Popen(['python ../G-SchNetOE62/template_filter_generated.py ./models/iteration'+str(i)+'/generated/generated.mol_dict'],shell=True)
            filter_.wait()
            path = os.path.join("./models/iteration"+str(i)+"/", "analysis")
            os.mkdir(path)
        
        geoms = ase.io.read('./models/iteration'+str(i)+'/generated/generated_molecules.db',':')
        number_of_molecules = MoleculeAnalysis.get_molecule_sizes(geoms)
        ring_data = MoleculeAnalysis.get_rings(geoms)
        with open('./models/iteration'+str(i)+'/analysis/number_of_molecules.pkl', 'wb') as f:
            pickle.dump(number_of_molecules, f)
        with open('./models/iteration'+str(i)+'/analysis/rings.pkl', 'wb') as f:
            pickle.dump(ring_data, f)

        for bond in bond_length_pairs:
            bond_lengths = MoleculeAnalysis.get_bond_distances(geoms,bond)
            with open('./models/iteration'+str(i)+'/analysis/'+(bond[0]+bond[1])+'.pkl', 'wb') as f:
                pickle.dump(bond_lengths, f)

        for bond_angle in bond_angle_trios:
            bond_lengths = MoleculeAnalysis.get_angles(geoms,bond_angle)
            with open('./models/iteration'+str(i)+'/analysis/'+(bond_angle[0]+bond_angle[1]+bond_angle[2])+'.pkl', 'wb') as f:
                pickle.dump(bond_lengths, f)
        orbital_energies_prediction1  =  subprocess.Popen(['python /home/chem/mssdjc/software/SchNarc/src/scripts/run_schnet_ev.py pred ./models/iteration'+str(i)+'/generated/generated_molecules.db ./Models/PBE0_1 --parallel --batch_size 1 --cuda'],shell=True)
        orbital_energies_prediction1.wait()
        orbital_energies_prediction2  =  subprocess.Popen(['python /home/chem/mssdjc/software/SchNarc/src/scripts/run_schnet_ev.py pred ./models/iteration'+str(i)+'/generated/generated_molecules.db ./Models/PBE0_2 --parallel --batch_size 1 --cuda'],shell=True)
        orbital_energies_prediction2.wait()
        quasi_energies_prediction  =  subprocess.Popen(['python /home/chem/mssdjc/software/SchNarc/src/scripts/run_schnet_ev.py pred ./models/iteration'+str(i)+'/generated/generated_molecules.db ./Models/Delta --parallel --batch_size 1 --cuda'],shell=True)
        quasi_energies_prediction.wait()
        path = os.path.join("./models/iteration"+str(i)+"/", "energy_predictions")
        os.mkdir(path)
        shutil.move("./Models/PBE0_1/predictions.npz", "./models/iteration"+str(i)+"/energy_predictions/PBE01_predictions.npz")
        shutil.move("./Models/PBE0_2/predictions.npz", "./models/iteration"+str(i)+"/energy_predictions/PBE02_predictions.npz")
        shutil.move("./Models/Delta/predictions.npz", "./models/iteration"+str(i)+"/energy_predictions/Delta_predictions.npz")
        dbname1="./models/iteration"+str(i)+"/energy_predictions/PBE01_predictions.npz"
        dbname2="./models/iteration"+str(i)+"/energy_predictions/PBE02_predictions.npz"
        dbname3="./models/iteration"+str(i)+"/energy_predictions/Delta_predictions.npz"
        pbe0 = np.load(dbname1,allow_pickle=True)["eigenvalues_pbe0"]
        pbe0_2 =np.load(dbname2,allow_pickle=True)["eigenvalues_pbe0"]
        delta = np.load(dbname3,allow_pickle=True)["delta_eigenvalues_pbe0_gbw"]
        sorted_gw,geoms = SchNetHAnalysis.energy_analysis(pbe0,pbe0_2,delta,geoms)
        HOMO = sorted_gw[:,50].reshape(-1)
        LUMO = sorted_gw[:,51].reshape(-1)
        HL=np.abs(HOMO-LUMO)
        with open('./models/iteration'+str(i)+'/energy_predictions/HOMO_energies.pkl', 'wb') as f:
            pickle.dump(HOMO, f)
        with open('./models/iteration'+str(i)+'/energy_predictions/LUMO_energies.pkl', 'wb') as f:
            pickle.dump(LUMO, f)
        ####### Edit below for HOMO or HLGAP #######
        std = np.std(HL)
        mean = np.mean(HL)
        new_db = []
        for j, energy in enumerate(HL):
            if energy < mean - std:
                new_db.append(geoms[j])
        os.system("mv ./data/train.db ./data/train%i.db"%(i-1))
        ase.io.write("./data/train.db", new_db)

        os.remove("./data/train_gschnet.db")
        database_size = len(new_db)
        print(database_size)
        ############################################

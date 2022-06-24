import ase.io
import pickle
import sys
import time
sys.path.append('./GSchNetOE62')
from ase import neighborlist
from utility_classes import Molecule
import numpy as np

class MoleculeAnalysis():
    @staticmethod 
    def get_neighbours(geoms,element):
        dist=[]
        nneighbours = []
        ntype=[]
        #g=[]
        for mol in geoms:
            # we add 1 A to the natural cutoffs defined in ase
            found_geom = False
            found_geom2= False
            cutOff = np.array(neighborlist.natural_cutoffs(mol))
            nl = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
            nl.update(mol)
            distances = mol.get_all_distances()
            # iterate over all atoms in a molecule
            atomtypes = mol.get_chemical_symbols()
            n_neighbor = 0
            for iatom_index, iatom_type in enumerate(atomtypes):
                # ask if atom is O
                nO=0
                if iatom_type[0] == element:
                    nO+=1
                    n_neighbor=0
                    # get neighbors
                    neighborlist_mol = nl.get_neighbors(iatom_index)
                    n_neighbor+=len(neighborlist_mol[0])
                    for neighbor in neighborlist_mol[0]:
                        # avoid double counting
                        if neighbor < iatom_index:
                            pass
                        else:
                            # Do your routine here
                            # Ask if atom neighbor is a
                            dist.append(distances[iatom_index][neighbor])
                            ntype.append(atomtypes[neighbor])
                    nneighbours.append(n_neighbor)
                    
        #ase.io.write("outlier.db",g)
        return ntype,nneighbours,dist
    @staticmethod 

    def get_ntype(db):
        natom = {}
        available_atom_types = [1, 3, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53]
        for iatom in available_atom_types:
            natom[iatom]=[]
        for mol in db:
            atom = {}
            for iatom in available_atom_types:
                atom[iatom]=0
            atypes=mol.get_atomic_numbers()
            natoms = len(atypes)
            for iatom in atypes:
                atom[iatom]+=1
            
            for iatom in available_atom_types:
                natom[iatom].append(atom[iatom]/natoms)
        return natom
    
    @staticmethod
    def get_molecule_sizes(geoms):
        """
                Finds sizes of molecules for whole database
        """
        molecule_sizes = []
        for geom in geoms:
            #print(geom)
            #print(type(geom))
            distances = geom.get_positions()
            molecule_sizes.append(len(distances))
        return molecule_sizes
        
    @staticmethod
    def get_bond_distances(geoms,elements):
        """
            Finds bond lengths between two elements for all molecules in the database
            :return: List of bond lengths for a particular atom type
        """
        dist = []
        selectmolec=[]
        selectmolec2=[]
        for mol in geoms:
            # we add 1 A to the natural cutoffs defined in ase
            found_geom = False
            found_geom2= False
            cutOff = np.array(neighborlist.natural_cutoffs(mol)) + 1
            nl = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
            nl.update(mol)
            distances = mol.get_all_distances()
            # iterate over all atoms in a molecule
            atomtypes = mol.get_chemical_symbols()
            for iatom_index, iatom_type in enumerate(atomtypes):
                # ask if atom is C
                if iatom_type[0] == elements[0]:
                    # get neighbors
                    neighborlist_mol = nl.get_neighbors(iatom_index)
                    for neighbor in neighborlist_mol[0]:
                        # avoid double counting
                        if neighbor < iatom_index:
                            pass
                        else:
                            # Do your routine here
                            # Ask if atom neighbor is a
                            if atomtypes[neighbor] == elements[1]:
                                dist.append(distances[iatom_index][neighbor])
                                """if distances[iatom_index][neighbor] < 1.55 and distances[iatom_index][neighbor]  > 1.5:
                                    if found_geom == False:
                                        found_geom = True
                                        selectmolec.append(mol)
                                if distances[iatom_index][neighbor]<1.35:
                                    if found_geom2 == False:
                                        found_geom2 = True
                                        selectmolec2.append(mol)"""

        #import ase.io
        #ase.io.write("CCmolecs.db",selectmolec)
        #ase.io.write("COmolecs.db",selectmolec2)
        return dist

    @staticmethod
    def get_angles(geoms,elements):
        """
                Finds bond angle between three consecutive atoms for all molecules in the database
                :return: List of angles
        """
        angles = []
        for mol in geoms:
            # we add 0.5 A to the natural cutoffs defined in ase
            cutOff = np.array(neighborlist.natural_cutoffs(mol))+1
            nl = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
            nl.update(mol)
            # iterate over all atoms in a molecule
            atomtypes = mol.get_chemical_symbols()
            for iatom_index,iatom_type in enumerate(atomtypes):
                # ask if atom is C
                if iatom_type[0] == elements[0]:
                    #get neighbors
                    neighborlist_mol=nl.get_neighbors(iatom_index)
                    for neighbor in neighborlist_mol[0]:
                        #avoid double counting
                        if neighbor < iatom_index:
                            pass
                        else:
                            #Do your routine here
                            #Ask if atom neighbor is a
                            if atomtypes[neighbor] == elements[1]:
                                for neighbor2 in neighborlist_mol[0]:
                                    if neighbor==neighbor2:
                                        pass
                                    else:
                                        if atomtypes[neighbor2] == elements[2]:
                                            angles.append(mol.get_angle(neighbor, iatom_index, neighbor2))
                                        else:
                                            pass
        return angles

    @staticmethod
    def get_rings(geoms):
        rings = {}
        length_of_database = 0
        for geom in geoms:
            pos = geom.get_positions()
            atypes = geom.get_atomic_numbers()
            mol = Molecule(pos, atypes, store_positions=False)
            ring_counts = mol.get_ring_counts()
            #Only considers molecules of size more than 7
            if len(pos) > 7:
                length_of_database += 1
                for i in ring_counts:
                    if str(i) in rings.keys():
                        rings[str(i)] = rings[str(i)] + 1
                    else:
                        rings[str(i)] = 1
        for key in rings.keys():
            rings[key] = rings[key] / length_of_database
        return rings


class SchNetHAnalysis():

    @staticmethod
    def energy_analysis(pred,pred2,delta,geoms):
        shift = 13.70494
        gw = pred + delta - shift
        # also sort geometries
        geomsnew = []
        # let's consider HOMO, HOMO-1, HOMO-2, and LUMO for sorting
        # We used a mask for training to consider only values down to PBE0 eigenvalues -10
        
        MAE = np.mean(np.abs((pred[:,48:51]-pred2[:,48:51])))
    
        sorted_gw = []

        for i in range(len(gw)):
            if np.mean(np.abs(pred[i,48:51]-pred2[i,48:51]))>2*MAE or len(geoms[i])<=2:
                pass
            else:
                sorted_gw.append(gw[i])
                geomsnew.append(geoms[i])
        sorted_gw = np.array(sorted_gw)

        return sorted_gw,geomsnew
    def energy_analysis_indices(pred,pred2,delta,geoms):
        shift = 13.70494
        gw = pred + delta - shift
        # also sort geometries
        geomsnew = []
        # let's consider HOMO, HOMO-1, HOMO-2, and LUMO for sorting
        # We used a mask for training to consider only values down to PBE0 eigenvalues -10
        
        MAE = np.mean(np.abs((pred[:,48:51]-pred2[:,48:51])))
    
        sorted_gw = []
        indices=[]
        for i in range(len(gw)):
            if np.mean(np.abs(pred[i,48:51]-pred2[i,48:51]))>2*MAE or len(geoms[i])<=2:
                pass
            else:
                sorted_gw.append(gw[i])
                geomsnew.append(geoms[i])
                indices.append(i)
        sorted_gw = np.array(sorted_gw)

        return sorted_gw,geomsnew,indices

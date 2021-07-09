import ase.io
import ase
import numpy as np
import ase.neighborlist
db = ase.io.read("OE62.db", ":")
alldists = []
maxdists = []
neighbordists=[]
valencedict = {}
# make a dict that has an entry for each atom type
available_atom_types = [1, 3, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 52, 53]
for i in available_atom_types:
    valencedict[i]=[]
for i in range(len(db)):
    dists = db[i].get_all_distances()
    # this gives you an ase object with some properties
    neighbors = ase.neighborlist.build_neighbor_list(db[i])
    neighborlist = neighbors.nl.neighbors
    # distances of all neighbors
    neighbordist = []
    # get atom types
    atypes = db[i].get_atomic_numbers()
    #every atom has a list of neighbors, we iterate over those
    for index,iatom in enumerate(neighborlist):
        #the first index is always the atom we look at
        # the other indices contain the neighbors
        #append the dict entry for the atom type with the len-1 of the neighbor list
        # check if we have not missed an atom type:
        if atypes[index] not in valencedict:
            print(atypes[index], "not an entry of the dictionary, check if this atom type is missed in the OE62-publication")
        valencedict[atypes[index]].append(len(iatom)-1)
        for ineighbor in iatom:
            # now we append the neighbordistance-list with all distances of one atom to its neighbors
            neighbordist.append(dists[iatom[0],ineighbor])
    #now we want the maximum distance of all neighbour-distances
    maxdists.append(np.max(neighbordist))
radial_limit_upper = np.max(maxdists)
print(radial_limit_upper)
radial_limit_lower = np.min(maxdists)
print(radial_limit_lower)
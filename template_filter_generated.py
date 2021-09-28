
import numpy as np
import pickle
import os
import argparse
import time
import collections

from scipy.spatial.distance import pdist
from schnetpack import Properties
from utility_classes import Molecule, ConnectivityCompressor
from utility_functions import update_dict
from ase import Atoms
from ase.db import connect
import ase

def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser() 
    main_parser.add_argument('data_path',
                             help='Path to generated molecules in .mol_dict format, '
                                  'a database called "generated_molecules.db" with the '
                                  'filtered molecules along with computed statistics '
                                  '("generated_molecules_statistics.npz") will be '
                                  'stored in the same directory as the input file/s '
                                  '(if the path points to a directory, all .mol_dict '
                                  'files in the directory will be merged and filtered '
                                  'in one pass)')
    main_parser.add_argument('--valence',
                             default=[1,1,3,1, 5,3, 6,4, 7,3, 8,2, 9,1, 14,4, 15,5, 16,6, 17,1, 33,5, 34,6, 35,1, 52,6, 53,1], type=int,
                             nargs='+',
                             help='the valence of atom types in the form '
                                  '[type1 valence type2 valence ...] '
                                  '(default: %(default)s)')
    main_parser.add_argument('--filters', type=str, nargs='*',
                             default=['valence', 'disconnected', 'unique'],
                             choices=['valence', 'disconnected', 'unique'],
                             help='Select the filters applied to identify '
                                  'invalid molecules (default: %(default)s)')
    main_parser.add_argument('--store', type=str, default='valid',
                             choices=['all', 'valid'],
                             help='How much information shall be stored '
                                  'after filtering: \n"all" keeps all '
                                  'generated molecules and statistics, '
                                  '\n"valid" keeps only valid molecules'
                                  '(default: %(default)s)')
    main_parser.add_argument('--print_file',
                             help='Use to limit the printing if results are '
                                  'written to a file instead of the console ('
                                  'e.g. if running on a cluster)',
                             action='store_true')
    return main_parser


def _get_atoms_per_type_str(mol):
    '''
    Get a string representing the atomic composition of a molecule (i.e. the number
    of atoms per type in the molecule, e.g. H2C3O1, where the order of types is
    determined by increasing nuclear charge).
    Args:
        mol (utility_classes.Molecule or numpy.ndarray: the molecule (or an array of
            its atomic numbers)
    Returns:
        str: the atomic composition of the molecule
    '''
    if isinstance(mol, Molecule):
        n_atoms_per_type = mol.get_n_atoms_per_type()
    else:
        # assume atomic numbers were provided
        n_atoms_per_type = np.bincount(mol, minlength=10)[
            np.array(list(Molecule.type_infos.keys()), dtype=int)]
    s = ''
    for t, n in zip(Molecule.type_infos.keys(), n_atoms_per_type):
        s += f'{Molecule.type_infos[t]["name"]}{int(n):d}'
    return s

def _update_dict(old_dict, **kwargs):
    '''
    Update an existing dictionary (any->list of any) with new entries where the new
    values are either appended to the existing lists if the corresponding key already
    exists in the dictionary or a new list under the new key is created.
    Args:
        old_dict (dict (any->list of any)): original dictionary that shall be updated
        **kwargs: keyword arguments that can either be a dictionary of the same format
            as old_dict (new_dict=dict (any->list of any)) which will be merged into
            old_dict or a single key-value pair that shall be added (key=any, val=any)
    Returns:
        dict (any->list of any): the updated dictionary
    '''
    if 'new_dict' in kwargs:
        for key in kwargs['new_dict']:
            if key in old_dict:
                old_dict[key] += kwargs['new_dict'][key]
            else:
                old_dict[key] = kwargs['new_dict'][key]
    if 'val' in kwargs and 'key' in kwargs:
        if kwargs['key'] in old_dict:
            old_dict[kwargs['key']] += [kwargs['val']]
        else:
            old_dict[kwargs['key']] = [kwargs['val']]
    return old_dict

def remove_disconnected(connectivity_batch, valid=None):
    '''
    Identify structures which are actually more than one molecule (as they consist of
    disconnected structures) and mark them as invalid.
    Args:
        connectivity_batch (numpy.ndarray): batch of connectivity matrices
        valid (numpy.ndarray, optional): array of the same length as connectivity_batch
            which flags molecules as valid, if None all connectivity matrices are
            considered to correspond to valid molecules in the beginning (default:
            None)
    Returns:
        dict (str->numpy.ndarray): a dictionary containing an array which marks
            molecules as valid under the key 'valid' (identified disconnected
            structures will now be marked as invalid in contrast to the flag in input
            argument valid)
    '''
    if valid is None:
        valid = np.ones(len(connectivity_batch), dtype=bool)
    # find disconnected parts for every given connectivity matrix
    for i, con_mat in enumerate(connectivity_batch):
        # only work with molecules categorized as valid
        if not valid[i]:
            continue
        seen, queue = {0}, collections.deque([0])
        while queue:
            vertex = queue.popleft()
            for node in np.argwhere(con_mat[vertex] > 0).flatten():
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        # if the seen nodes do not include all nodes, there are disconnected
        #  parts and the molecule is invalid
        if seen != {*range(len(con_mat))}:
            valid[i] = False
    return {'valid': valid}

def check_valency(positions, numbers, valence, filter_by_valency=True,
                  print_file=True, prog_str=None, picklable_mols=False):
    '''
    Build utility_classes.Molecule objects from provided atom positions and types
    of a set of molecules and assess whether they are meeting the valency
    constraints or not (i.e. all of their atoms have the correct number of bonds).
    Note that all input molecules need to have the same number of atoms.
    Args:
        positions (list of numpy.ndarray): list of positions of atoms in euclidean
            space (n_atoms x 3) for each molecule
        numbers (numpy.ndarray): list of nuclear charges/types of atoms
            (e.g. 1 for hydrogens, 6 for carbons etc.) for each molecule
        valence (numpy.ndarray): list of valency of each atom type where the index in
            the list corresponds to the type (e.g. [0, 1, 0, 0, 0, 0, 2, 3, 4, 1] for
            qm9 molecules as H=type 1 has valency of 1, O=type 6 has valency of 2,
            N=type 7 has valency of 3 etc.)
        filter_by_valency (bool, optional): whether molecules that fail the valency
            check should be marked as invalid, else all input molecules will be
            classified as valid but the connectivity matrix is still computed and
            returned (default: True)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (default: None)
        picklable_mols (bool, optional): set True to remove all the information in
            the returned list of utility_classes.Molecule objects that can not be
            serialized with pickle (e.g. the underlying Open Babel ob.Mol object,
            default: False)
    Returns:
        dict (str->list/numpy.ndarray): a dictionary containing a list of
            utility_classes.Molecule ojbects under the key 'mols', a numpy.ndarray with
            the corresponding (n_atoms x n_atoms) connectivity matrices under the key
            'connectivity', and a numpy.ndarray (key 'valid') that marks whether a
            molecule has passed (entry=1) or failed (entry=0) the valency check if
            filter_by_valency is True (otherwise it will be 1 everywhere)
    '''
    n_atoms = len(numbers[0])
    n_mols = len(numbers)
    thresh = n_mols if n_mols < 30 else 30
    connectivity = np.zeros((len(positions), n_atoms, n_atoms))
    valid = np.ones(len(positions), dtype=bool)
    mols = []
    for i, (pos, num) in enumerate(zip(positions, numbers)):
        mol = Molecule(pos, num, store_positions=False)
        con_mat = mol.get_connectivity()
        random_ord = range(len(pos))
        # filter incorrect valence if desired
        if filter_by_valency:
            nums = num
            # try to fix connectivity if it isn't correct already
            for _ in range(10):
                if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                    val = True
                    break
                else:
                    val = False
                    con_mat = mol.get_fixed_connectivity()
                    if np.all(
                            np.sum(con_mat, axis=0) == valence[nums]):
                        val = True
                        break
                    random_ord = np.random.permutation(range(len(pos)))
                    mol = Molecule(pos[random_ord], num[random_ord])
                    con_mat = mol.get_connectivity()
                    nums = num[random_ord]
            valid[i] = val

            if ((i + 1) % thresh == 0) and not print_file \
                and prog_str is not None:
                print('\033[K', end='\r', flush=True)
                print(f'{prog_str} ({100 * (i + 1) / n_mols:.2f}%)',
                      end='\r', flush=True)

        # reverse random order and save fixed connectivity matrix
        rand_ord_rev = np.argsort(random_ord)
        connectivity[i] = con_mat[rand_ord_rev][:, rand_ord_rev]
        if picklable_mols:
            mol.get_fp_bits()
            mol.get_can()
            mol.get_mirror_can()
            mol.remove_unpicklable_attributes(restorable=False)
        mols += [mol]
    return {'mols': mols, 'connectivity': connectivity, 'valid': valid}

def remove_disconnected(connectivity_batch, valid=None):
    '''
    Identify structures which are actually more than one molecule (as they consist of
    disconnected structures) and mark them as invalid.
    Args:
        connectivity_batch (numpy.ndarray): batch of connectivity matrices
        valid (numpy.ndarray, optional): array of the same length as connectivity_batch
            which flags molecules as valid, if None all connectivity matrices are
            considered to correspond to valid molecules in the beginning (default:
            None)
    Returns:
        dict (str->numpy.ndarray): a dictionary containing an array which marks
            molecules as valid under the key 'valid' (identified disconnected
            structures will now be marked as invalid in contrast to the flag in input
            argument valid)
    '''
    if valid is None:
        valid = np.ones(len(connectivity_batch), dtype=bool)
    # find disconnected parts for every given connectivity matrix
    for i, con_mat in enumerate(connectivity_batch):
        # only work with molecules categorized as valid
        if not valid[i]:
            continue
        seen, queue = {0}, collections.deque([0])
        while queue:
            vertex = queue.popleft()
            for node in np.argwhere(con_mat[vertex] > 0).flatten():
                if node not in seen:
                    seen.add(node)
                    queue.append(node)
        # if the seen nodes do not include all nodes, there are disconnected
        #  parts and the molecule is invalid
        if seen != {*range(len(con_mat))}:
            valid[i] = False
    return {'valid': valid}

def filter_unique(mols, valid=None, use_bits=False):
    '''
    Identify duplicate molecules among a large amount of generated structures.
    The first found structure of each kind is kept as valid original and all following
    duplicating structures are marked as invalid (the molecular fingerprint and
    canonical smiles representation is used which means that different spatial
    conformers of the same molecular graph cannot be distinguished).
    Args:
        mols (list of utility_classes.Molecule): list of all generated molecules
        valid (numpy.ndarray, optional): array of the same length as mols which flags
            molecules as valid (invalid molecules are not considered in the comparison
            process), if None, all molecules in mols are considered as valid (default:
            None)
        use_bits (bool, optional): set True to use the list of non-zero bits instead of
            the pybel.Fingerprint object when comparing molecules (results are
            identical, default: False)
    Returns:
        numpy.ndarray: array of the same length as mols which flags molecules as
            valid (identified duplicates are now marked as invalid in contrast to the
            flag in input argument valid)
        numpy.ndarray: array of length n_mols where entry i is -1 if molecule i is
            an original structure (not a duplicate) and otherwise it is the index j of
            the original structure that molecule i duplicates (j<i)
        numpy.ndarray: array of length n_mols that is 0 for all duplicates and the
            number of identified duplicates for all original structures (therefore
            the sum over this array is the total number of identified duplicates)
    '''
    if valid is None:
        valid = np.ones(len(mols), dtype=bool)
    else:
        valid = valid.copy()
    accepted_dict = {}
    duplicating = -np.ones(len(mols), dtype=int)
    duplicate_count = np.zeros(len(mols), dtype=int)
    for i, mol1 in enumerate(mols):
        if not valid[i]:
            continue
        mol_key = _get_atoms_per_type_str(mol1)
        found = False
        if mol_key in accepted_dict:
            for j, mol2 in accepted_dict[mol_key]:
                # compare fingerprints and canonical smiles representation
                if mol1.tanimoto_similarity(mol2, use_bits=use_bits) >= 1:
                    if (mol1.get_can() == mol2.get_can()
                            or mol1.get_can() == mol2.get_mirror_can()):
                        found = True
                        valid[i] = False
                        duplicating[i] = j
                        duplicate_count[j] += 1
                        break
        if not found:
            accepted_dict = _update_dict(accepted_dict, key=mol_key, val=(i, mol1))
    return valid, duplicating, duplicate_count

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file
    printed_todos = False
    print(args.data_path) 
    # read input file or fuse dictionaries if data_path is a folder
    if not os.path.isdir(args.data_path):
        if not os.path.isfile(args.data_path):
            print(f'\n\nThe specified data path ({args.data_path}) is neither a file '
                  f'nor a directory! Please specify a different data path.')
            raise FileNotFoundError
        else:
#            print(args.data_path)
            with open(args.data_path, 'rb') as f:
                res = pickle.load(f)  # read input file
#            with open("./models/model1/generated/generated.mol_dict", 'rb') as f:
#                res2 = pickle.load(f)
#            for i in res2.keys():
#                try:
#                    res[i]['_positions'] = np.array(list(res[i]['_positions']) + list(res2[i]['_positions']))
#                    res[i]['_atomic_numbers'] = np.array(list(res[i]['_atomic_numbers']) + list(res2[i]['_atomic_numbers']))
#                except:
#                    res[i] = {}
#                    res[i]['_positions'] = list(res2[i]['_positions'])
#                    res[i]['_atomic_numbers'] =  list(res2[i]['_atomic_numbers'])
            #print(len(res)))

            target_db = os.path.join(os.path.dirname(args.data_path),
                                     'generated_molecules.db')
    else:
        print("here")
        print(f'\n\nFusing .mol_dict files in folder {args.data_path}...')
        mol_files = [f for f in os.listdir(args.data_path)
                     if f.endswith(".mol_dict")]
        if len(mol_files) == 0:
            print(f'Could not find any .mol_dict files at {args.data_path}! Please '
                  f'specify a different data path!')
            raise FileNotFoundError
        res = {}
        res2 = {}
        for file in mol_files:
            with open(os.path.join(args.data_path, file), 'rb') as f:
                cur_res = pickle.load(f)
                update_dict(res, cur_res)
            with open("./models/model1/generated/generated.mol_dict", 'rb') as f:
                cur_res2 = pickle.load(f)
                update_dict(res2, cur_res2)

        target_db = os.path.join(args.data_path, 'generated_molecules.db')
    print("Done")
    # compute array with valence of provided atom types
    max_type = max(args.valence[::2])
    valence = np.zeros(max_type+1, dtype=int)
    valence[args.valence[::2]] = args.valence[1::2]

    # print the chosen settings
    valence_str = ''
    for i in range(max_type+1):
        if valence[i] > 0:
            valence_str += f'type {i}: {valence[i]}, '
    filters = []
    if 'valence' in args.filters:
        filters += ['valency']
    if 'disconnected' in args.filters:
        filters += ['connectedness']
    if 'unique' in args.filters:
        filters += ['uniqueness']
    if len(filters) >= 3:
        edit = ', '
    else:
        edit = ' '
    for i in range(len(filters) - 1):
        filters[i] = filters[i] + edit
    if len(filters) >= 2:
        filters = filters[:-1] + ['and '] + filters[-1:]
    string = ''.join(filters)
    print(f'\n\n1. Filtering molecules according to {string}...')
    print(f'\nTarget valence:\n{valence_str[:-2]}\n')

    # initial setup of array for statistics and some counters
    n_generated = 0
    n_valid = 0
    n_non_unique = 0
    stat_heads = ['n_atoms', 'id', 'valid', 'duplicating', 'n_duplicates',
                  'known', 'equals', 'C', 'N', 'O', 'F', 'H','B','Li','Si','P','S','Cl','As','Se','Br','Te','I', 'H1C', 'H1N',
                  'H1O', 'C1C', 'C2C', 'C3C', 'C1N', 'C2N', 'C3N', 'C1O',
                  'C2O', 'C1F', 'N1N', 'N2N', 'N1O', 'N2O', 'N1F', 'O1O',
                  'O1F', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R>8']
    stats = np.empty((len(stat_heads), 0))
    all_mols = []
    connectivity_compressor = ConnectivityCompressor()

    # iterate over generated molecules by length (all generated molecules with n
    # atoms are stored in one batch, so we loop over all available lengths n)
    # this is useful e.g. for finding duplicates, since we only need to compare
    # molecules of the same length (and can actually further narrow down the
    # candidates by looking at the exact atom type composition of each molecule)
    start_time = time.time()
    for n_atoms in res:
        if not isinstance(n_atoms, int) or n_atoms == 0:
            continue
        prog_str = lambda x: f'Checking {x} for molecules of length {n_atoms}'
        work_str = 'valence' if 'valence' in args.filters else 'dictionary'
        if not print_file:
            print('\033[K', end='\r', flush=True)
            print(prog_str(work_str) + ' (0.00%)', end='\r', flush=True)
        else:
            print(prog_str(work_str), flush=True)

        d = res[n_atoms]  # dictionary containing molecules of length n_atoms
        all_pos = d[Properties.R]  # n_mols x n_atoms x 3 matrix with atom positions
        all_numbers = d[Properties.Z]  # n_mols x n_atoms matrix with atom types
        n_mols = len(all_pos)
        #if args.threads <= 0:
        results = check_valency(all_pos, all_numbers, valence,
                                    'valence' in args.filters, print_file,
                                    prog_str(work_str))
        connectivity = results['connectivity']
        mols = results['mols']
        valid = np.ones(n_mols, dtype=int)  # all molecules are valid in the beginning
        # check valency of molecules with length n
        if 'valence' in args.filters:
            if not printed_todos:
                print('Please implement a procedure to check the valence in generated '
                      'molecules! Skipping valence check...')
            # TODO
            # Implement a procedure to assess the valence of generated molecules here!
            # You can adapt and use the Molecule class in utility_classes.py,
            # but the current code is tailored towards the QM9 dataset. In fact,
            # the OpenBabel algorithm to kekulize bond orders is not very reliable
            # and we implemented some heuristics in the Molecule class to fix these
            # flaws for structures made of C, N, O, and F atoms. However, when using
            # more complex structures with a more diverse set of atom types, we think
            # that the reliability of bond assignment in OpenBabel might further
            # degrade and therefore do no recommend to use valence checks for
            # analysis unless it is very important for your use case.

        # detect molecules with disconnected parts if desired
        valid = remove_disconnected(connectivity, valid)['valid']
        valid, duplicating, duplicate_count = \
                filter_unique(mols, valid, use_bits=False)
        #print(valid)
        #time.sleep(100)
            # TODO
            # Implement a procedure to assess the connectedness of generated
            # molecules here! You can for example use a connectivity matrix obtained
            # from kekulized bond orders (as we do in our QM9 experiments) or
            # calculate the connectivity with a simple cutoff (e.g. all atoms less
            # then 2.0 angstrom apart are connected, see get_connectivity function in
            # template_preprocess_dataset script).
            # We will remove all molecules where two atoms are closer than 0.3
            


        # identify identical molecules (e.g. using fingerprints)
        # TODO
        # Implement procedure to identify duplicate structures here.
        # This can (heuristically) be achieved in many ways but perfectly identifying
        # all duplicate structures without false positives or false negatives is
        # probably impossible (or computationally prohibitive).
        # For our QM9 experiments, we compared fingerprints and canonical smiles
        # strings of generated molecules using the Molecule class in utility_classes.py
        # that provides functions to obtain these. It would also be possible to compare
        # learned embeddings, e.g. from SchNet or G-SchNet, either as an average over
        # all atoms, over all atoms of the same type, or combined with an algorithm
        # to find the best match between atoms of two molecules considering the
        # distances between embeddings. A similar procedure could be implemented
        # using the root-mean-square deviation (RMSD) of atomic positions. Then it
        # would be required to find the best match between atoms of two structures if
        # they are rotated such that the RMSD given the match is minimal. Again,
        # the best procedure really depends on the experimental setup, e.g. the
        # goals of the experiment, used data and size of molecules in the dataset etc.

        # duplicate_count contains the number of duplicates found for each structure
        duplicate_count = np.zeros(n_mols, dtype=int)
        # duplicating contains -1 for original structures and the id of the duplicated
        # original structure for duplicates
        duplicating = -np.ones(n_mols, dtype=int)
        # remove duplicate structures from list of valid molecules if desired
        if 'unique' in args.filters:
            valid[duplicating != -1] = 0
        # count number of non-unique structures
        n_non_unique += np.sum(duplicate_count)

        # store list of valid molecules in dictionary
        d.update({'valid': valid})

        # collect statistics of generated data
        n_generated += len(valid)
        n_valid += np.sum(valid)
        # count number of atoms per type (here for C, N, O, F, and H as example)
        n_of_types = [np.sum(all_numbers == i, axis=1) for i in [6, 7, 8, 9, 1,3,5,14,15,16, 17,33,34, 35, 52, 53]]
        stats_new = np.stack(
            (np.ones(len(valid)) * n_atoms,     # n_atoms
             np.arange(0, len(valid)),          # id
             valid,                             # valid
             duplicating,                       # id of duplicated molecule
             duplicate_count,                   # number of duplicates
             -np.ones(len(valid)),              # known
             -np.ones(len(valid)),              # equals
             *n_of_types,# n_atoms per type
             *np.zeros((19, len(valid))),       # n_bonds per type pairs
             *np.zeros((7, len(valid)))
             ),
            axis=0)
        stats = np.hstack((stats, stats_new))
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f'Needed {h:d}h{m:02d}m{s:02d}s.')

    # Update and print results
    res.update({'n_generated': n_generated,
                'n_valid': n_valid,
                'stats': stats,
                'stat_heads': stat_heads})

    print(f'Number of generated molecules: {n_generated}\n'
          f'Number of duplicate molecules: {n_non_unique}')
    if 'unique' in args.filters:
        print(f'Number of unique and valid molecules: {n_valid}')
    else:
        print(f'Number of valid molecules (including duplicates): {n_valid}')

    # Remove invalid molecules from results if desired
    if args.store != 'all':
        shrunk_res = {}
        shrunk_stats = np.empty((len(stats), 0))
        i = 0
        for key in res:
            if isinstance(key, str):
                shrunk_res[key] = res[key]
                continue
            if key == 0:
                continue
            d = res[key]
            start = i
            end = i + len(d['valid'])
            idcs = np.where(d['valid'])[0]
            if len(idcs) < 1:
                i = end
                continue
            # shrink stats
            idx_id = stat_heads.index('id')
            idx_known = stat_heads.index('known')
            new_stats = stats[:, start:end]
            new_stats = new_stats[:, idcs]
            new_stats[idx_id] = np.arange(len(new_stats[idx_id]))  # adjust ids
            shrunk_stats = np.hstack((shrunk_stats, new_stats))
            # shrink positions and atomic numbers
            shrunk_res[key] = {Properties.R: d[Properties.R][idcs],
                               Properties.Z: d[Properties.Z][idcs]}
            i = end

        shrunk_res['stats'] = shrunk_stats
        res = shrunk_res

    # transfer results to ASE db
    # get filename that is not yet taken for db
    if os.path.isfile(target_db):
        file_name, _ = os.path.splitext(target_db)
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name + '_' + str(expand)
            if os.path.isfile(new_file_name + '.db'):
                continue
            else:
                target_db = new_file_name + '.db'
                break
    print(f'Transferring generated molecules to database at {target_db}...')
    # open db
    with connect(target_db) as conn:
        # store metadata
        conn.metadata = {'n_generated': int(n_generated),
                         'n_non_unique': int(n_non_unique),
                         'n_valid': int(n_valid),
                         'non_unique_removed_from_valid': 'unique' in args.filters}
        # store molecules
        for n_atoms in res:
            if isinstance(n_atoms, str) or n_atoms == 0:
                continue
            d = res[n_atoms]
            all_pos = d[Properties.R]
            all_numbers = d[Properties.Z]
            for pos, num in zip(all_pos, all_numbers):
                at = Atoms(num, positions=pos)
                conn.write(at)

    # store gathered statistics in separate file
    np.savez_compressed(os.path.splitext(target_db)[0] + f'_statistics.npz',
                        stats=res['stats'], stat_heads=res['stat_heads'])
    print(target_db)
    geoms = ase.io.read(target_db,":")
    print(len(geoms))
    molecule_to_keep = []
    for geom in geoms:
        #os = geom.get_positions()
        atypes = geom.get_atomic_numbers()
        if(sum(atypes)%2 == 0):
            molecule_to_keep.append(geom)
        else:
            pass
    print(len(molecule_to_keep))
    ase.io.write(target_db, molecule_to_keep)
    print("Done")

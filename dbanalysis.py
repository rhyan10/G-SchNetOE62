import ase.io
import ase.db
import sys
import os
import re
import numpy as np
import argparse
sys.path.append('./GSchNetOE62')
from utility_classes import Molecule
import math, sys, random, os
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import json
import six
import sys
project_root = "/storage/chem/mssdjc/share/DesignJoe/screenedDBs/scscore/"
sys.path.append(project_root+"/scscore")
from standalone_model_numpy import SCScorer
#import SCScorer 

class dbAutoAnalysis:
    '''
    Automatically runs the molecular analysis routines defined in MoleculeAnalysis()
    on a given database, parsing the molecules inside and returning a new database
    which is tagged with the implemented analysis metrics. Currently planned metrics
    include:

    * Number of X atoms
    * Number of X-Y bonds
    * Total number of rings
    * Number of 3-, 4-, 5-, 6-, 7- and 8-rings
    * Number of double bonds(?)
    * Number of triple bonds(?)
    * SMILES string

    Each database entry also gets saved with its corresponding bond lengths for every
    X-Y bond pair, and its corresponding bond angles for every X-Y-Z bond triplet.

    Args:
        verbose (boolean, optional): Whether to output all parsing steps or stay quiet.
        save_chunksize (int, optional): Chunk size to break the database up into for more efficient writing to disk.
    '''
    def __init__(self, verbose=True, save_chunksize=50):
        self.verbose = verbose
        self.save_chunksize = save_chunksize

        self.implemented_elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Te', 'Br', 'I']
        self.implemented_bond_orders = [1, 2, 3, 4, 5]
        self.implemented_bond_orders_string = ['single', 'double', 'triple', 'quadruple', 'quintuple']
        self.implemented_ring_counts = [3, 4, 5, 6, 7, 8]

        self.implemented_properties = [
            'natoms',
            'nbonds_and_nrings',
            'aromaticity',
            'SMILES',
            'InChI',
            'SCscore'
        ]

        # Create dict of analysis functions, sorted by key.
        self.analysis_functions = {
            'natoms': self.get_natoms,
            'nbonds_and_nrings': self.get_nbonds_and_nrings,
            'aromaticity': self.get_aromaticity,
            'SMILES': self.get_SMILES,
            'InChI': self.get_InChI,
            'SCscore': self.get_SCscore
        }

        # Create pdb metadata.
        self.db_meta_columns = [
            'id',
            'formula',
            'HOMO',
            'LUMO',
            'HL',
            'natoms',
            'nbonds_1',
            'nbonds_2',
            'nbonds_3',
            'nrings_4',
            'nrings_5',
            'nrings_6',
            'nrings_7',
            'aromaticity',
            'SCscore'
        ]

        self.db_meta_descs = {
            'HOMO': ('HOMO', 'Predicted HOMO energy', 'eV'),
            'LUMO': ('LUMO', 'Predicted LUMO energy', 'eV'),
            'HL': ('HL', 'HOMO-LUMO gap energy', 'eV'),
            'SMILES': ('SMILES', 'Canonical SMILES string', ''),
            'InChI': ('InChI', 'InChI string', ''),
            'aromaticity': ('aromaticity', '% non-H aromaticity of atoms', ''),
            'SCscore': ('SCscore', 'SCscore Synthesisability', '')
        }
        for index, i in enumerate(self.implemented_elements):
            for j in self.implemented_elements[index:]:
                for k in self.implemented_bond_orders:
                    self.db_meta_descs[f'nbonds_{i}{k}{j}'] = (f'nbonds_{i}{k}{j}', f'Number of {i}-{j} {self.implemented_bond_orders_string[k-1]} bonds', '')
        for i in self.implemented_bond_orders:
            self.db_meta_descs[f'nbonds_{i}'] = (f'nbonds_{i}', f'Overall number of {self.implemented_bond_orders_string[i-1]} bonds', '')
        for i in self.implemented_elements:
            self.db_meta_descs[f'natoms_{i}'] = (f'natoms_{i}', f'Number of {i} atoms', '')
            if i not in ['H', 'Li', 'F', 'Cl', 'Br', 'I']:
                self.db_meta_descs[f'n_aromatic_{i}'] = (f'n_aromatic_{i}', f'Number of aromatic {i} atoms', '')
        for i in self.implemented_ring_counts:
            self.db_meta_descs[f'nrings_{i}'] = (f'nrings_{i}', f'Number of {i}-rings', '')
            self.db_meta_descs[f'nrings_{i}_aromatic'] = (f'nrings_{i}_aromatic', f'Number of aromatic {i}-rings', '')
        
        if self.verbose: 
            print('Properties currently implemented in parser:')
            for prop in self.implemented_properties:
                print(f'    {prop}')


    def __call__(self, db_file, action):
        '''
        Load in database, choose analysis type based on action.

        Args:
            db_file (string): Path to ASE database file.
            action (string, one of [parse, update]): Action to perform on database. 'parse'
                takes an unparsed database, such as the raw output database from an iteration,
                and adds all implemented properties as database keys. 'update' takes an
                existing parsed database and updates its keys to include any newly implemented
                properties.
        '''
        # Check args
        if action not in ['parse', 'update']:
            raise ValueError('Unknown value for argument "action". Must be one of ["parse", "update"].')
        if not os.path.exists(db_file):
            raise ValueError(f'No file found at {db_file}. Check your path is correct.')

        if action == 'parse':
            # Connect to unparsed database.
            self.db = ase.db.connect(db_file)
            if self.verbose: 
                print(f'Connected to existing ASE database at {db_file}.')
                print(f'Database contains {len(self.db)} entries.')
            # Error if trying to use a database which has already been parsed.
            if 'has_been_parsed' in self.db.metadata.keys():
                raise RuntimeError(f'Database in {db_file} has already been parsed before! If you are trying to update this database, use "action=\'update\'" instead.')

            # Create new parsed database.
            db_folder, db_name_ext = os.path.split(db_file)
            db_name, db_ext = os.path.splitext(db_name_ext)
            pdb_file = f'{db_folder}/{db_name}_parsed{db_ext}'
            self.pdb = ase.db.connect(pdb_file)
            _metadata = self.pdb.metadata
            _metadata.update({
                'title': f'dbAutoAnalysis Parsed Version of {db_name_ext}',
                'has_been_parsed': 'True'
            })
            _metadata.update({
                'key_descriptions': self.db_meta_descs,
                'default_columns': self.db_meta_columns
            })
            self.pdb.metadata = _metadata
            # Hacky fix to update metadata now.
            self.pdb = ase.db.connect(pdb_file)
            if self.verbose: print(f'Created empty parsed database at {pdb_file}.')

            # Analyse database and add parsed rows to pdb.
            self.parse_db(update_only=False, db_write_file=pdb_file)

            print(f'Parsing complete, parsed database saved to {pdb_file}')

        else:
            # Connect to outdated parsed database.
            self.pdb = ase.db.connect(db_file)
            if self.verbose:
                print(f'Connected to existing ASE database at {db_file}.')
                print(f'Database contains {len(self.pdb)} entries.')
            # Error if trying to use a database which hasn't been parsed before.
            if not self.pdb.metadata['has_been_parsed']:
                raise RuntimeError(f'Database in {db_file} has not been parsed yet! If you are trying to parse this database for the first time, use "action=\'parse\'" instead.')

            # Analyse pdb only where necessary and update with new tags.
            self.parse_db(update_only=True, db_write_file=db_file)

            _metadata = self.pdb.metadata
            _metadata.update({
                'key_descriptions': self.db_meta_descs,
                'default_columns': self.db_meta_columns
            })
            self.pdb.metadata = _metadata
            self.pdb = ase.db.connect(db_file)

            print(f'Re-parsing complete, updated database saved.')


    def parse_db(self, update_only, db_write_file):
        '''
        Run through a database, generating data for all tags in each molecule if update_only==False,
        or generating only the data for newly implemented tags if update_only==True.
        '''
        #  If parsing a new database.
        if not update_only:
            if self.verbose: print('Starting to parse properties from database...')
            row_iterator = self.db.select()

            # Divide the database up into savepoints, so that it doesn't have to save after each row.
            num_rows = self.db.count()
            if self.save_chunksize > num_rows:
                raise ValueError('save_chunksize is greater than the number of rows in the database! Lower save_chunksize and try again.')
            num_rows_divisible = num_rows - (num_rows % self.save_chunksize)
            num_rows_remainder = num_rows - num_rows_divisible
            savepoints = [i for i in range(self.save_chunksize, num_rows_divisible + self.save_chunksize, self.save_chunksize)]
            if self.save_chunksize != num_rows:
                savepoints.append(num_rows)
            savepoints = [sp - 1 for sp in savepoints]

            pdb_cache = None
            for i, row in enumerate(row_iterator):
                if self.verbose:
                    print('\n------------------------------------------')
                    print(f'| Molecule {i+1}                           |')
                    print('------------------------------------------')
                tag_dict = {}
                atoms = row.toatoms(add_additional_information=True)
                # Loop through all properties
                for prop in self.implemented_properties:
                    # Call correct analysis function to get the desired group of properties.
                    tags = self.analysis_functions[prop](atoms)
                    if self.verbose: 
                        prop_keys = tags.keys()
                        for key in prop_keys:
                            print(f'{key}: {tags[key]}')
                    # Insert these properties into the tag dictionary.
                    tag_dict.update(tags)

                # If there is any data in the row's current data_dict (eg. energies), place it in the tag_dict.
                row_data = row.data
                if row_data is not None:
                    for key in row_data.keys():
                        if isinstance(row_data[key], (np.ndarray)):
                            tag_dict[key] = float(row_data[key])
                        else:
                            tag_dict[key] = row_data[key]
                
                # If there is any data in the row's current key_value_pairs (eg. energies), place it in the tag_dict.
                row_kvp = row.key_value_pairs
                if row_kvp is not None:
                    for key in row_kvp.keys():
                        if isinstance(row_kvp[key], (np.ndarray)):
                            tag_dict[key] = float(row_kvp[key])
                        else:
                            tag_dict[key] = row_kvp[key]
                

                # Obtain extra arrays of all unique bond lengths and bond angles (not searchable)
                if self.verbose: print('\nObtaining all bond lengths and bond angles...')
                # bond_lengths, bond_angles = self.get_bond_lengths_and_angles(atoms)
                # ext_table = {
                #     'bond lengths': bond_lengths,
                #     'bond angles': bond_angles
                # }
                # if self.verbose: print('Done.')
                if self.verbose: print('This has not been implemented yet.')
                ext_table = None

                # Write implemented properties to non-searchable data dict.
                data_dict = {'implemented_properties': self.implemented_properties}

                # Cache this row.
                rowdata = [atoms, tag_dict, data_dict, ext_table]
                if pdb_cache is None:
                    pdb_cache = [rowdata]
                else:
                    pdb_cache.append(rowdata)
                if self.verbose:
                    print(f'\nSaved molecule {i+1} to the row cache.')
                    print('------------------------------------------')

                # Check if we are at a savepoint.
                if i in savepoints:
                    # If yes, write the cached rows to the parsed database.
                    with ase.db.connect(db_write_file) as db_writer:
                        for rowdata in pdb_cache:
                            if rowdata[3] is not None:
                                db_writer.write(rowdata[0], rowdata[1], data=rowdata[2], external_tables=rowdata[3])
                            else:
                                db_writer.write(rowdata[0], rowdata[1], data=rowdata[2])

                    if self.verbose:
                        print(f'\nSaved {len(pdb_cache)} cached rows to parsed database.')

                    # Reset database cache.
                    pdb_cache = None


        # If updating an existing database.
        else:
            if self.verbose: print('Starting to parse unimplemented properties from database...')
            row_iterator = self.pdb.select()

            # Divide the database up into savepoints, so that it doesn't have to save after each row.
            num_rows = self.pdb.count()
            if self.save_chunksize > num_rows:
                raise ValueError('save_chunksize is greater than the number of rows in the database! Lower save_chunksize and try again.')
            num_rows_divisible = num_rows - (num_rows % self.save_chunksize)
            num_rows_remainder = num_rows - num_rows_divisible
            savepoints = [i for i in range(self.save_chunksize, num_rows_divisible + self.save_chunksize, self.save_chunksize)]
            if self.save_chunksize != num_rows:
                savepoints.append(num_rows)
            savepoints = [sp - 1 for sp in savepoints]

            pdb_cache = None
            for i, row in enumerate(row_iterator):
                if self.verbose:
                    print('\n------------------------------------------')
                    print(f'| Molecule {i+1}                           |')
                    print('------------------------------------------')
                    print('Searching for properties not implemented in database...')
                row_properties = row.data['implemented_properties']
                missing_properties = []
                for prop in self.implemented_properties:
                    if prop not in row_properties:
                        missing_properties.append(prop)

                if missing_properties == []:
                    print('No unimplemented properties detected!')
                    continue
                else:
                    if self.verbose: 
                        print(f'{len(missing_properties)} unimplemented properties detected:')
                        for prop in missing_properties:
                            print(f'    {prop}')
                        print('')

                tag_dict = {}
                rowid = row.id
                atoms = row.toatoms()
                # Loop through necessary properties
                for prop in missing_properties:
                    # Call correct analysis function to get the desired group of properties.
                    tags = self.analysis_functions[prop](atoms)
                    if self.verbose: 
                        prop_keys = tags.keys()
                        for key in prop_keys:
                            print(f'{key}: {tags[key]}')
                    # Insert these properties into the tag dictionary.
                    tag_dict.update(tags)
                    # Update the row's implemented properties list.
                    data_dict = {'implemented_properties': row_properties.append(prop)}

                # Cache this row.
                rowdata = [rowid, tag_dict, data_dict]
                if pdb_cache is None:
                    pdb_cache = [rowdata]
                else:
                    pdb_cache.append(rowdata)
                if self.verbose:
                    print(f'\nSaved molecule {i+1} to the row cache.')
                    print('------------------------------------------')

                # Check if we are at a savepoint.
                if i in savepoints:
                    # If yes, write the cached rows to the parsed database.
                    with ase.db.connect(db_write_file) as db_writer:
                        for rowdata in pdb_cache:
                            db_writer.update(rowdata[0], **rowdata[1], data=rowdata[2])

                    if self.verbose:
                        print(f'\nUpdated {len(pdb_cache)} cached rows in parsed database.')

                    # Reset database cache.
                    pdb_cache = None

    
    def get_natoms(self, atoms):
        '''
        Gets numbers of atoms of each element in a molecular structure.
        '''
        element_count = None
        for atom in atoms.get_chemical_symbols():
            if element_count is None:
                element_count = {atom: 1}
            else:
                if atom not in element_count.keys():
                    element_count.update({atom: 1})
                else:
                    element_count[atom] += 1
        
        formatted_count = {}
        for elem in element_count.keys():
            formatted_count[f'natoms_{elem}'] = element_count[elem]

        return formatted_count


    def get_nbonds_and_nrings(self, atoms):
        '''
        Gets numbers of bonds, separated by type, and numbers of rings, separated
        by ring size, from a molecular structure.
        '''
        positions = atoms.positions
        numbers = atoms.numbers

        analysis = Molecule(positions, numbers)
        nbonds_and_nrings = analysis.get_bond_stats(ring_analysis='OpenBabel')

        formatted_nbonds_and_nrings = {}
        for key in nbonds_and_nrings.keys():
            bond_order = None
            # If key belongs to an nrings descriptor (RDKit)...
            if key[0] == 'R':
                if key[1] == '>':
                    num_rings = key[2:]
                else:
                    num_rings = key[1:]
                formatted_nbonds_and_nrings[f'nrings_{num_rings}'] = nbonds_and_nrings[key]
            # Else if key belongs to an nrings descriptor(OpenBabel)...
            elif 'nrings_' in key:
                formatted_nbonds_and_nrings[key] = nbonds_and_nrings[key]
            # Otherwise, must be an nbonds descriptor.
            else:
                formatted_nbonds_and_nrings[f'nbonds_{key}'] = nbonds_and_nrings[key]
                bond_order = re.findall('[0-9]+', key)[0]
                if bond_order is None:
                    print(f'Could not extract bond order from key {key}.')
                else:
                    bond_order = f'nbonds_{bond_order}'
                    if bond_order not in formatted_nbonds_and_nrings.keys():
                        formatted_nbonds_and_nrings[bond_order] = nbonds_and_nrings[key]
                    else:
                        formatted_nbonds_and_nrings[bond_order] += nbonds_and_nrings[key]
                
        return formatted_nbonds_and_nrings


    def get_aromaticity(self, atoms):
        '''
        Gets percentage of non-hydrogen aromatic atoms in a molecule, alongside
        the numbers of aromatic atoms of each element type.
        '''
        positions = atoms.positions
        numbers = atoms.numbers

        analysis = Molecule(positions, numbers)
        aro_percent, aro_dict = analysis.get_aromaticity()
        aro_dict['aromaticity'] = aro_percent

        return aro_dict

    
    def get_SMILES(self, atoms):
        '''
        Gets a SMILES string from a molecular structure.
        '''
        positions = atoms.positions
        numbers = atoms.numbers

        analysis = Molecule(positions, numbers)
        smiles = analysis.get_can()

        return {'SMILES': smiles}


    def get_InChI(self, atoms):
        '''
        Gets an InChI string from a molecular structure.
        '''
        positions = atoms.positions
        numbers = atoms.numbers

        analysis = Molecule(positions, numbers)
        inchi = analysis.get_inchi_string()

        return {'InChI': inchi}


    def get_SCscore(self, atoms):
        '''
        Gets a molecule's synthesisability (SCscore) from its structure.
        '''
        positions = atoms.positions
        numbers = atoms.numbers

        analysis = Molecule(positions, numbers)
        smiles = analysis.get_can()

        # INSERT SCscore CODE HERE
        model = SCScorer()
        model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
       
        smiles, SCscore = model.get_score_from_smi(smiles)
        print(SCscore)
        return {'SCscore': SCscore}


    def get_bond_lengths_and_angles(self, atoms):
        '''
        Gets arrays of all unique bond lengths and angles from a molecular structure.
        '''
        raise NotImplementedError()


# Check if being called as a standalone script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dbAutoAnalysis on a given database.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'db',
        help='Path to the database file to be parsed/updated.'
    )
    parser.add_argument(
        'mode',
        choices=['parse', 'update'],
        help=('Which mode to run dbAutoAnalysis in. Running in parse mode will take an'
            'unparsed database and fully parse it to attach the currently implemented'
            'properties. Running in update mode will search each database entry for missing'
            'properties, calculating them and updating database rows if needs be.')
    )
    parser.add_argument(
        '--verbose',
        default='True',
        choices=['True', 'False'],
        help='Whether to output all parsing steps or stay quiet.'
    )
    parser.add_argument(
        '--chunksize',
        default=50,
        type=int,
        help='Chunk size to break the database up into for more efficient writing to disk.'
    )

    args = parser.parse_args()
    db_file = args.db
    db_mode = args.mode
    db_verb = args.verbose
    if db_verb == 'True':
        db_verb = True 
    else:
        db_verb = False
    db_chunksize = args.chunksize

    dbAA = dbAutoAnalysis(db_verb, db_chunksize)
    dbAA(db_file, db_mode)

    exit

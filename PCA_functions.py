import os
import struct
import numpy as np
import ase.db as adb
from sklearn.decomposition import PCA, IncrementalPCA
from dscribe.descriptors import Descriptor, SOAP, MBTR

# Object type imports
from typing import Generator, Tuple, Union, Any, Optional
from numpy.typing import ArrayLike
from ase.db.core import Database


class db_iter_PCA:
    '''Contains functions for running PCA on `dbAutoAnalysis`-parsed databases.
    
    Designed to read in a series of ASE databases corresponding to the iterations
    of a certain energy type (HOMO, LUMO or HL). These databases must have been
    parsed with the `dbAutoAnalysis` class to extract the requisite bonding
    features and save them into the database rows. Can also be used to apply a
    previously fitted PCA model to a new database, such as the parsed
    OE62 database.

    The PCA in question reduces the dimensionality of all molecules\' chosen
    structural descriptors (either SOAP or MBTR), as well as their bonding
    descriptors, assembled from the features in each parsed database. It also
    extracts the HOMO, LUMO and HOMO-LUMO gap energies from the databases, for
    use in clustering plots.

    The output arrays from this PCA can then be saved into a .npz archive for
    later use, as the code takes a long time to run and effort should be made
    not to rerun calculations unless really neccessary.
    '''


    def __init__(self, 
        db_folder: str, 
        db_type: str, 
        descriptor_type: str,
        n_PCA_components: int, 
        n_descriptor_jobs: int,
        print_descriptor_progress: bool = False):
        '''Initialise PCA functions, sanitise user inputs.

        Args:
            db_folder: Path to directory containing all the relevant databases.
            db_type: Type of datasets to load in. Choose from \'HOMO\', \'LUMO\',
                \'HL\' or \'OE62\'.
            descriptor_type: Type of structural descriptor to generate for each
                molecule. Choose from \'SOAP\' or \'MBTR\'
            n_PCA_components: Number of principal components to keep from each PCA. If
                applying an existing PCA to a new db, this must be the same number which
                was used for that PCA.
            n_descriptor_jobs: Number of parallel jobs to run when computing
                descriptors.
            print_descriptor_progress: Whether to write out progress of the
                descriptor construction routines. Can cause crashes when constructing
                descriptors in parallel (when n_descriptor_jobs > 1)
        '''
        # Load in arguments as class attributes.
        self.db_folder = db_folder
        self.db_type = db_type
        self.descriptor_type = descriptor_type
        self.n_PCA_components = n_PCA_components
        self.n_descriptor_jobs = n_descriptor_jobs
        self.verbose_descs = print_descriptor_progress

        # Set other class variables.
        self.implemented_elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Te', 'Br', 'I']
        self.implemented_bond_orders = [1, 2, 3, 4, 5]
        self.implemented_ring_sizes = np.arange(3, 10)

        # Sanitise inputs
        if not os.path.exists(self.db_folder):
            raise ValueError(f'Path {self.db_folder} does not exist.')
        if self.db_type not in ['HOMO', 'LUMO', 'HL', 'OE62', 'Initial']:
            raise ValueError(f'db_type {self.db_type} not recognised.')
        if self.descriptor_type not in ['SOAP', 'MBTR']:
            raise ValueError(f'descriptor_type {self.descriptor_type} not recognised.')
        if self.n_descriptor_jobs > 1 and self.verbose_descs:
            print('Warning: printing descriptor construction progress when constructing descriptors in parallel has been known to cause crashes. Proceed with caution.')

        # Create bonding descriptor template.
        self.bonding_properties = []
        for index, i in enumerate(self.implemented_elements):
            self.bonding_properties.append(f'natoms_{i}')
            self.bonding_properties.append(f'n_aromatic_{i}')
            for j in self.implemented_elements[index:]:
                for k in self.implemented_bond_orders:
                    self.bonding_properties.append(f'nbonds_{i}{k}{j}')
        for i in self.implemented_bond_orders:
            self.bonding_properties.append(f'nbonds_{i}')
        for i in self.implemented_ring_sizes:
            self.bonding_properties.append(f'nrings_{i}')
            self.bonding_properties.append(f'nrings_{i}_aromatic')
        self.bonding_properties.append('aromaticity')
        self.n_bonding_features = len(self.bonding_properties)

        self.energy_properties = ['HOMO', 'LUMO', 'HL']
            

    def load_db(self, db_iter: Optional[int]=None) -> Tuple[Database, Generator[Any, None, None]]:
        '''Loads an ASE database of a specified type and iteration number.

        Args:
            db_iter: The iteration of the database set to load.
        
        Returns:
            The ase.db.connect() object interface and a Generator for iterating
            over all database rows (from db.select()).
        '''
        if self.db_type == 'OE62':
            if db_iter is not None:
                raise ValueError('db_type OE62 has no iterations, arg db_iter should be None')
            else:
                db_path = f'{self.db_folder}/{self.db_type}_parsed.db'
        elif self.db_type == 'Initial':
            if db_iter is not None:
                raise ValueError('db_type Initial has no iterations, arg db_iter should be None')
            else:
                db_path = f'{self.db_folder}/{self.db_type}_parsed.db'
        else:
            if db_iter is None:
                raise ValueError('db_types other than OE62 and Initial require db_iter to be an integer value.')
            else:
                db_path = f'{self.db_folder}/{self.db_type}{db_iter}_parsed.db'

        # Check db file exists
        if not os.path.exists(db_path):
            raise RuntimeError(f'ASE database not found at {db_path}')
        db = adb.connect(db_path)
        iterator = db.select()

        return db, iterator


    def create_desc_gen(self) -> Descriptor:
        '''Creates a descriptor generator object which can be called under a common name.
        
        Descriptor generators are created by DScribe and can be used to generate
        the structural descriptors (SOAP or MBTR) for large batches of molecules
        in parallel.

        Returns:
            The descriptor generator object, which can be called with `desc.create()`
            to generate descriptors for a set of molecules.
        '''
        if self.descriptor_type == 'SOAP':
            desc = SOAP(
                species = self.implemented_elements,    # List of the elements which the descriptor can accept
                periodic = False,                       # We are only studying isolated molecules
                rcut = 3.5,                             # Cutoff for local region in Angstroms
                nmax = 8,                               # Number of radial basis functions
                lmax = 6,                               # Maximum degree of spherical harmonics
                average='inner'                         # Averaging over all sites to create a global descriptor.
            )
        else:
            desc = MBTR(
                species = self.implemented_elements, 
                k2 = {
                "geometry": {"function": "distance"},
                "grid": {"min": 0.0, "max": 10.0, "sigma": 0.1, "n": 50},
                "weighting": {"function": "exp", "scale": 0.75, "threshold": 1e-2}
                },
                k3 = {
                    "geometry": {"function": "angle"},
                    "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
                    "weighting" : {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                }, 
                periodic = False,
                normalization = 'l2_each',
                flatten=True
            )
        
        return desc


    def construct_descriptors(self, db_iterator) -> ArrayLike:
        '''Construct a SOAP/MBTR descriptor for each molecule in a database selection.
    
        Iterates through a generator to extract an ASE Atoms object from each
        database row, then generates a single flattened descriptor vector from
        each Atoms object.
        
        Descriptor vector creation can be parallelised by passing a value to 
        self.n_descriptor_jobs.
        
        When creating SOAP descriptor vectors, this function additionally iterates
        through every Atoms object and moves its centre of mass to (0, 0, 0). This
        allows the SOAP descriptor to use a common reference point between molecules.
        
        Args:
            db_iterator: `Generator` object returned by an ASE database's `.select()`
                method.

        Returns:
            NumPy array of descriptor vectors of size `n_molecules x n_features`.
        '''
        # Create base descriptor generator.
        desc = self.create_desc_gen()
        print(f'Created {self.descriptor_type} descriptor generator.')

        # Create initial list of Atoms objects (may take up a significant chunk of memory).
        atoms_list = [row.toatoms() for row in db_iterator]
        n_molecules = len(atoms_list)
        print(f'Loaded in {n_molecules} Atoms objects.')

        if self.descriptor_type == 'SOAP':
            print('Starting generation of SOAP descriptors...')
            all_descriptors = desc.create(atoms_list, n_jobs=self.n_descriptor_jobs)
            print('Done.')

            # Explicitly try to deallocate memory in use by this large array.
            del atoms_list

            # return all_descriptors[:, 0, :]
            return all_descriptors

        else:
            print('Starting generation of MBTR descriptors...')
            all_descriptors = desc.create(atoms_list, n_jobs=self.n_descriptor_jobs)

            # Explicitly try to deallocate memory in use by this large array.
            del atoms_list

            return all_descriptors


    def fit_PCAs(self, return_fits=None) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        '''Construct and fit PCAs for the structural and bonding descriptors.
        
        Loops through all parsed databases of `self.db_type` in `self.db_folder`, 
        constructing descriptor vectors for each molecule in each database. These 
        descriptor vectors are composed of a structural descriptor vector (SOAP or
        MBTR) and a bonding descriptor vector (bond order data from parsed database).

        Due to the size of the structural descriptor vectors, these get fed into an
        incremental PCA (iPCA) model as training data, database by database, in order
        to maintain memory efficiency. The bonding descriptor vectors are fed into a
        regular PCA as a single batch of training data. These two PCAs are used to
        reduce the dimensionality of the two vectors to a `self.n_PCA_components` length
        vector for each molecule.

        Args:
            return_fits: If a str is passed, save the fitted PCA objects to a .npz archive
                under this filename.

        Returns:
            An [n_molecules x 2 x n_PCA_components] ndarray of the final principal
            components. The first row along axis 1 represents the principal structural
            components, while the second row represents the principal bonding components.
            Returns another ndarray of size [n_molecules x 3] with columns representing
            the HOMO, LUMO and HOMO-LUMO gap energies of each molecule. Also returns a 
            third ndarray of size [2 x n_PCA_components] containing the variance ratios
            for each principal component. The first column represents variance ratios of
            the principal structural components, while the second column represents variance
            ratios of the principal bonding components. Finally, returns a [n_molecules]
            ndarray of the database iteration that each corresponding molecule came from.
        '''
        print('----------------------------------------------')
        print('Initialising...')
        db_iters = 0
        for f in os.listdir(self.db_folder):
            if f.startswith(self.db_type) and f.endswith('_parsed.db'): db_iters += 1
        print(f'Number of {self.db_type} iterations: {db_iters}')

        print('Initialisation complete. Starting iterations over databases.')
        print('----------------------------------------------')
        print()

        # Set up global data arrays
        all_struct_descs = None
        all_bond_descs = None
        all_energies = None
        iteration_map = None

        # Set up iPCA for structural descriptors
        structure_PCA = IncrementalPCA(n_components=self.n_PCA_components)

        for i in range(1, db_iters+1):
            print('----------------------------------------------')
            print(f'Database: {self.db_type}{i}')
            
            db, iterator = self.load_db(i)
            n_molecules = len(db)
            if iteration_map is None:
                iteration_map = np.ones(n_molecules, dtype=int)
            else:
                iteration_map = np.concatenate((iteration_map, np.full(n_molecules, i, dtype=int)))

            print('Database loaded, constructing structural descriptors.')
            print()
            iter_struct_descs = self.construct_descriptors(iterator)

            print('Constructing bonding descriptors and collecting energies.')
            iter_bond_descs = np.zeros((n_molecules, self.n_bonding_features))
            iter_energies = np.zeros((n_molecules, 3))
            iterator = db.select()
            row_counter = -1
            for row in iterator:
                row_counter += 1
                row_features = row.key_value_pairs.keys()
                for j, feature in enumerate(self.bonding_properties):
                    if feature in row_features:
                        iter_bond_descs[row_counter, j] = row[feature]
                for j, e_type in enumerate(self.energy_properties):
                    iter_energies[row_counter, j] = row[e_type]

            # Concatenate to global arrays
            print('Concatenating descriptors to global arrays.')
            if all_struct_descs is None:
                all_struct_descs = iter_struct_descs
            else:
                all_struct_descs = np.concatenate((all_struct_descs, iter_struct_descs))

            if all_bond_descs is None:
                all_bond_descs = iter_bond_descs
            else:
                all_bond_descs = np.concatenate((all_bond_descs, iter_bond_descs))

            if all_energies is None:
                all_energies = iter_energies
            else:
                all_energies = np.concatenate((all_energies, iter_energies))

            # Incrementally train PCA
            print('Training iPCA on structural descriptors...')
            structure_PCA.partial_fit(iter_struct_descs)
            print('Done.')

            # Try to force Python to free up memory
            del iter_struct_descs
            del iter_bond_descs

            print('Iteration complete.')
            print('----------------------------------------------')
            print()

        # Once all databases have been looped through, train bonding PCA on the overall array.
        print('----------------------------------------------')
        print('Database loops complete, training PCA on bonding descriptors...')
        bonding_PCA = PCA(n_components=self.n_PCA_components)
        bonding_PCA.fit(all_bond_descs)
        print('Done.')

        print('Applying dimensionality reduction to both sets of descriptors...')
        n_molecules_tot = len(all_bond_descs)
        pca_results = np.zeros((n_molecules_tot, 2, self.n_PCA_components))
        ratio_results = np.zeros((2, self.n_PCA_components))

        struct_pca_results = structure_PCA.transform(all_struct_descs)
        print('Transformed structural descriptors.')
        pca_results[:, 0, :] = struct_pca_results
        ratio_results[0, :] = structure_PCA.explained_variance_ratio_
        # Free up memory
        del struct_pca_results
        print('Placed structural principal components in final results array.')

        bond_pca_results = bonding_PCA.transform(all_bond_descs)
        print('Transformed bonding descriptors.')
        pca_results[:, 1, :] = bond_pca_results
        ratio_results[1, :] = bonding_PCA.explained_variance_ratio_
        # Free up memory
        del bond_pca_results
        print('Placed bonding principal components in final results array.')

        if return_fits is not None:
            self.save_PCA_objects(return_fits, bonding_PCA, structure_PCA)
            print(f'Saved PCA objects to {return_fits}.')

        print('Finished.')
        print('----------------------------------------------')

        return pca_results, all_energies, ratio_results, iteration_map

    
    def fit_OE62_PCA(self, return_fits=None, split_db: bool=False) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        '''Construct and fit PCAs for the structural and bonding descriptors.
        
        Loops through molecules in parsed OE62 database, constructing descriptor vectors
        for each molecule. These descriptor vectors are composed of a structural 
        descriptor vector (SOAP or MBTR) and a bonding descriptor vector (bond order
        data from parsed database).

        Both descriptor vectors are fed into a regular PCA as single batches of training
        data. These two PCAs are used to reduce the dimensionality of the two vectors to
        a `self.n_PCA_components` length vector for each molecule.

        Args:
            return_fits: If a str is passed, save the fitted PCA objects to a .npz archive
                under this filename.
            split_db: Whether to divide the database in two when constructing the
                structural descriptors. Useful for very large databases.

        Returns:
            An [n_molecules x 2 x n_PCA_components] ndarray of the final principal
            components. The first row along axis 1 represents the principal structural
            components, while the second row represents the principal bonding components.
            Returns another ndarray of size [n_molecules x 3] with columns representing
            the HOMO, LUMO and HOMO-LUMO gap energies of each molecule. Also returns a 
            third ndarray of size [2 x n_PCA_components] containing the variance ratios
            for each principal component. The first column represents variance ratios of
            the principal structural components, while the second column represents variance
            ratios of the principal bonding components.
        '''
        print('----------------------------------------------')
        print('Initialising...')

        db, iterator = self.load_db()
        n_molecules = len(db)

        print('Initialisation complete. Starting PCA.')
        print('----------------------------------------------')
        print()

        print('Constructing structural descriptors.')
        if not split_db:
            struct_descs = self.construct_descriptors(iterator)
        else:
            print('  - split_db enabled, constructing descriptors for first half of database.')
            iterator_1 = db.select(f'id<{n_molecules/2}')
            struct_descs_1 = self.construct_descriptors(iterator_1)
            print('  - Constructing descriptors for second half of database.')
            iterator_2 = db.select(f'id>={n_molecules/2}')
            struct_descs_2 = self.construct_descriptors(iterator_2)
            print('  - Concatenating split descriptor set.')
            struct_descs = np.concatenate((struct_descs_1, struct_descs_2))
            # Manually free up memory
            del struct_descs_1; del struct_descs_2

        print('Constructing bonding descriptors and collecting energies.')
        bond_descs = np.zeros((n_molecules, self.n_bonding_features))
        energies = np.zeros((n_molecules, 3))
        iterator = db.select()
        row_counter = -1
        for row in iterator:
            row_counter += 1
            row_features = row.key_value_pairs.keys()
            for j, feature in enumerate(self.bonding_properties):
                if feature in row_features:
                    bond_descs[row_counter, j] = row[feature]
            for j, e_type in enumerate(self.energy_properties):
                energies[row_counter, j] = row[e_type]

        # Once database has been looped through, train PCAs on the overall arrays.
        print('----------------------------------------------')
        print('Descriptor generation complete, training PCA on bonding descriptors...')
        bonding_PCA = PCA(n_components=self.n_PCA_components)
        bonding_PCA.fit(bond_descs)
        print('Done.')
        print('Training PCA on structural descriptors...')
        structure_PCA = PCA(n_components=self.n_PCA_components)
        structure_PCA.fit(struct_descs)

        print('Applying dimensionality reduction to both sets of descriptors...')
        pca_results = np.zeros((n_molecules, 2, self.n_PCA_components))
        ratio_results = np.zeros((2, self.n_PCA_components))

        struct_pca_results = structure_PCA.transform(struct_descs)
        print('Transformed structural descriptors.')
        pca_results[:, 0, :] = struct_pca_results
        ratio_results[0, :] = structure_PCA.explained_variance_ratio_
        # Free up memory
        del struct_pca_results
        print('Placed structural principal components in final results array.')

        bond_pca_results = bonding_PCA.transform(bond_descs)
        print('Transformed bonding descriptors.')
        pca_results[:, 1, :] = bond_pca_results
        ratio_results[1, :] = bonding_PCA.explained_variance_ratio_
        # Free up memory
        del bond_pca_results
        print('Placed bonding principal components in final results array.')

        if return_fits is not None:
            self.save_PCA_objects(return_fits, bonding_PCA, structure_PCA)
            print(f'Saved PCA objects to {return_fits}.')

        print('Finished.')
        print('----------------------------------------------')

        return pca_results, energies, ratio_results


    def transform_db(self, bonding_PCA: PCA, structure_PCA: IncrementalPCA, split_db: bool=False):
        '''Use a set of existing fitted PCA models to run PCA on an unfitted database.
        
        Currently only supports transforming bonding and structural descriptors
        from OE62 dataset or initially generated G-SchNet dataset.
        
        Args:
            bonding_PCA: Fitted PCA object to transform a database's bonding
                descriptors with.
            structure_PCA: Fitted PCA object to transform a database's structural
                descriptors with.
            split_db: Whether to divide the database in two when constructing the
                structural descriptors. Useful for very large databases.
                
        Returns:
            An [n_molecules x 2 x n_PCA_components] ndarray of the final principal
            components. The first row along axis 1 represents the principal structural
            components, while the second row represents the principal bonding components.
            Returns another ndarray of size [n_molecules x 3] with columns representing
            the HOMO, LUMO and HOMO-LUMO gap energies of each molecule.
        '''
        # Set up global data arrays
        struct_descs = None
        bond_descs = None
        energies = None

        db, iterator = self.load_db()
        n_molecules = len(db)

        print('----------------------------------------------')
        print('Database loaded, constructing structural descriptors.')
        
        if not split_db:
            struct_descs = self.construct_descriptors(iterator)
        else:
            print('  - split_db enabled, constructing descriptors for first half of database.')
            iterator_1 = db.select(f'id<{n_molecules/2}')
            struct_descs_1 = self.construct_descriptors(iterator_1)
            print('  - Constructing descriptors for second half of database.')
            iterator_2 = db.select(f'id>={n_molecules/2}')
            struct_descs_2 = self.construct_descriptors(iterator_2)
            print('  - Concatenating split descriptor set.')
            struct_descs = np.concatenate((struct_descs_1, struct_descs_2))
            # Manually free up memory
            del struct_descs_1; del struct_descs_2

        print()
        print('Constructing bonding descriptors and collecting energies.')
        bond_descs = np.zeros((n_molecules, self.n_bonding_features))
        energies = np.zeros((n_molecules, 3))
        iterator = db.select()
        row_counter = -1
        for row in iterator:
            row_counter += 1
            row_features = row.key_value_pairs.keys()
            for j, feature in enumerate(self.bonding_properties):
                if feature in row_features:
                    bond_descs[row_counter, j] = row[feature]
            for j, e_type in enumerate(self.energy_properties):
                energies[row_counter, j] = row[e_type]

        print('Descriptor construction complete.')
        print('Applying dimensionality reduction to both sets of descriptors...')
        pca_results = np.zeros((n_molecules, 2, self.n_PCA_components))

        struct_pca_results = structure_PCA.transform(struct_descs)
        print('Transformed structural descriptors.')
        pca_results[:, 0, :] = struct_pca_results
        # Free up memory
        del struct_pca_results
        print('Placed structural principal components in final results array.')

        bond_pca_results = bonding_PCA.transform(bond_descs)
        print('Transformed bonding descriptors.')
        pca_results[:, 1, :] = bond_pca_results
        # Free up memory
        del bond_pca_results
        print('Placed bonding principal components in final results array.')        

        print('Finished.')
        print('----------------------------------------------')

        return pca_results, energies


    def transform_db_set(self, bonding_PCA: PCA, structure_PCA: PCA) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        '''Use a set of existing fitted PCA models to run PCA on an unfitted set of databases.

        Can be used to transform a set of parsed G-SchNet iteration databases
        with a set of bonding/structural PCAs fitted on other data, e.g. on
        the parsed OE26 database.

        Args:
            bonding_PCA: Fitted PCA object to transform a database's bonding
                descriptors with.
            structure_PCA: Fitted PCA object to transform a database's structural
                descriptors with.

        Returns:
            An [n_molecules x 2 x n_PCA_components] ndarray of the final principal
            components. The first row along axis 1 represents the principal structural
            components, while the second row represents the principal bonding components.
            Returns another ndarray of size [n_molecules x 3] with columns representing
            the HOMO, LUMO and HOMO-LUMO gap energies of each molecule. Finally, returns 
            a [n_molecules] ndarray of the database iteration that each corresponding 
            molecule came from.
        '''
        print('----------------------------------------------')
        print('Initialising...')
        db_iters = 0
        for f in os.listdir(self.db_folder):
            if f.startswith(self.db_type) and f.endswith('_parsed.db'): db_iters += 1
        print(f'Number of {self.db_type} iterations: {db_iters}')

        print('Initialisation complete. Starting iterations over databases.')
        print('----------------------------------------------')
        print()

        # Set up global data arrays
        all_struct_descs = None
        all_bond_descs = None
        all_energies = None
        iteration_map = None

        for i in range(1, db_iters+1):
            print('----------------------------------------------')
            print(f'Database: {self.db_type}{i}')
            
            db, iterator = self.load_db(i)
            n_molecules = len(db)
            if iteration_map is None:
                iteration_map = np.ones(n_molecules, dtype=int)
            else:
                iteration_map = np.concatenate((iteration_map, np.full(n_molecules, i, dtype=int)))

            print('Database loaded, constructing structural descriptors.')
            print()
            iter_struct_descs = self.construct_descriptors(iterator)

            print('Constructing bonding descriptors and collecting energies.')
            iter_bond_descs = np.zeros((n_molecules, self.n_bonding_features))
            iter_energies = np.zeros((n_molecules, 3))
            iterator = db.select()
            row_counter = -1
            for row in iterator:
                row_counter += 1
                row_features = row.key_value_pairs.keys()
                for j, feature in enumerate(self.bonding_properties):
                    if feature in row_features:
                        iter_bond_descs[row_counter, j] = row[feature]
                for j, e_type in enumerate(self.energy_properties):
                    iter_energies[row_counter, j] = row[e_type]

            # Concatenate to global arrays
            print('Concatenating descriptors to global arrays.')
            if all_struct_descs is None:
                all_struct_descs = iter_struct_descs
            else:
                all_struct_descs = np.concatenate((all_struct_descs, iter_struct_descs))

            if all_bond_descs is None:
                all_bond_descs = iter_bond_descs
            else:
                all_bond_descs = np.concatenate((all_bond_descs, iter_bond_descs))

            if all_energies is None:
                all_energies = iter_energies
            else:
                all_energies = np.concatenate((all_energies, iter_energies))

            # Try to force Python to free up memory
            del iter_struct_descs
            del iter_bond_descs

            print('Iteration complete.')
            print('----------------------------------------------')
            print()

        print('Descriptor construction complete.')
        print('Applying dimensionality reduction to both sets of descriptors...')
        n_molecules_tot = len(all_bond_descs)
        pca_results = np.zeros((n_molecules_tot, 2, self.n_PCA_components))

        struct_pca_results = structure_PCA.transform(all_struct_descs)
        print('Transformed structural descriptors.')
        pca_results[:, 0, :] = struct_pca_results
        # Free up memory
        del struct_pca_results
        print('Placed structural principal components in final results array.')

        bond_pca_results = bonding_PCA.transform(all_bond_descs)
        print('Transformed bonding descriptors.')
        pca_results[:, 1, :] = bond_pca_results
        # Free up memory
        del bond_pca_results
        print('Placed bonding principal components in final results array.')        

        print('Finished.')
        print('----------------------------------------------')
        
        return pca_results, all_energies, iteration_map
      

    def save_PCA_results(self, filename: str, pca_results: ArrayLike, energies: ArrayLike, 
            pca_variance_ratios: Optional[ArrayLike]=None, iter_map: Optional[ArrayLike]=None):
        '''Save final results as an uncompressed .npz archive.
        
        Args:
            filename: Filename to save archive to.
            pca_results: First return from fit_PCAs() or transform_db().
            energies: Second return from fit_PCAs() or transform_db().
            pca_variance_ratios: Third return from fit_PCAs().
            iter_map: Fourth return from fit_PCAs()
        '''
        np.savez(filename, pca_results=pca_results, energies=energies, pca_variance_ratios=pca_variance_ratios, iter_map=iter_map)


    def save_PCA_objects(self, filename: str, b_pca: PCA, s_pca: PCA):
        '''Save final trained PCA objects as an uncompressed .npz archive.
        
        These can then be used to retrieve principal components of other
        databases, using the PCA fitting done here.
        
        Args:
            filename: Filename to save archive to.
            b_pca: Trained bonding PCA object from fit_PCAs().
            s_pca: Trained structural PCA object from fit_PCAs().
            '''
        np.savez(filename, b_pca=b_pca, s_pca=s_pca)

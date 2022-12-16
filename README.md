# G-SchNetOE62 
Implementation of a method which looks to iteratively build molecules with an increasingly smaller HOMO-LUMO gap using a technique biasing in the training of the generative model G-SchNet. The full acompanying pre print can be found here: https://arxiv.org/pdf/2207.01476.pdf.
For tutorials and installation on how to set up G-SchNet, please see the original G-SchNet repository: from https://github.com/atomistic-machine-learning/G-SchNet. A full tutorial, including code, on how to train and use G-SchNet for the OE62 dataset can be found here: Westermayr, Julia (2022): G-SchNet for OE62. figshare. Dataset. https://doi.org/10.6084/m9.figshare.20146943.v2.

In this code base we only look at using the dataset OE62 with target features of small HOMO-LUMO gap, large LUMO and high HOMO. We provide all the neccesary code to run the generative model and all necessary filtering models and routines. Eventhough we only look at one dataset and a handful of target features G-SchNet should be easily transferable to use a different dataset. A full tutorial on how to adapt G-SchNet in order to train on another dataset can be seen here: https://github.com/atomistic-machine-learning/G-SchNet. Additionally the filtering routines can easily be adapted in order to fit different target features. 

The files loopHL, loopHOMO and loopLUMO are initially used to iteratively train G-SchNet models on OE62 and then generated databases. The biased model and generated databases are placed in directories at each time step. Filtering is also done on every generated database before a biased retraining is conducted in order to remove disconnected molecules, molecules with invalid valencies, duplicate molecules and radicals. In order to create our biased dataset we also train various SchNet and SchNet + H models to predict HOMO and LUMO energies up to a GW level. Molecules are selected for retraining based on the desired characteristic. In addition the analysis of the databases is done automatically and put into an analysis folder at each iteration. The analysis includes distributions of bond distances, ring sizes, molecule sizes and angle values however additional analysis can easily be implemented at each iteration.

For independently retraining G-SchNet on the OE62 data set, use this code and follow the tutorial, but replace the QM9 data set with the OE62 data set (https://www.nature.com/articles/s41597-020-0385-y). 
Note: The adapted script to train OE62 is "template_data", hence add "--datset_name template_data" to the command used for training. 
 

# Citation
If you are using G-SchNet and this code in your research, please cite the corresponding papers:

J. Westermayr, J. Gilkes, R. Barrett, and R. J. Maurer, High-throughput property-driven generative design of functional organic molecules, arXiv:2207.01476, 2022. 

N. Gebauer, M. Gastegger, and K. Schütt. Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, _Advances in Neural Information Processing Systems 32_, pages 7564–7576. Curran Associates, Inc., 2019.

    @incollection{NIPS2019_8974,
    title = {Symmetry-adapted generation of 3d point sets for the targeted discovery of molecules},
    author = {Gebauer, Niklas and Gastegger, Michael and Sch\"{u}tt, Kristof},
    booktitle = {Advances in Neural Information Processing Systems 32},
    editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
    pages = {7566--7578},
    year = {2019},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/8974-symmetry-adapted-generation-of-3d-point-sets-for-the-targeted-discovery-of-molecules.pdf}
    }

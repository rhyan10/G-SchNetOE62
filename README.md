# G-SchNetOE62 
Implementation of a method which looks to iteratively bias molecules with a small HOMO-LUMO gap using the generative model G-SchNet.
For tutorials and installation on how to set up G-SchNet, please see the original G-SchNet repository: from https://github.com/atomistic-machine-learning/G-SchNet. A full tutorial, including code, on how to train and use G-SchNet for the OE62 dataset can be found here: Westermayr, Julia (2022): G-SchNet for OE62. figshare. Dataset. https://doi.org/10.6084/m9.figshare.20146943.v2

For training G-SchNet on the OE62 data set, use this code and follow the tutorial, but replace the QM9 data set with the OE62 data set (https://www.nature.com/articles/s41597-020-0385-y). 
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

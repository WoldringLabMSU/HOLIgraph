# HOLI-GNN Workflow
### 1. Obtain ligand-bound OATP structures for all ligands of interest
###### This may be done with any desired method (e.g., cryo-EM, molecular docking simulations)
- In this work, we docked 222 ligands (listed in `smiles.csv`) to cryo-EM structures of OATP1B1 [1]
- Our docking protocol was performed with  RosettaLigand [2,3] to generate 1000 docked poses for each ligand
  - In our optimized model, only the best (most stable) 30 poses of each ligand were used in subsequent workflow steps.
  - In the provided tutorial files, data is included for only the single best pose of each ligand.
### 2. Extract the raw protein-ligand interaction data 
###### The Protein-Ligand Interaction Profiler (PLIP, https://github.com/pharmai/plip) generates interaction reports as XML files [4]
- This should be performed for all ligands of interest, and for all poses if applicable
- All XML outputs should be located in a single directory and share a common suffix (e.g., `<unique prefix>-1B1in-raw.xml`)
- `Example_XMLs.tar.gz` contains the XML files for the best pose of each 222 ligands in our dataset
### 3. Parse interaction data into Protein-Ligand Interaction Features (PLIFs)
###### The script `PLIP_to_PLIF.py` parses all XML files into a single CSV file of PLIFs and adds class labels from the file `label_ligandID_key.csv`
- The naming convention of PLIFs is as follows: `<interaction type><occurence #>.<feature name>`
  - For example, the feature `hbond2.dist` contains the distance between the two atoms involved in the second hydrogen bond detected in a given pose
  - Possible interaction types: hydrophobic_int, hbond, halogenbond, saltbridge, picat_int, pistack, waterbridge
- The table at the end of this document provides a description of all PLIFs and lists the interaction types to which each applies
### 4. Process PLIF data to prepare for HOLI-graph construction
###### The `preprocess_HOLI-GNN.py` code maps the heterogeneous edge-feature pairs from the PLIF data for each pose
- Dependencies: scikit-learn (https://github.com/scikit-learn/scikit-learn)
### 5. Implement HOLI-GNN
###### Model training, optimization, and testing is done with `train-test.py`, reliant on functions from `HOLI_GNN.py`
- Depedencies:
  - Biopython (https://github.com/biopython)
  - PyTorch (https://github.com/pytorch)
  - RDKit (https://github.com/rdkit/rdkit)
  - scikit-learn (https://github.com/scikit-learn/scikit-learn)
- This script splits the dataset into train, validation, and test sets. For each pose, a HeteroData() graph is constructed.
- Within this script, training and validation occurs to optimize the model for each test set. Following this, the optimized model is evaluated on the test set.

### PLIF Descriptions, Examples, and Applicable Interaction Types
| Feature | Description | Example | Interaction Types |
| :------ | :---------- | :------ | :---------------- |
| resnr | OATP residue number involved in interaction | 356 | ALL |
| restype | OATP residue type involved in interaction | PHE | ALL |
| dist | Distance between interacting atoms (&#x212b;) | 4.7 | hydrophob_int, halogenbond, saltbridge, picat_int |
| dist_h-a | Distance between hydrogen and acceptor atom (&#x212b;) | 3.2 | hbond |
| dist_d-a | Distance between donor and acceptor atom (&#x212b;) | 3.4 | hbond |
| dist_a-w | Distance between water and acceptor atom (&#x212b;) | 4.2 | waterbridge |
| dist_d-w | Distance between water and donor atom (&#x212b;) | 2.9 | waterbridge |
| centdist | Distance between ring centers (&#x212b;) | 5.1 | pistack |
| offset | Offset between interacting groups (&#x212b;) | 1.68 | picat_int, pistack |
| lig_group | Ligand functional group involved in interaction | carboxylate | picat_int, saltbridge | 
| sidechain | Is the sidechain involved? | True/False | hbond, halogenbond, waterbridge |
| protisdon | Is the protein the donor? | True/False | hbond, halogenbond, waterbridge |
| donortype | Atom type of donor | O3 | hbond, halogenbond, waterbridge |
| acceptortype | Atom type of acceptor | N2 | hbond, halogenbond |
| don_angle | Angle at the donor | 108.43 | hbond, halogenbond, waterbridge |
| acc_angle | Angle at the acceptor | 123.90 | halogenbond |
| water_angle | Angle at interacting water | 123.90 | waterbridge |
| angle | Angle between ring planes | 174 | pistack |
| type | Pi-stacking type (p=perpendicular, t=T-shape) | p/t | pistack |
| protispos | Does the protein carry the positive charge? | True/False | saltbridge |
| protcharged | Does the protein provide the charge? | True/False | picat_int |
| donoridx | Atom ID of donor atom | 10280 | hbond, halogenbond, waterbridge |
| acceptoridx | Atom ID of acceptor atom | 4432 | hbond, halogenbond |
| water_idx | Atom ID of water oxygen atom | 9028 | waterbridge |
| lig_idx_list.idx[#] | Atom IDs from ligand functional group | 18 | saltbridge, picat_int, pistack |
| prot_idx_list.idx[#] | Atom IDs from protein functional group | 10221 | saltbridge, picat_int, pistack |
| ligcarbonidx | Atom ID of interacting ligand carbon atom | 12 | hydrophob_int |
| protcarbonidx | Atom ID of interacting protein carbon atom | 9762 | hydrophob_int |

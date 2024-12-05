# HOLIgraph: Heterogeneous OATP-Ligand Interaction Graph Neural Network
###### The HOLIgraph model was designed to predict small molecule inhibitors of the organic anion transporting polypeptide (OATP) 1B1, a key hepatic drug uptake protein. For more context, please refer to **preprint, coming soon!**

## Background
Example files are provided to complete a tutorial on HOLIgraph implementation (steps 3-5 below). 
Preliminary steps (1 and 2 below) will not be covered in detail here, as they will be unique to individual workflows. Existing documentation for these steps is provided in references. 

### 1. Obtain ligand-bound structures
###### Structures may be generated via any desired method.
- __Cryo-EM:__ The example file `E3S-1B1in.pdb` is the cryo-EM structure of estrone-3-sulfate (E3S) bound to the inward-facing OATP1B1 (PDB 8HND) [1]    
- __Docking Simulations:__ The example file `Erlotinib-1B1in.pdb` is a computationally generated pose of erlotinib bound to the inward-facing OATP1B1, simulated using Rosetta [2,3]

### 2. Extract raw protein-ligand interaction data
###### Performed with the Protein-Ligand Interaction Profiler (PLIP, https://github.com/pharmai/plip) [4]
- Specify the PLIP output format to be an XML file. This will be used in subsequent workflow steps.
- Example files `E3S-1B1in-raw.xml` and `Erlotinib-1B1in-raw.xml` were generated with PLIP.


## Usage
NOTE: In all provided example scripts, lines that must be updated with user-specific information are tagged with</i> <b>`## CHANGE`</b>

### 3. Parse raw interaction data into PLIFs & add class labels
###### Use the python script `PLIP_to_PLIF.py` to convert raw PLIP outputs for multiple ligands into PLIFs with labeled classes.
- <b>Inputs:</b> 
    - **Raw PLIP XML files** must all be located in the same base directory, and must be named with the same suffix which users may redefine in this script as needed. A set of example XML files is provided for this tutorial in `Example_XMLs`.
    - **Ligand labeling key** in the form of a CSV file to be used for labeling the PLIF dataset. For the tutorial, use `label_ligandID_key.csv`
- <b>Output:</b> CSV file, for example `1B1in-PLIF.csv`

### 4. Map PLIFs to the HOLI-graph edges and corresponding edge features
###### Use the python script `preprocess_HOLI-GNN.py` to prepare PLIF data for HOLI-graph construction.
- <b>Input:</b>  CSV file containing labeled PLIF data (generated in Step 3).
- <b>Output:</b> PKL file, for example `1B1in-HOLI-features.pkl`

### 5. Implement HOLIgraph
###### Use the python script `train-test.py`, dependent on functions from `HOLI_GNN.py`, to construct graphs, train and optimize the model, and evaluate model performance on test sets.
- <b>Inputs:</b>
    - **Labeled PLIF data** from the CSV file generated in Step 3.
    - **HOLI-graph edge/feature mapping** from the PKL file generated in Step 4.
    - **Test set compositions** defined in a csv, such as `test-sets.csv`
    - **Ligand SMILES** defined in a csv, such as `smiles.csv`
- <b>Outputs</b>
    - **Best model is output following optimization** for each test set. For example, `best_model_testset1.pt`
    - **Performance metrics** for each test set output as a single CSV such as `1B1in-HOLI-scores.csv`
    - **Per-ligand predictions** made for all validation and test sets in a single CSV such as `1B1in-HOLI-predictions.csv`
 
---
## References
<p style="text-indent: -20px; margin-left: 20px;">
[1] Z. Shan, et al., Cryo-EM structures of human organic anion transporting polypeptide OATP1B1. Cell Res 33, 940–951 (2023).
</p>
<p style="text-indent: -20px; margin-left: 20px;">
[2] K. W. Kaufmann, J. Meiler, Using RosettaLigand for Small Molecule Docking into Comparative Models. PLoS One 7 (2012).
</p>
<p style="text-indent: -20px; margin-left: 20px;">
[3] J. Meiler, D. Baker, ROSETTALIGAND: Protein-small molecule docking with full side-chain flexibility. Proteins: Structure, Function and Genetics 65, 538–548 (2006).
</p>
<p style="text-indent: -20px; margin-left: 20px;">
[4] M. F. Adasme, et al., PLIP 2021: Expanding the scope of the protein-ligand interaction profiler to DNA and RNA. Nucleic Acids Res 49, W530–W534 (2021).
</p>

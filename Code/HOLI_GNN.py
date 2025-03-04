
import torch
from torch_geometric.data import HeteroData, DataLoader, Batch
from rdkit import Chem
import numpy as np
import pandas as pd 
from torch.nn import Sequential, ReLU,SELU, Linear, Dropout, BatchNorm1d
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from torch_geometric.nn import GATConv, HeteroConv, global_mean_pool,NNConv
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from Bio.SeqUtils.ProtParam import ProteinAnalysis



# One-hot encoding function
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    return [int(x == s) for s in permitted_list]

# Get atom features
def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 
        'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 
        'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 
        'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]
    if not hydrogens_implicit:
        permitted_list_of_atoms.insert(0, 'H')
    
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    atomic_mass_scaled = [(atom.GetMass() - 10.812) / 116.092]
    vdw_radius_scaled = [(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6]
    covalent_radius_scaled = [(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76]
    
    atom_feature_vector = (
        atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + 
        hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + 
        atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
    )
    
    if use_chirality:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit:
        n_hydrogens_enc = one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)

# Get bond features
def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    
    return np.array(bond_feature_vector)

# Get amino acid features
def get_aa_features(aa_char):
    analysis = ProteinAnalysis(aa_char)
    one_hot = [1 if k == aa_char else 0 for k in three_to_one.values()]
    properties = [
        analysis.molecular_weight(),
        analysis.aromaticity(),
        analysis.isoelectric_point(),
        hydrophobicity_scale[aa_char],
        flexibility_scale[aa_char],
        *analysis.secondary_structure_fraction()
    ]
    properties.extend(one_hot)
    return torch.tensor(properties, dtype=torch.float)

# Hydrophobicity and flexibility scales
hydrophobicity_scale = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5,
    'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
    'W': -0.9, 'Y': -1.3
}

flexibility_scale = {
    'A': 0.360, 'C': 0.310, 'D': 0.510, 'E': 0.500, 'F': 0.310,
    'G': 0.540, 'H': 0.320, 'I': 0.460, 'K': 0.470, 'L': 0.370,
    'M': 0.295, 'N': 0.460, 'P': 0.510, 'Q': 0.490, 'R': 0.530,
    'S': 0.510, 'T': 0.440, 'V': 0.390, 'W': 0.310, 'Y': 0.420
}

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

protein_sequence = 'MDQNQHLNKTAEAQPSENKKTRYCNGLKMFLAALSLSFIAKTLGAIIMKSSIIHIERRFEISSSLVGFIDGSFEIGNLLVIVFVSYFGSKLHRPKLIGIGCFIMGIGGVLTALPHFFMGYYRYSKETNINSSENSTSTLSTCLINQILSLNRASPEIVGKGCLKESGSYMWIYVFMGNMLRGIGETPIVPLGLSYIDDFAKEGHSSLYLGILNAIAMIGPIIGFTLGSLFSKMYVDIGYVDLSTIRITPTDSRWVGAWWLNFLVSGLFSIISSIPFFFLPQTPNKPQKERKASLSLHVLETNDEKDQTANLTNQGKNITKNVTGFFQSFKSILTNPLYVMFVLLTLLQVSSYIGAFTYVFKYVEQQYGQPSSKANILLGVITIPIFASGMFLGGYIIKKFKLNTVGIAKFSCFTAVMSLSFYLLYFFILCENKSVAGLTMTYDGNNPVTSHRDVPLSYCNSDCNCDESQWEPVCGNNGITYISPCLAGCKSSSGNKKPIVFYNCSCLEVTGLQNRNYSAHLGECPRDDACTRKFYFFVAIQVLNLFFSALGGTSHVMLIVKIVQPELKSLALGFHSMVIRALGGILAPIYFGALIDTTCIKWSTNNCGTRGSCRTYNSTSFSRVYLGLSSMLRVSSLVLYIILIYAMKKKYQEKDINASENGSVMDEANLESLNKNKHFVPSAGADSETHC'

def process_molecule(row, interactions, protein_sequence, get_atom_features, get_aa_features, get_bond_features):

    molecule = Chem.MolFromSmiles(row['SMILES'])
    if not molecule:
        print(f"Invalid SMILES: {row['SMILES']}")
        return None
            
    atom_id_to_new_index = {}
    aa_id_to_new_index = {}
    atom_features = []
    interaction_edge_indices = []
    interaction_edge_features = []
    aa_features = []

    # Separate edge lists for aa to atom and atom to aa
    aa_to_atom_edge_indices = []
    aa_to_atom_edge_features = []
    atom_to_aa_edge_indices = []
    atom_to_aa_edge_features = []

    for (aa_idx, atom_idx), features in interactions.items():

        if atom_idx not in atom_id_to_new_index and atom_idx < molecule.GetNumAtoms():
            atom = molecule.GetAtomWithIdx(atom_idx)
            atom_feature = get_atom_features(atom)
            atom_features.append(torch.tensor(atom_feature, dtype=torch.float))
            atom_id_to_new_index[atom_idx] = len(atom_features) - 1
        
        if aa_idx not in aa_id_to_new_index and aa_idx < len(protein_sequence):
            aa_char = protein_sequence[aa_idx]
            aa_feature = get_aa_features(aa_char)
            aa_features.append(torch.tensor(aa_feature, dtype=torch.float))
            aa_id_to_new_index[aa_idx] = len(aa_features) - 1
        
        if atom_idx in atom_id_to_new_index and aa_idx in aa_id_to_new_index:
            new_atom_index = atom_id_to_new_index[atom_idx]
            new_aa_index = aa_id_to_new_index[aa_idx]
            aa_to_atom_edge_indices.append([new_aa_index, new_atom_index])
            aa_to_atom_edge_features.append(torch.tensor(features, dtype=torch.float))
            # Add the reverse edge
            atom_to_aa_edge_indices.append([new_atom_index, new_aa_index])
            atom_to_aa_edge_features.append(torch.tensor(features, dtype=torch.float))

    if len(atom_features) == 0 or len(aa_features) == 0 or (len(aa_to_atom_edge_indices) == 0 and len(atom_to_aa_edge_indices) == 0):
        # print(f"No valid features for SMILES: {row['SMILES']}")
        return None

    bond_edge_indices = []
    bond_edge_features = []
    
    for bond in molecule.GetBonds():
        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        
        if start_atom_idx not in atom_id_to_new_index:
            atom = molecule.GetAtomWithIdx(start_atom_idx)
            atom_feature = get_atom_features(atom)
            atom_features.append(torch.tensor(atom_feature, dtype=torch.float))
            atom_id_to_new_index[start_atom_idx] = len(atom_features) - 1
        
        if end_atom_idx not in atom_id_to_new_index:
            atom = molecule.GetAtomWithIdx(end_atom_idx)
            atom_feature = get_atom_features(atom)
            atom_features.append(torch.tensor(atom_feature, dtype=torch.float))
            atom_id_to_new_index[end_atom_idx] = len(atom_features) - 1


        start_atom_new_idx = atom_id_to_new_index[start_atom_idx]
        end_atom_new_idx = atom_id_to_new_index[end_atom_idx]
        bond_edge_indices.append([start_atom_new_idx, end_atom_new_idx])
        bond_feature = get_bond_features(bond)
        bond_edge_features.append(torch.tensor(bond_feature, dtype=torch.float))
        # Add the reverse edge
        bond_edge_indices.append([end_atom_new_idx, start_atom_new_idx])
        bond_edge_features.append(torch.tensor(bond_feature, dtype=torch.float))

    data = HeteroData()
    if atom_features:
        data['atom'].x = torch.stack(atom_features, dim=0)
    if aa_features:
        data['aa'].x = torch.stack(aa_features, dim=0)
    if aa_to_atom_edge_indices:
        data['aa', 'interacts_with', 'atom'].edge_index = torch.tensor(aa_to_atom_edge_indices, dtype=torch.long).t().contiguous()
        data['aa', 'interacts_with', 'atom'].edge_attr = torch.stack(aa_to_atom_edge_features)
    if atom_to_aa_edge_indices:
        data['atom', 'interacts_with', 'aa'].edge_index = torch.tensor(atom_to_aa_edge_indices, dtype=torch.long).t().contiguous()
        data['atom', 'interacts_with', 'aa'].edge_attr = torch.stack(atom_to_aa_edge_features)
    if bond_edge_indices:
        data['atom', 'bonded_to', 'atom'].edge_index = torch.tensor(bond_edge_indices, dtype=torch.long).t().contiguous()
        data['atom', 'bonded_to', 'atom'].edge_attr = torch.stack(bond_edge_features)

    data.y = torch.tensor(row['Class'], dtype=torch.float)  # Ensure y is a float
    return data

def get_max_feature_dims(graphs, ligand_only=False):
    ''' Determines the number of dimensions to be defined in the 
        constructed heterogeneous graph objects. '''
    if not ligand_only:
        node_feature_dims = {}
        edge_feature_dims = {}
        for data in graphs:
            for key in data.x_dict.keys():
                dim = data.x_dict[key].shape[1]
                node_feature_dims[key] = max(node_feature_dims.get(key, 0), dim)
            for key in data.edge_attr_dict.keys():
                dim = data.edge_attr_dict[key].shape[1]
                edge_feature_dims[key] = max(edge_feature_dims.get(key, 0), dim)
    else:
        node_feature_dims_list = []
        edge_feature_dims_list = []
        for data in graphs:
            node_feature_dims_list.append(data.x.shape[1])
            try:
                edge_feature_dims_list.append(data.edge_attr.shape[1])
            except IndexError:
                continue
        node_feature_dims = max(node_feature_dims_list)
        edge_feature_dims = max(edge_feature_dims_list)
        
    return node_feature_dims, edge_feature_dims

class HeteroGNN(torch.nn.Module):
    def __init__(self, atom_features_dim, aa_features_dim, edge_feature_dims, dim, common_dim, dropout):
        super(HeteroGNN, self).__init__()

        self.atom_features_dim = atom_features_dim
        self.aa_features_dim = aa_features_dim

        # MLP for aa to atom interactions
        self.edge_mlp_aa_to_atom = Sequential(
            Linear(edge_feature_dims[('aa', 'interacts_with', 'atom')], dim),
            ReLU(),
            Linear(dim, aa_features_dim * atom_features_dim)
        )
        
        # MLP for atom to aa interactions
        self.edge_mlp_atom_to_aa = Sequential(
            Linear(edge_feature_dims[('atom', 'interacts_with', 'aa')], dim),
            ReLU(),
            Linear(dim, atom_features_dim * aa_features_dim)
        )
        
        # MLP for atom to atom bonds
        self.edge_mlp_atom_to_atom = Sequential(
            Linear(edge_feature_dims[('atom', 'bonded_to', 'atom')], dim),
            ReLU(),
            Linear(dim, atom_features_dim * atom_features_dim)
        )
        
        # NNConv layer for aa to atom interactions
        self.conv_aa_to_atom = NNConv(
            in_channels=(aa_features_dim, atom_features_dim), 
            out_channels=atom_features_dim, 
            nn=self.edge_mlp_aa_to_atom, 
            aggr='mean'
        )

        # NNConv layer for atom to aa interactions
        self.conv_atom_to_aa = NNConv(
            in_channels=(atom_features_dim, aa_features_dim), 
            out_channels=aa_features_dim, 
            nn=self.edge_mlp_atom_to_aa, 
            aggr='mean'
        )
        
        # NNConv layer for atom to atom bonds
        self.conv_atom_to_atom = NNConv(
            in_channels=(atom_features_dim, atom_features_dim), 
            out_channels=atom_features_dim, 
            nn=self.edge_mlp_atom_to_atom, 
            aggr='mean'
        )
     
        # HeteroConv to combine all types of interactions
        self.hetero_conv = HeteroConv({
            ('aa', 'interacts_with', 'atom'): self.conv_aa_to_atom,
            ('atom', 'interacts_with', 'aa'): self.conv_atom_to_aa,
            ('atom', 'bonded_to', 'atom'): self.conv_atom_to_atom
        }, aggr='mean')

        # Linear layers to project node features to a common dimension
        self.atom_projector = Linear(atom_features_dim, common_dim)
        self.aa_projector = Linear(aa_features_dim, common_dim)

        # Graph classifier
        self.graph_classifier = Sequential(
            Linear(common_dim * 2, common_dim),  # common_dim * 2 because we concatenate atom and aa features
            ReLU(),
            Dropout(dropout),
            Linear(common_dim, 1)  # Output a single logit
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict['aa'] = x_dict['aa'].float().to(self.device)
        x_dict['atom'] = x_dict['atom'].float().to(self.device)

        # Forward pass through HeteroConv
        x_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

        # Debugging: Print keys and shapes
       # print(f"x_dict keys after HeteroConv: {x_dict.keys()}")
        #if 'aa' in x_dict:
         #   print(f"AA feature shape after HeteroConv: {x_dict['aa'].shape} on device {x_dict['aa'].device}")
        #print(f"Atom feature shape after HeteroConv: {x_dict['atom'].shape} on device {x_dict['atom'].device}")

        # Project node features to a common dimension
        x_dict['atom'] = self.atom_projector(x_dict['atom'])
        x_dict['aa'] = self.aa_projector(x_dict['aa'])

        # Pooling separately for 'atom' and 'aa' nodes
        atom_pooled = global_mean_pool(x_dict['atom'], torch.zeros(x_dict['atom'].size(0), dtype=torch.long, device=x_dict['atom'].device))
        aa_pooled = global_mean_pool(x_dict['aa'], torch.zeros(x_dict['aa'].size(0), dtype=torch.long, device=x_dict['aa'].device))

        # Combine the pooled features
        x = torch.cat([atom_pooled, aa_pooled], dim=1)

        out = self.graph_classifier(x)
    
        return out.view(-1)

    @property
    def device(self):
        return next(self.parameters()).device
# HeteroGNN2 model with GATConv
class HeteroGNN2(torch.nn.Module):
    def __init__(self, atom_features_dim, aa_features_dim, edge_feature_dims, heads=8, common_dim=128, dropout=0.4):
        super(HeteroGNN2, self).__init__()

        self.atom_features_dim = atom_features_dim
        self.aa_features_dim = aa_features_dim
        self.heads = heads
        self.common_dim = common_dim
        self.dropout = dropout

        # GATConv layer for aa to atom interactions
        self.gat_aa_to_atom = GATConv((aa_features_dim, atom_features_dim), 
                                      atom_features_dim, 
                                      heads=heads, 
                                      dropout=dropout, 
                                      concat=False,
                                      add_self_loops=False)  # Set add_self_loops=False
        
        # GATConv layer for atom to aa interactions
        self.gat_atom_to_aa = GATConv((atom_features_dim, aa_features_dim), 
                                      aa_features_dim, 
                                      heads=heads, 
                                      dropout=dropout, 
                                      concat=False,
                                      add_self_loops=False)  # Set add_self_loops=False
        
        # GATConv layer for atom to atom bonds
        self.gat_atom_to_atom = GATConv((atom_features_dim, atom_features_dim), 
                                        atom_features_dim, 
                                        heads=heads, 
                                        dropout=dropout, 
                                        concat=False,
                                        add_self_loops=False)  # Set add_self_loops=False

        # HeteroConv to combine all types of interactions
        self.hetero_conv = HeteroConv({
            ('aa', 'interacts_with', 'atom'): self.gat_aa_to_atom,
            ('atom', 'interacts_with', 'aa'): self.gat_atom_to_aa,
            ('atom', 'bonded_to', 'atom'): self.gat_atom_to_atom
        }, aggr='mean')

        # Linear layers to project node features to a common dimension
        self.atom_projector = Linear(atom_features_dim, common_dim)
        self.aa_projector = Linear(aa_features_dim, common_dim)

        # Graph classifier
        self.graph_classifier = Sequential(
            Linear(common_dim * 2, 64),  # common_dim * 2 because we concatenate atom and aa features
            SELU(),
            Dropout(dropout),
            Linear(64, 1)  # Output a single logit
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict['aa'] = x_dict['aa'].float()
        x_dict['atom'] = x_dict['atom'].float()

        # Forward pass through HeteroConv
        x_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

    
        # Project node features to a common dimension
        x_dict['atom'] = self.atom_projector(x_dict['atom'])
        if 'aa' in x_dict:
            x_dict['aa'] = self.aa_projector(x_dict['aa'])

        # Pooling separately for 'atom' and 'aa' nodes
        atom_pooled = global_mean_pool(x_dict['atom'], torch.zeros(x_dict['atom'].size(0), dtype=torch.long))
        
        # Check if 'aa' key exists in x_dict before pooling
        if 'aa' in x_dict:
            aa_pooled = global_mean_pool(x_dict['aa'], torch.zeros(x_dict['aa'].size(0), dtype=torch.long))
            # Combine the pooled features
            x = torch.cat([atom_pooled, aa_pooled], dim=1)
            print(f"Combined feature shape: {x.shape}")
        else:
            print("Warning: 'aa' node type not found in x_dict after HeteroConv")
            x = atom_pooled  # Use only atom features if aa features are not available
            print(f"Atom feature shape: {x.shape}")

        out = self.graph_classifier(x)
        print(f"Output shape: {out.shape}")
        return out.view(-1)
    
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Dropout, SELU, LayerNorm, ReLU
from torch_geometric.nn import NNConv, HeteroConv, global_mean_pool, GATConv
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Dropout, SELU, LayerNorm, ReLU
from torch_geometric.nn import NNConv, HeteroConv, global_mean_pool, GATConv

class HeteroGNN3(torch.nn.Module):
    def __init__(self, atom_features_dim, aa_features_dim, edge_feature_dims, dim, common_dim, dropout):
        super(HeteroGNN3, self).__init__()

        self.atom_features_dim = atom_features_dim
        self.aa_features_dim = aa_features_dim

        # MLP for aa to atom interactions
        self.edge_mlp_aa_to_atom = Sequential(
            Linear(edge_feature_dims[('aa', 'interacts_with', 'atom')], dim),
            ReLU(),
            LayerNorm(dim),
            Linear(dim, dim),
            ReLU(),
            Linear(dim, aa_features_dim * atom_features_dim)
        )

        # MLP for atom to aa interactions
        self.edge_mlp_atom_to_aa = Sequential(
            Linear(edge_feature_dims[('atom', 'interacts_with', 'aa')], dim),
            ReLU(),
            LayerNorm(dim),
            Linear(dim, dim),
            ReLU(),
            Linear(dim, atom_features_dim * aa_features_dim)
        )

        # MLP for atom to atom bonds
        self.edge_mlp_atom_to_atom = Sequential(
            Linear(edge_feature_dims[('atom', 'bonded_to', 'atom')], dim),
            ReLU(),
            LayerNorm(dim),
            Linear(dim, dim),
            ReLU(),
            Linear(dim, atom_features_dim * atom_features_dim)
        )

        # NNConv layer for aa to atom interactions
        self.conv_aa_to_atom = NNConv(
            in_channels=(aa_features_dim, atom_features_dim),
            out_channels=atom_features_dim,
            nn=self.edge_mlp_aa_to_atom,
            aggr='mean'
        )

        # NNConv layer for atom to aa interactions
        self.conv_atom_to_aa = NNConv(
            in_channels=(atom_features_dim, aa_features_dim),
            out_channels=aa_features_dim,
            nn=self.edge_mlp_atom_to_aa,
            aggr='mean'
        )

        # NNConv layer for atom to atom bonds
        self.conv_atom_to_atom = NNConv(
            in_channels=(atom_features_dim, atom_features_dim),
            out_channels=atom_features_dim,
            nn=self.edge_mlp_atom_to_atom,
            aggr='mean'
        )

        # HeteroConv to combine all types of interactions
        self.hetero_conv = HeteroConv({
            ('aa', 'interacts_with', 'atom'): self.conv_aa_to_atom,
            ('atom', 'interacts_with', 'aa'): self.conv_atom_to_aa,
            ('atom', 'bonded_to', 'atom'): self.conv_atom_to_atom
        }, aggr='mean')

        # Linear layers to project node features to a common dimension
        self.atom_projector = Sequential(
            Linear(atom_features_dim, common_dim),
            ReLU(),
            LayerNorm(common_dim)
        )
        self.aa_projector = Sequential(
            Linear(aa_features_dim, common_dim),
            ReLU(),
            LayerNorm(common_dim)
        )

        # Attention layers to enhance feature interactions
        self.attention_aa = GATConv(common_dim, common_dim, heads=4, concat=False, dropout=dropout)
        self.attention_atom = GATConv(common_dim, common_dim, heads=4, concat=False, dropout=dropout)

        # Graph classifier
        self.graph_classifier = Sequential(
            Linear(common_dim * 2, common_dim * 2),  # Increased capacity
            SELU(),
            Dropout(dropout),
            Linear(common_dim * 2, common_dim),
            SELU(),
            Dropout(dropout),
            Linear(common_dim, 1)  # Output a single logit
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict['aa'] = x_dict['aa'].float()
        x_dict['atom'] = x_dict['atom'].float()

        # Forward pass through HeteroConv
        x_dict = self.hetero_conv(x_dict, edge_index_dict, edge_attr_dict)

 
        # Project node features to a common dimension
        x_dict['atom'] = self.atom_projector(x_dict['atom'])
        x_dict['aa'] = self.aa_projector(x_dict['aa'])

        # Apply attention layers
        x_dict['aa'] = self.attention_aa(x_dict['aa'], edge_index_dict[('aa', 'interacts_with', 'atom')])
        x_dict['atom'] = self.attention_atom(x_dict['atom'], edge_index_dict[('atom', 'bonded_to', 'atom')])

        # Pooling separately for 'atom' and 'aa' nodes
        atom_pooled = global_mean_pool(x_dict['atom'], torch.zeros(x_dict['atom'].size(0), dtype=torch.long))
        aa_pooled = global_mean_pool(x_dict['aa'], torch.zeros(x_dict['aa'].size(0), dtype=torch.long))
        
        # Combine the pooled features
        x = torch.cat([atom_pooled, aa_pooled], dim=1)
        print(f"Combined feature shape: {x.shape}")

        out = self.graph_classifier(x)
        print(f"Output shape: {out.shape}")
        return out.view(-1)


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_loader.dataset)


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            if 'atom' in data.x_dict and 'aa' in data.x_dict:
                if data['atom'].x.shape[0] > 0 and data['aa'].x.shape[0] > 0:
                    logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
                    
                    # Apply sigmoid to logits
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long()

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(data.y.cpu().numpy())
                else:
                    print("Empty atom or AA features encountered.")
            else:
                print("Expected keys not found in data.x_dict")

    if len(all_preds) == 0 or len(all_labels) == 0:
        print("Warning: No predictions made or no labels found.")
        return 0, 0, 0, 0, 0  # Return zeros to avoid further errors
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("Predictions:", all_preds)
    print("Labels:", all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return accuracy, f1, precision, recall, balanced_acc

def create_datasets(df, df_test_sets, iteration, inter_edge_dict):
    test_ids = df_test_sets.iloc[iteration, :]
    test_set = df[df['ligandID'].isin(test_ids)]
    train_set = df[~df['ligandID'].isin(test_ids)]
    train_set, val_set = train_test_split(train_set, test_size=0.1, stratify=train_set['Class'], random_state=42)
    
    # Filter out rows with missing features
    train_graphs = []
    for idx, row in train_set.iterrows():
        interactions = inter_edge_dict.get(idx, {})
        data = process_molecule(row, interactions, protein_sequence, get_atom_features, get_aa_features, get_bond_features)
        if data:
            data.y = torch.tensor(row['Class'], dtype=torch.float)  # Ensure y is a float
            train_graphs.append(data)

    val_graphs = []
    for idx, row in val_set.iterrows():
        interactions = inter_edge_dict.get(idx, {})
        data = process_molecule(row, interactions, protein_sequence, get_atom_features, get_aa_features, get_bond_features)
        if data:
            data.y = torch.tensor(row['Class'], dtype=torch.float)  # Ensure y is a float
            val_graphs.append(data)

    test_graphs = []
    for idx, row in test_set.iterrows():
        interactions = inter_edge_dict.get(idx, {})
        data = process_molecule(row, interactions, protein_sequence, get_atom_features, get_aa_features, get_bond_features)
        if data:
            data.y = torch.tensor(row['Class'], dtype=torch.float)  # Ensure y is a float
            test_graphs.append(data)

    return train_graphs, val_graphs, test_graphs

def create_datasets2(df, df_test_sets, iteration, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence):
    print(f'df shape = {df.shape}')
    print(f'df_test_sets shape = {df_test_sets.shape}')
    print(f'inter_edge_dict len = {len(inter_edge_dict)}')
    test_ids = df_test_sets.iloc[iteration, :]
    test_set = df[df['ligandID'].isin(test_ids)]
    
    remaining_set = df[~df['ligandID'].isin(test_ids)]
    ligand_names = remaining_set['ligandID'].unique()
    
    # Split ligand names into train and validation sets
    train_ligands, val_ligands = train_test_split(ligand_names, test_size=0.1, random_state=42)
    
    train_set = remaining_set[remaining_set['ligandID'].isin(train_ligands)]
    val_set = remaining_set[remaining_set['ligandID'].isin(val_ligands)]

    # Generate data for train, validation, and test sets
    train_graphs = [process_and_move_data(row, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence) for idx, row in train_set.iterrows()]
    val_graphs = [process_and_move_data(row, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence) for idx, row in val_set.iterrows()]
    test_graphs = [process_and_move_data(row, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence) for idx, row in test_set.iterrows()]

    print(f'Shape train graph df: {len(train_graphs)}')
    print(f'Shape val graph df: {len(val_graphs)}')
    print(f'Shape test graph df: {len(test_graphs)}')
    
    # Filter out None entries if processing failed
    train_graphs = [data for data in train_graphs if data is not None]
    val_graphs = [data for data in val_graphs if data is not None]
    test_graphs = [data for data in test_graphs if data is not None]

 

    return train_graphs, val_graphs, test_graphs

def create_datasets2_ligand_only(df, df_test_sets, iteration, device, process_molecule, get_atom_features, get_bond_features):
    test_ids = df_test_sets.iloc[iteration, :]
    test_set = df[df['ligandID'].isin(test_ids)]
    
    remaining_set = df[~df['ligandID'].isin(test_ids)]
    ligand_names = remaining_set['ligandID'].unique()
    
    train_ligands, val_ligands = train_test_split(ligand_names, test_size=0.1, random_state=42)
    
    train_set = remaining_set[remaining_set['ligandID'].isin(train_ligands)]
    val_set = remaining_set[remaining_set['ligandID'].isin(val_ligands)]

    def process_ligand(row):
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol is None:
                return None
            
            atom_features = get_atom_features(mol)
            bond_features = get_bond_features(mol)
            edge_index = torch.tensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()] +
                                      [(b.GetEndAtomIdx(), b.GetBeginAtomIdx()) for b in mol.GetBonds()],
                                      dtype=torch.long).t().contiguous()
            
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_attr = torch.tensor(bond_features, dtype=torch.float)
            y = torch.tensor([float(row['1B1_class'])], dtype=torch.float)
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        except Exception as e:
            print(f"Error processing ligand {row['ligandID']}: {str(e)}")
            return None

    train_graphs = [process_ligand(row) for _, row in train_set.iterrows()]
    val_graphs = [process_ligand(row) for _, row in val_set.iterrows()]
    test_graphs = [process_ligand(row) for _, row in test_set.iterrows()]

    train_graphs = [data for data in train_graphs if data is not None]
    val_graphs = [data for data in val_graphs if data is not None]
    test_graphs = [data for data in test_graphs if data is not None]
 
    return train_graphs, val_graphs, test_graphs

def process_and_move_data(row, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence):
    interactions = inter_edge_dict.get(row.name, {})
    data = process_molecule(row, interactions, protein_sequence, get_atom_features, get_aa_features, get_bond_features)
    # print(len(data))
    if data:
        data.y = torch.tensor([row['Class']], dtype=torch.float).to(device)  # Convert and move labels to the device
        # Move all other tensor components of the data to the specified device
        data = move_data_to_device(data, device)
        data['ligandID'] = row['ligandID']
    return data 

def move_data_to_device(data, device):
    if data:
        data.x_dict = {key: value.to(device) for key, value in data.x_dict.items()}
        data.edge_index_dict = {key: value.to(device) for key, value in data.edge_index_dict.items()}
        data.edge_attr_dict = {key: value.to(device) for key, value in data.edge_attr_dict.items()}
        data.y = data.y.to(device)
    return data   
import torch.nn as nn
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=2):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        rce = -1 * (labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))
        rce = rce.mean()
        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

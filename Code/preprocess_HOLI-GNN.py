# NOTE: line tags begining with "## CHANGE" should be updated to contain user-specific information 

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def filter_data(df, int_type):
    '''
    Filter out only columns for specified interaction type.
    
    Args:
        df (pd.DataFrame): DataFrame containing all PLIP data
        int_type (str): column prefix indicating interaction type to filter for
        
    Returns:
        filtered_df (pd.DataFrame): DataFrame containing only PLIP data for
        the specified interaction type
    '''
    selected_cols = ['Class', 'ligandID'] ## CHANGE IF NEEDED
    selected_cols += [col for col in df.columns if int_type in col]
    selected_cols_list = list(dict.fromkeys(selected_cols))
    filtered_df = df[selected_cols_list]
    return filtered_df


def int_prefixes(df, int_type):
    '''
    Get the column names for columns pertaining to a single interaction.
    
    Args:
        df (pd.DataFrame): DataFrame containing all PLIP data
        int_occurance (str): column prefix indicating a single interaction
    
    Returns:
        int_cols (list): List of columns pertaining to the specified interaction
    '''
    int_cols = []
    
    for col in df.columns:
        if int_type in col:
            prefix = col.split('.')[0]
            if prefix not in int_cols:
                int_cols.append(prefix)
    return int_cols


def encode_categorical(df, column_names):
    '''
    Encode categorical features.
    
    Args:
        df (pd.DataFrame): DataFrame requiring encoding
        column_names (list): List of columns to be encoded
        
    Returns:
        encoder (OneHotEncoder)
    '''
    unique_values_list = []

    for name in column_names:
        if name in df.columns:
            unique_values = pd.Series(df[name].dropna().unique(), dtype=str)
            unique_values_list.append(unique_values)

    categorical_values = pd.concat(unique_values_list).unique()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(categorical_values.reshape(-1, 1))
    return encoder


def normalize_columns(df, column_names):
    '''
    Normalize numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame requiring encoding
        column_names (list): List of columns to be encoded
        
    Returns:
        scaler (StandardScaler)    
    '''
    numerical_values = np.array([])
    for name in column_names:
        if name in df.columns:
            values = df[name].dropna().values.astype(float)
            numerical_values = np.append(numerical_values, values)

    scaler = StandardScaler()
    scaler.fit(numerical_values.reshape(-1, 1))
    return scaler


def process_interaction_type(df, interaction_prefix, cat_feats, num_feats):
    '''
    Process columns for a specified interaction type into graph edges.
    
    Args:
        df (pd.DataFrame): DataFrame containing all raw PLIF data
        interaction_prefix (str): column prefix indicating interaction type to filter for
        cat_feats (list): categorical features to be encoded; specific to the interaction type
        num_feats (list): numerical features to be normalized; specific to the interaction type
        
    Returns:
        interaction_dict (dict): dictionary mapping PLIFs to edge features and indices
        of graph 
    '''
    int_df = filter_data(df, interaction_prefix)

    column_prefixes = int_prefixes(int_df, interaction_prefix)
    if len(column_prefixes) == 0:
        return {}
    cat_columns_full = [prefix + '.' + cat for prefix in column_prefixes for cat in cat_feats]
    num_columns_full = [prefix + '.' + num for prefix in column_prefixes for num in num_feats]

    categorical_encoder = encode_categorical(int_df, cat_columns_full)
    numerical_scaler = normalize_columns(int_df, num_columns_full)

    interaction_dict = {}

    for index, row in int_df.iterrows():
        edge_index = []
        edge_features = []

        for prefix in column_prefixes:
            resnr = row.get(f'{prefix}.resnr')

            if pd.isna(resnr):
                continue
            elif interaction_prefix == 'hydrophob':
                target_idx = row.get(f'{prefix}.ligcarbonidx')
            elif interaction_prefix == 'halogen':
                target_idx = row.get(f'{prefix}.don_idx')
            elif interaction_prefix == 'hbond':
                target_idx = row.get(f'{prefix}.acceptoridx') if row[f'{prefix}.protisdon'] else row.get(f'{prefix}.donoridx')

            cat_features = []
            for cat in categorical_columns:
                cat_col_full = f'{prefix}.{cat}'
                if cat_col_full in int_df.columns and not pd.isna(row.get(cat_col_full)):
                    encoded_cat = categorical_encoder.transform([[str(row[cat_col_full])]]).flatten()
                    cat_features.append(encoded_cat)
                else:
                    cat_features.append(np.zeros(len(categorical_encoder.categories_[0])))

            num_features = []
            for num in numerical_columns:
                num_col_full = f'{prefix}.{num}'
                if num_col_full in int_df.columns and not pd.isna(row.get(num_col_full)):
                    normalized_num = numerical_scaler.transform([[float(row[num_col_full])]]).flatten()
                    num_features.append(normalized_num)
                else:
                    num_features.append(np.zeros(1))

            features = np.concatenate(cat_features + num_features) if cat_features or num_features else np.array([], dtype=np.float)
            if features.size > 0:
                if interaction_prefix in ['halogen', 'hbond', 'hydrophob']:
                    edge_index.append((int(resnr) - 1, int(target_idx) - 1))
                    edge_features.append(features)
                else:
                    lig_idxs = [row[f'{prefix}.lig_idx_list.idx{i}'] for i in range(1, int_df.columns.str.contains(f'{prefix}.lig_idx_list.idx').sum() + 1) if not pd.isna(row.get(f'{prefix}.lig_idx_list.idx{i}'))] 
                    for lig_idx in lig_idxs:
                        edge_index.append((int(resnr) - 1, int(lig_idx) - 1))
                        edge_features.append(features)                    

        interaction_dict[index] = [edge_index, edge_features]

    return interaction_dict


def get_all_edges(dicts):
    ''' 
    Extract unique edges from list of dictionaries containing edge-feature data.
    '''
    return {edge for dictionary in dicts for edges, _ in dictionary.values() for edge in edges}


def get_feature_lengths(dicts):
    ''' 
    Retrieve length of first non-empty feature in each dictionary.
    '''
    lengths = []
    for dictionary in dicts:
        for edges, features in dictionary.values():
            if features:
                lengths.append(len(features[0]))
                break
    return lengths


def handle_dictionary(d, feature_length, all_edges):
    ''' 
    Create edge-feature dictionary. 
    
    Args:
        d (dict): dictionary mapping PLIF dataframe rows to edge-feature pairs
        feature_length (int): number of features for each edge
        all_edges (set): all possible unique edges
    
    Yields:
        tuple: row index, populated edge-feature dictionary
    '''
    zero_array = np.zeros(feature_length)
    template = {edge: zero_array.copy() for edge in all_edges}
    
    for key, (edges, features) in d.items():
        feature_dict = template.copy()
        for i, edge in enumerate(edges):
            if edge in feature_dict:
                feature_dict[edge] = features[i]
        yield key, feature_dict


def combine_features(dicts, all_edges, dict_lengths):
    '''
    Align features from multiple dictionaries to their corresponding edges.
    
    Args:
        dicts (list): list of dictionaries mapping PLIF dataframe rows to edge-feature pairs
        all_edges (set): all possible unique edges
        dict_lengths (list): list of feature lengths for each dictionary
        
    Returns:
        combined_features (dict): dictionary of all edges and associated features 
    '''
    combined_features = {}
    for d, feature_length in zip(dicts, dict_lengths):
        for key, feature_dict in handle_dictionary(d, feature_length, all_edges):
            if key not in combined_features:
                combined_features[key] = {edge: [] for edge in all_edges}
            for edge, feature_array in feature_dict.items():
                combined_features[key][edge].append(feature_array)

    prune_zero_vectors(combined_features)
    return combined_features


def prune_zero_vectors(combined_features):
    '''
    Remove meaningless edges (i.e., those with no features) 
    
    Args:
        combined_features (dict): dictionary of all edges and associated features 
    '''
    for key in list(combined_features.keys()):
        for edge in list(combined_features[key].keys()):
            if all(np.all(np.isclose(arr, 0)) for arr in combined_features[key][edge]):
                del combined_features[key][edge]
            if not combined_features[key]:
                del combined_features[key]


# Reading plip data from CSV
plip_csv_path = '1B1in-PLIF.csv' ## CHANGE IF NEEDED
df = pd.read_csv(plip_csv_path)
df

# Removing any columns containing features not suitable for graph format
df = df.drop(columns=[col for col in df.columns if 'count' in col])

# Preparing one hot encoder
amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
one_hot_encoder = OneHotEncoder(sparse_output=False, categories=[amino_acids])
one_hot_encoder.fit([[aa] for aa in amino_acids])

categorical_columns_dict = {
    'hydrophob': ['restype'],
    'halogen': ['restype', 'sidechain', 'donortype', 'acceptortype'],
    'hbond': ['restype', 'sidechain', 'protisdon', 'donortype', 'acceptortype'],
    'picat': ['restype', 'protcharged', 'lig_group'],
    'pistack': ['restype', 'type'],
    'salt': ['restype', 'protispos', 'lig_group']
    }

numerical_columns_dict = {
    'hydrophob': ['dist'],
    'halogen': ['dist', 'acc_angle'],
    'hbond': ['dist_h-a', 'dist_d-a', 'don_angle'],
    'picat': ['dist', 'offset'],
    'pistack': ['centdist', 'angle', 'offset'],
    'salt': ['dist']
    }

interaction_dicts = []
for interaction_prefix in categorical_columns_dict.keys():
    categorical_columns = categorical_columns_dict[interaction_prefix]
    numerical_columns = numerical_columns_dict[interaction_prefix]
    
    interaction_dicts.append(process_interaction_type(df, interaction_prefix=interaction_prefix, cat_feats=categorical_columns, num_feats=numerical_columns))

all_edges = get_all_edges(interaction_dicts)
dict_lengths = get_feature_lengths(interaction_dicts)
combined_features = combine_features(interaction_dicts, all_edges, dict_lengths)

prune_zero_vectors(combined_features)
for key, subdict in combined_features.items():
    for edge, values in subdict.items():
        try:
            subdict[edge] = np.concatenate(values)
        except ValueError:
            pass

output_pkl_filepath = '1B1in-HOLI-features.pkl' ## CHANGE IF NEEDED
with open(output_pkl_filepath, 'wb') as f:
    pickle.dump(combined_features, f)


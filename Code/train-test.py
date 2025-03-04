import torch
from torch_geometric.loader import DataLoader
import pandas as pd
import pickle
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from HOLI_GNN import HeteroGNN, get_max_feature_dims, create_datasets2, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence
from torch.nn import BCEWithLogitsLoss
import numpy as np
import os
import random


def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call the function to set the seed before running any other code
set_seed(23)

# Read and prepare data
df = pd.read_csv('1B1in-PLIF.csv') ## CHANGE IF NEEDED

print(f'Shape of df read from input csv: {df.shape}')

with open('1B1in-HOLI-features.pkl', 'rb') as f:
    inter_edge_dict_full = pickle.load(f)
df = df[df.index.isin(inter_edge_dict_full.keys())]
inter_edge_dict = {k: inter_edge_dict_full[k] for k in df.index}
print(f'Number inter_edge features: {len(inter_edge_dict)}')

df2 = pd.read_csv('smiles.csv')
df2['ligandID'] = df2['ligandID']
merged_df = pd.merge(df, df2, on='ligandID', how='left')
df = merged_df

df['SMILES'] = df['SMILES'].astype(str)
df_test_sets = pd.read_csv('test-sets.csv', header=None, index_col=None)
print(f'Test set df shape: {df_test_sets.shape}')

def move_data_to_device(data, device):
    for key in data.x_dict.keys():
        data.x_dict[key] = data.x_dict[key].float().to(device)
    for key in data.edge_index_dict.keys():
        data.edge_index_dict[key] = data.edge_index_dict[key].long().to(device)
    for key in data.edge_attr_dict.keys():
        data.edge_attr_dict[key] = data.edge_attr_dict[key].float().to(device)
    data.y = data.y.float().to(device)
    return data

# Define train, validate, and evaluate functions
def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = move_data_to_device(data, device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss = criterion(output, data.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad(), torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        for data in val_loader:
            data = move_data_to_device(data, device)
            output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def predict_and_evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_ligand_names = []
    
    with torch.no_grad(), torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        for data in loader:
            data = move_data_to_device(data, device)
            output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            preds = torch.sigmoid(output).cpu().numpy().flatten()
            if np.isnan(preds).any():
                print(data['ligandID'])
                print(preds)
            else:
                all_preds.extend(preds)
                all_labels.extend(data.y.cpu().numpy().flatten())
                all_ligand_names.extend(data['ligandID'])
    
    accuracy = accuracy_score(all_labels, np.round(all_preds))
    f1 = f1_score(all_labels, np.round(all_preds), average='macro')
    precision = precision_score(all_labels, np.round(all_preds), average='macro')
    recall = recall_score(all_labels, np.round(all_preds), average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, np.round(all_preds))
    auc = roc_auc_score(all_labels, all_preds)
    
    return accuracy, f1, precision, recall, balanced_acc, auc, all_preds, all_labels, all_ligand_names


# Initialize model and optimize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training, validation, and test 
results_df_list = []
pred_df_list = []

for test_set in range(df_test_sets.shape[0]): 
    print(f'MAKING GRAPHS FOR TEST SET {test_set+1}!')
    
    train_graphs, val_graphs, test_graphs = create_datasets2(df, df_test_sets, test_set, inter_edge_dict, device, process_molecule, get_atom_features, get_aa_features, get_bond_features, protein_sequence)
    print(f'Shape train graph df: {len(train_graphs)}')
    print(f'Shape val graph df: {len(val_graphs)}')
    print(f'Shape test graph df: {len(test_graphs)}')
    
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

    # Get max dimensions for train data
    node_feat_dims, edge_feat_dims = get_max_feature_dims(train_loader)

    model = HeteroGNN(
        atom_features_dim=node_feat_dims['atom'],
        aa_features_dim=node_feat_dims['aa'],
        edge_feature_dims=edge_feat_dims,
        dim=215,
        common_dim=79,
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    criterion = BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print(f'STARTING MODEL OPTIMIZATION FOR TEST SET {test_set+1}!')    
    for epoch in range(100):  ## CHANGE if more/less epochs desired
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_testset{test_set+1}.pt') ## CHANGE IF NEEDED
            patience_counter = 0
        else:
            patience_counter += 1
        
        scheduler.step()
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    print(f'LOADING BEST MODEL FOR TEST SET {test_set+1}!')
    model_path = f'best_model_testset{test_set+1}.pt' ## CHANGE IF NEEDED
    if not os.path.exists(model_path):
        continue

    model.load_state_dict(torch.load(model_path))
    
    print(f'EVALLUATING BEST MODEL FOR TEST SET {test_set+1}!') 
    val_metrics = predict_and_evaluate(model, val_loader, device)
    test_metrics = predict_and_evaluate(model, test_loader, device)
    
    # Save test results
    results_df_list.append({
        'Test Set': test_set + 1,
        'Set': 'Test',
        'Accuracy': test_metrics[0],
        'Precision': test_metrics[2],
        'Recall': test_metrics[3],
        'F1_Score': test_metrics[1],
        'Balanced_Accuracy': test_metrics[4],
        'AUC': test_metrics[5]
    })
    
    # Save predictions
    for set_name, set_metrics in {'Validation': val_metrics, 'Test': test_metrics}.items():
        for i, ligand in enumerate(set_metrics[8]):
            try:
                pred_df_list.append({
                    'Test Set': test_set + 1,
                    'Set': set_name,
                    'ligandID': ligand,
                    'True Label': set_metrics[7][i],
                    'Predicted Label': set_metrics[6][i]
                })
            except ValueError:
                pass
    
    print(f'FINISHED EVALUATIONS FOR TEST SET {test_set+1}!')

results_df = pd.DataFrame(results_df_list)
print(results_df)
results_df.to_csv('1B1in-HOLI-scores.csv', index=False) ## CHANGE IF NEEDED

pred_df = pd.DataFrame(pred_df_list)
print(pred_df)
pred_df.to_csv('1B1in-HOLI-predictions.csv', index=False) ## CHANGE IF NEEDED


#!/usr/bin/env python
# coding: utf-8

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D
import colorsys
import random
import pandas as pd
import json
import hypernetx as hnx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import csv
from pathlib import Path
from tqdm import tqdm
from rdkit import RDLogger
from bs4 import BeautifulSoup
from pathlib import Path
import numpy as np
import dgl
from dgl.nn import HeteroGraphConv, GraphConv
from scipy.sparse import coo_matrix
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    root_mean_squared_error, mean_absolute_error, r2_score, accuracy_score
)
from sklearn.model_selection import train_test_split
import time
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
import csv
from datasets import load_dataset
from sklearn.multioutput import MultiOutputClassifier
# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')  # Suppresses all RDKit warnings

# 1. Load dataset ------------------------------------------------------------------------------------------------------------------------------------------
print("Step 1: Load dataset")

# List of all datasets to process
datasets = [
    ('1', 'BACE', "hf://datasets/scikit-fingerprints/MoleculeNet_BACE/bace.csv"),
    ('2', 'BBBP', "hf://datasets/scikit-fingerprints/MoleculeNet_BBBP/bbbp.csv"),
    ('3', 'ClinTox', "hf://datasets/scikit-fingerprints/MoleculeNet_ClinTox/clintox.csv"),
    ('4', 'ESOL', "hf://datasets/scikit-fingerprints/MoleculeNet_ESOL/esol.csv"),
    ('5', 'FreeSolv', "hf://datasets/scikit-fingerprints/MoleculeNet_FreeSolv/freesolv.csv"),
    ('6', 'HIV', "hf://datasets/scikit-fingerprints/MoleculeNet_HIV/hiv.csv"),
    ('7', 'Lipophilicity', "hf://datasets/scikit-fingerprints/MoleculeNet_Lipophilicity/lipophilicity.csv"),
    ('8', 'SIDER', "hf://datasets/scikit-fingerprints/MoleculeNet_SIDER/sider.csv"),
    ('9', 'Tox21', "hf://datasets/scikit-fingerprints/MoleculeNet_Tox21/tox21.csv")
    # ('10', 'QM7', "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm7.csv")
]

# Feature options to test
feature_options = {
    '1': "graph",
    '2': "text",
    '3': "hypergraph",
    '4': "graph+text",
    '5': "graph+hypergraph",
    '6': "text+hypergraph",
    '7': "graph+text+hypergraph"
}

# Main processing loop
import os
import pandas as pd
from pathlib import Path

# Directory to store saved datasets to avoid redownloading
DATASET_DIR = "./dataset_store"
os.makedirs(DATASET_DIR, exist_ok=True)  # Create cache directory if it doesn't exist

# List of all datasets to process
datasets = [
    ('1', 'BACE', "hf://datasets/scikit-fingerprints/MoleculeNet_BACE/bace.csv"),
    ('2', 'BBBP', "hf://datasets/scikit-fingerprints/MoleculeNet_BBBP/bbbp.csv"),
    ('3', 'ClinTox', "hf://datasets/scikit-fingerprints/MoleculeNet_ClinTox/clintox.csv"),
    ('4', 'ESOL', "hf://datasets/scikit-fingerprints/MoleculeNet_ESOL/esol.csv"),
    ('5', 'FreeSolv', "hf://datasets/scikit-fingerprints/MoleculeNet_FreeSolv/freesolv.csv"),
    ('6', 'HIV', "hf://datasets/scikit-fingerprints/MoleculeNet_HIV/hiv.csv"),
    ('7', 'Lipophilicity', "hf://datasets/scikit-fingerprints/MoleculeNet_Lipophilicity/lipophilicity.csv"),
    ('8', 'SIDER', "hf://datasets/scikit-fingerprints/MoleculeNet_SIDER/sider.csv"),
    ('9', 'Tox21', "hf://datasets/scikit-fingerprints/MoleculeNet_Tox21/tox21.csv")
]
# choice = input('pick a dataset to load: \n')
for dataset_num, dataset_name, dataset_url in datasets:
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Define local cache file path
    cache_file = Path(DATASET_DIR) / f"{dataset_name}.csv"
    
    # Load from cache if available, else download and cache
    if cache_file.exists():
        print(f"Loading {dataset_name} from cache...")
        dataset = pd.read_csv(cache_file)
    else:
        print(f"Downloading {dataset_name}...")
        dataset = pd.read_csv(dataset_url)
        dataset.to_csv(cache_file, index=False)  # Save to cache
        print(f"Saved {dataset_name} to cache.")
    
    # Print class imbalance information (unchanged)
    if dataset_name in ['BACE', 'BBBP', 'ClinTox', 'HIV', 'SIDER', 'Tox21']:
        label_columns = dataset.columns[1:]  # Skip 'SMILES'
        
        print("\nClass Imbalance per Task:")
        for label in label_columns:
            positives = (dataset[label] == 1).sum()
            negatives = (dataset[label] == 0).sum()
            total = len(dataset[label])
            
            print(f"\nTask: {label}")
            print(f"  Positives (1): {positives} ({positives / total:.2%})")
            print(f"  Negatives (0): {negatives} ({negatives / total:.2%})")
    else:
        label_columns = dataset.columns[1:]  # Skip 'SMILES'
        
        print("\nClass Imbalance per Task:")
        for label in label_columns:
            positives = (dataset[label] > 0).sum()
            negatives = (dataset[label] < 0).sum()
            total = len(dataset[label])
            
            print(f"\nTask: {label}")
            print(f"  Positives: {positives} ({positives / total:.2%})")
            print(f"  Negatives: {negatives} ({negatives / total:.2%})")
    
    # 2. SMART extraction ---------------------------------------------------------------------------------------------------------------------------------------
    print("Step 2: Extract and save SMARTS patterns")
    
    def extract_and_save_smarts(html_file, output_json):
        """
        Extract SMARTS patterns from html file (html copied from Tox24) and save to JSON file.
        """
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
    
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        for item in soup.find_all('div', class_='browser-item'):
            smart_name = item.find('div', class_='smart').get_text(strip=True)
            smarts_pattern = item.find('span', class_='smarts').get_text(strip=True)
            
            comment_div = item.find('div', class_='comment')
            endpoint = ""
            
            if comment_div:
                for sibling in comment_div.next_siblings:
                    if sibling.name == 'div' and 'article-data' in sibling.get('class', []):
                        break
                    if isinstance(sibling, str) and sibling.strip():
                        endpoint = sibling.replace('Endpoint:', '').strip()
                        break
            results.append({
                'SMART_Template': smart_name,
                'SMARTS': smarts_pattern,
                'Endpoint': endpoint
            })
    
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {len(results)} SMARTS patterns to {output_json}")
        return results
    
    smarts_file = Path('smarts.json')
    
    if smarts_file.exists():
        try:
            with open(smarts_file, 'r') as f:
                smarts_patterns = json.load(f)
            print(f"Loaded {len(smarts_patterns)} SMARTS patterns from cache")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading SMARTS cache: {str(e)}. Re-extracting...")
    else:
        print("No SMARTS cache found. Extracting new SMARTS...")
        smarts_patterns = extract_and_save_smarts('patterns.html', smarts_file)    
    
    def generate_distinct_colors(n):
        """Generate n visually distinct RGB colors"""
        colors = []
        for i in range(n):
            hue = i * (1.0/max(1,n-1))
            saturation = 0.7 + random.random() * 0.3
            lightness = 0.5 + random.random() * 0.2
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append((r, g, b))
        return colors
    
    def highlight_all_substructures(mol, matches):
        """Create SVG highlighting all substructure matches"""
        if not matches or not mol:
            return None
    
        try:
            AllChem.Compute2DCoords(mol)
        except:
            return None
    
        drawer = rdMolDraw2D.MolDraw2DSVG(800, 600)
        drawer.SetFontSize(0.8)
        colors = generate_distinct_colors(len(matches))
    
        atom_colors = {}
        bond_colors = {}
        radii = {}
    
        for i, match in enumerate(matches):
            for atom_idx in match:
                atom_colors[atom_idx] = colors[i]
                radii[atom_idx] = 0.5
            for bond in mol.GetBonds():
                if bond.GetBeginAtomIdx() in match and bond.GetEndAtomIdx() in match:
                    bond_colors[bond.GetIdx()] = colors[i]
    
        try:
            drawer.DrawMolecule(
                mol,
                highlightAtoms=list(atom_colors.keys()),
                highlightAtomColors=atom_colors,
                highlightBonds=list(bond_colors.keys()),
                highlightBondColors=bond_colors,
                highlightAtomRadii=radii
            )
            drawer.FinishDrawing()
            return drawer.GetDrawingText()
        except Exception as e:
            print(f"Error drawing molecule: {e}")
            return None
    
    def process_molecules(smiles_list, smarts_patterns, dataset_name, output_base="substructure_matches"):
        """Process molecules for one dataset, loading existing matches if available"""
        output_dir = Path(output_base) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths for cached results
        txt_path = output_dir / f"{dataset_name.upper()}_substructures.txt"
        pkl_path = output_dir / f"{dataset_name.upper()}_substructures.pkl"
        
        # Try to load existing results
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    print(f"Loading cached substructures for {dataset_name}")
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached substructures: {e}. Recomputing...")
        
        # If no cache exists, compute new substructures
        substruct_dict = {}
        
        for mol_id, smiles in enumerate(tqdm(smiles_list, desc=f"Processing {dataset_name}")):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    continue
    
                all_matches = []
                for pattern in smarts_patterns:
                    try:
                        patt = Chem.MolFromSmarts(pattern['SMARTS'])
                        if patt:
                            matches = mol.GetSubstructMatches(patt)
                            if matches:
                                all_matches.extend(matches)
                    except:
                        continue
    
                if not all_matches:
                    continue
    
                substructures = set()
                for match in all_matches:
                    try:
                        sub = Chem.MolFragmentToSmiles(mol, atomsToUse=list(match))
                        substructures.add(sub)
                    except:
                        continue
    
                if not substructures:
                    continue
    
                substruct_dict[mol_id] = list(substructures)
    
                # Save SVG visualization
                svg = highlight_all_substructures(mol, all_matches)
                if svg:
                    try:
                        svg_path = output_dir / f"mol_{mol_id}.svg"
                        with open(svg_path, 'w') as f:
                            f.write(svg)
                    except Exception as e:
                        print(f"Could not save SVG for mol {mol_id}: {e}")
    
            except Exception as e:
                print(f"Error processing mol {mol_id}: {e}")
                continue
    
        return substruct_dict
    
    
    def save_results(substruct_dict, dataset_name, output_base="substructure_matches"):
        """Save results to .txt and .pkl"""
        output_dir = Path(output_base) / dataset_name  # Save in dataset subdirectory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        txt_path = output_dir / f"{dataset_name.upper()}_substructures.txt"
        pkl_path = output_dir / f"{dataset_name.upper()}_substructures.pkl"
        
        # Save as JSON
        with open(txt_path, "w") as f:
            f.write(json.dumps(substruct_dict, indent=2))
        
        # Save as pickle (faster for reloading)
        with open(pkl_path, "wb") as f:
            pickle.dump(substruct_dict, f)
        
        print(f"Saved {dataset_name}: {len(substruct_dict)} molecules with matches.")
    
    
    # Usage remains the same
    smiles_list = dataset['SMILES'].dropna().tolist()
    substruct_dict = process_molecules(smiles_list, smarts_patterns, dataset_name)
    save_results(substruct_dict, dataset_name)

# 3. Hypergraph part -----------------------------------------------------------------------------------------------------------------------------------------
    print("Step 3: Create and save hypergraph")
    
    with open(f'./substructure_matches/{dataset_name}/{dataset_name.upper()}_substructures.txt', 'r') as f:
        DICT = json.load(f)
    
    # create hypergraph
    H = hnx.Hypergraph(DICT)
    
    # get list of hyperedges
    hyperedges = list(H.edges())
    
    # limit number of hyperedges to be drawn for neatness and clarity
    max_hyperedges = 13 
    if len(hyperedges) > max_hyperedges:
        sampled_hyperedges = random.sample(hyperedges, max_hyperedges)
    else:
        sampled_hyperedges = hyperedges
    
    # create a subset dictionary of sampled hyperedges
    subset_hypergraph_dict = {k: H.edges[k] for k in sampled_hyperedges}
    
    # create a new hypergraph using the subset
    H_subset = hnx.Hypergraph(subset_hypergraph_dict)
    
    # draw the hypergraph
    hnx.draw(H_subset, layout=nx.spring_layout, layout_kwargs={'k': 0.3, 'iterations': 50}, node_radius=0.5, node_labels_kwargs={'fontsize': 7}, edge_labels_kwargs={'fontsize': 10}, with_node_labels=True, with_edge_labels=True)
    
    plt.title(f"{dataset_name} Molecular Fragments Hypergraph", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # convert to sparse matrix
    LEN = len(DICT)
    nl = coo_matrix((LEN, LEN))
    nl.setdiag(1)
    values = nl.data
    indices = np.vstack((nl.row, nl.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = nl.shape
    nl = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
    # create molecule and chemical substructure lists
    molec_list = []
    chemicalsub_list = []
    molec_chemicalsub = {}
    
    for molec in tqdm(DICT.keys(), desc='Loading dictionary'):
        chemicalsubs = DICT[molec]
        if molec not in molec_list:
            molec_list.append(molec)
        idx = molec_list.index(molec)
        if idx not in molec_chemicalsub:
            molec_chemicalsub[idx] = []
    
        translated_p = []
        for chemicalsub in chemicalsubs:
            if chemicalsub not in chemicalsub_list:
                chemicalsub_list.append(chemicalsub)
            p_idx = chemicalsub_list.index(chemicalsub)
            translated_p.append(p_idx)  # Translate chemicalsub number into index
    
        molec_chemicalsub[idx] = translated_p
    
    chemicalsub_citing = []  # List of [chemicalsub, molec]
    n_chemicalsub = len(chemicalsub_list)
    n_hedge = len(molec_list)
    
    for molec in tqdm(molec_chemicalsub.keys(), desc='Loading molecules'):
        chemicalsubs = molec_chemicalsub[molec]
        for chemicalsub in chemicalsubs:
            chemicalsub_citing.append([chemicalsub, molec])
    
    chemicalsub_molec = torch.LongTensor(chemicalsub_citing)
    data_dict = {
        ('node', 'in', 'edge'): (chemicalsub_molec[:, 0], chemicalsub_molec[:, 1]),
        ('edge', 'con', 'node'): (chemicalsub_molec[:, 1], chemicalsub_molec[:, 0])
    }
    
    # Construct the hypergraph
    lst = []
    for i in tqdm(chemicalsub_citing, desc='Constructing hypergraph'):
        lst.append(i[0])
    s = set(lst)
    s = len(s)
    num_nodes_dict = {'edge': LEN, 'node': s}
    hyG = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    rows=n_chemicalsub
    columns=n_hedge
    
    molec_X=nl
    v_feat=coo_matrix((rows, 128))
    v_feat.setdiag(1)
    values = v_feat.data
    indices = np.vstack((v_feat.row, v_feat.col))
    i = torch.FloatTensor(indices)
    v = torch.FloatTensor(values)
    shape = v_feat.shape
    v_feat=torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
    hyG.ndata['h'] = {'edge' : molec_X.type('torch.FloatTensor'), 'node' : v_feat.type('torch.FloatTensor')}
    e_feat = molec_X.type('torch.FloatTensor')
    v_feat=v_feat.type('torch.FloatTensor')
    
    e_feat = e_feat.to(device)
    v_feat = v_feat.to(device)
    hyG=hyG.to(device)
    
    
    label_columns = dataset.drop(columns=['smiles']) if dataset_name == 'QM7' else dataset.drop(columns=['SMILES'])
    label_columns = label_columns.fillna(0)
    num_class = len(label_columns.columns)
    first_row_tensor = torch.tensor(label_columns.iloc[0].values, dtype=torch.float32)
    print(first_row_tensor)  # torch.Size([num_labels])
    
    
    class hypergraph_NN(nn.Module):
        def __init__(self, i_d, q_d, v_d, e_d, num_class, dropout=0.5):
            super(hypergraph_NN, self).__init__()
            self.dropout = dropout
            self.q_d = q_d
    
            self.first_layer_in = nn.Linear(i_d, v_d)
            self.not_first_layer_in = nn.Linear(v_d, v_d)
    
            self.w1 = nn.Linear(e_d, q_d)
            self.w2 = nn.Linear(v_d, q_d)
            self.w3 = nn.Linear(v_d, e_d)
            self.w4 = nn.Linear(v_d, q_d)
            self.w5 = nn.Linear(e_d, q_d)
            self.w6 = nn.Linear(e_d, v_d)
            self.cls = nn.Linear(e_d, num_class)
        
        def red_function(self, nodes):
            attention_score = F.softmax(nodes.mailbox['Attn'], dim=1)
            aggregated = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
            return {'h': aggregated}
        
        def attention(self, edges):
            attention_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
            c = attention_score / np.sqrt(self.q_d)
            return {'Attn': c}
        
        def msg_function(self, edges):
            return {'v': edges.src['v'], 'Attn': edges.data['Attn']}
    
        def forward(self, hyG, vfeat, efeat, first_layer=True, last_layer=True):
            with hyG.local_scope():
                if first_layer:
                    feat_e = self.first_layer_in(efeat)
                else:
                    feat_e = self.not_first_layer_in(efeat)
                feat_v = vfeat
    
                # Hyperedge attention
                hyG.ndata['h'] = {'edge': feat_e}
                hyG.ndata['k'] = {'edge': self.w5(feat_e)}
                hyG.ndata['v'] = {'edge': self.w6(feat_e)}
                hyG.ndata['q'] = {'node': self.w4(feat_v)}
                hyG.apply_edges(self.attention, etype='con')
                hyG.update_all(self.msg_function, self.red_function, etype='con')
                feat_v = hyG.ndata['h']['node']
    
                # Node attention
                hyG.ndata['k'] = {'node': self.w2(feat_v)}
                hyG.ndata['v'] = {'node': self.w3(feat_v)}
                hyG.ndata['q'] = {'edge': self.w1(feat_e)}
                hyG.apply_edges(self.attention, etype='in')
                hyG.update_all(self.msg_function, self.red_function, etype='in')
                feat_e = hyG.ndata['h']['edge']
    
                if not last_layer:
                    feat_v = F.dropout(feat_v, self.dropout)
                    return feat_v, feat_e  # Return intermediate features
                else:
                    pred = self.cls(feat_e)
                    return pred  # Return only predictions for last layer
    
    patience = 0
    
    # metric calculation functions
    def calculate_regression_metrics(pred, target):
        """Calculate regression metrics with per-target outputs"""
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        metrics = {
            'RMSE': root_mean_squared_error(target, pred),
            'MAE': mean_absolute_error(target, pred),
            'R2': r2_score(target, pred),
        }
        
        return metrics
    
    def calculate_classification_metrics(pred, target):
        """Calculate multilabel classification metrics with per-task outputs"""
        probs = torch.sigmoid(pred).detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        cls = (probs > 0.5).astype(int)
        
        metrics = {
            'Accuracy': accuracy_score(target, cls),
            'ROC-AUC': roc_auc_score(target, probs, average='macro'),
            'Per_Task': []
        }
        
        # Calculate per-task metrics
        for i in range(target.shape[1]):
            try:
                task_metrics = {
                    'Accuracy': accuracy_score(target[:, i], cls[:, i]),
                    'ROC-AUC': roc_auc_score(target[:, i], probs[:, i]) if len(np.unique(target[:, i])) > 1 else float('nan'),
                    'Num_Positive': np.sum(target[:, i]),
                    'Num_Negative': np.sum(target[:, i] == 0)
                }
                metrics['Per_Task'].append(task_metrics)
            except ValueError:
                metrics['Per_Task'].append({
                    'Accuracy': float('nan'),
                    'ROC-AUC': float('nan'),
                    'Num_Positive': 0,
                    'Num_Negative': 0
                })
        
        return metrics
    
    def print_per_task_metrics(metrics, set_name):
        """Print per-task metrics for a given set (train/val) with averages"""
        total_acc = 0
        total_auc = 0
        num_tasks = 0
        
        print(f"\n{set_name} Per-Task Metrics:")
        for i, task_metrics in enumerate(metrics['Per_Task']):
            print(f"  Task {i}:")
            print(f"    Accuracy: {task_metrics['Accuracy']:.4f}")
            print(f"    ROC-AUC: {task_metrics['ROC-AUC']:.4f}")
            print(f"    Num_Positive: {task_metrics['Num_Positive']}")
            print(f"    Num_Negative: {task_metrics['Num_Negative']}")
            
            if not np.isnan(task_metrics['Accuracy']) and not np.isnan(task_metrics['ROC-AUC']):
                total_acc += task_metrics['Accuracy']
                total_auc += task_metrics['ROC-AUC']
                num_tasks += 1
        
        if num_tasks > 0:
            avg_acc = total_acc / num_tasks
            avg_auc = total_auc / num_tasks
            print(f"\n  Averages:")
            print(f"    Accuracy: {avg_acc:.4f}")
            print(f"    ROC-AUC: {avg_auc:.4f}")
            print(f"    (Averaged across {num_tasks} tasks)")
        else:
            print("\n  Error: No tasks detected for averaging")
    
    def save_metrics_to_csv(dataset_name, task_type, metrics, model_name, feature_type, y_true=None, total_num_graphs=0):
        """
        Save metrics to CSV with feature type included.
        
        Args:
            dataset_name (str): Name of the dataset
            task_type (str): 'classification' or 'regression'
            metrics (dict): Dictionary of evaluation metrics
            model_name (str): Name of the model used
            feature_type (str): Type of features used (e.g., 'graph', 'text', 'graph+text')
            y_true (array-like): True labels (required for classification tasks)
        """
        # Create results directory if it doesn't exist
        os.makedirs('./results', exist_ok=True)
        os.makedirs(f'./{dataset_name}', exist_ok=True)
        filename = f'./results/{dataset_name}_metrics.csv'
        
        # Determine fieldnames based on task type
        if task_type == 'classification':
            fieldnames = [
                'dataset', 'model', 'feature_type', 'task_type', 'metric_type',
                'ROC_AUC', 'Accuracy', 'Num_Positive', 'Num_Negative', 'Total Graphs in Dataset'
            ]
        else:
            fieldnames = [
                'dataset', 'model', 'feature_type', 'task_type', 'metric_type',
                'RMSE', 'MAE', 'R2', 'Total Graphs in Dataset'
            ]
        
        # Prepare rows of data to write
        rows = []
        
        if task_type == 'classification':
            if metrics.get('Per_Task'):
                # Macro averages across tasks
                rows.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'feature_type': feature_type,
                    'task_type': task_type,
                    'metric_type': 'macro_avg',
                    'ROC_AUC': metrics.get('ROC_AUC', np.nan),
                    'Accuracy': np.nanmean([t.get('Accuracy', np.nan) for t in metrics['Per_Task']]),
                    'Num_Positive': 'does not apply',
                    'Num_Negative': 'does not apply',
                    'Total Graphs in Dataset': total_num_graphs
                })
                
                # Per-task metrics
                for i, task_metrics in enumerate(metrics['Per_Task']):
                    rows.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'feature_type': feature_type,
                        'task_type': task_type,
                        'metric_type': f'task_{i}',
                        'ROC_AUC': task_metrics.get('ROC_AUC', np.nan),
                        'Accuracy': task_metrics.get('Accuracy', np.nan),
                        'Num_Positive': task_metrics.get('Num_Positive', 0),
                        'Num_Negative': task_metrics.get('Num_Negative', 0),
                        'Total Graphs in Dataset': total_num_graphs
                    })
            else:
                # Single task classification
                rows.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'feature_type': feature_type,
                    'task_type': task_type,
                    'metric_type': 'overall',
                    'ROC_AUC': metrics.get('ROC_AUC', np.nan),
                    'Accuracy': metrics.get('Accuracy', np.nan),
                    'Total Graphs in Dataset': total_num_graphs
                })
        else:
            # Regression task
            rows.append({
                'dataset': dataset_name,
                'model': model_name,
                'feature_type': feature_type,
                'task_type': task_type,
                'metric_type': 'overall',
                'RMSE': metrics.get('RMSE', np.nan),
                'MAE': metrics.get('MAE', np.nan),
                'R2': metrics.get('R2', np.nan),
                'Total Graphs in Dataset': total_num_graphs
            })
        
        # Write to CSV
        file_exists = os.path.exists(filename)
        with open(filename, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(rows)
    
    # 4. Hypergraph Training-----------------------------------------------------------------------------------------------------------------------------------
    print("Step 4: Hypergraph training and evaluation")
    
    num_repeats = 1
    for i in range(int(num_repeats)):
        TASK_TYPE = 'regression' if dataset_name in ['ESOL', 'FreeSolv', 'Lipophilicity', 'QM7'] else 'classification'
        NUM_EPOCHS = 200 if dataset_name != 'QM7' else 300
        PATIENCE = 10  if dataset_name != 'QM7' else 150
        patience = 0
        
        # initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = hypergraph_NN(molec_X.shape[1], 64, 128, 128, num_class, 0.3).to(device)
        
        if TASK_TYPE == 'classification':
            loss_fn = nn.BCEWithLogitsLoss()
            best_val_metric = 0 
            print("Using BCEWithLogitsLoss for classification task")
        else:
            loss_fn = nn.MSELoss() 
            best_val_metric = float('inf')
            print("Using MSELoss for regression task")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        start_time = time.time()
        
        # training loop
        for epoch in tqdm(range(NUM_EPOCHS), desc=f'Training on {dataset_name}'):
            model.train()
            pred = model(hyG, v_feat, e_feat, True, True)
            
            # process predictions and labels
            f_pred = []
            f_labels = []
            for idx, (pred_item, label_row) in enumerate(zip(pred, label_columns.values)):
                if str(idx) in DICT:
                    f_pred.append(pred_item)
                    f_labels.append(label_row)
            
            f_labels = torch.tensor(np.array(f_labels), dtype=torch.float32)
            f_pred = torch.stack(f_pred)
            
            # 80-10-10 train-val-test split
            train_label, test_label = train_test_split(f_labels, test_size=0.1, random_state=42)
            train_pred, test_pred = train_test_split(f_pred, test_size=0.1, random_state=42)
            
            val_size = int(0.2 * len(train_label))
            val_label = train_label[:val_size]
            train_label = train_label[val_size:]
            val_pred = train_pred[:val_size]
            train_pred = train_pred[val_size:]
            
            
            train_label = train_label.to(device)
            val_label = val_label.to(device)
            train_pred = train_pred.to(device)
            val_pred = val_pred.to(device)
            
            # training step
            loss = loss_fn(train_pred, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # val metrics
            with torch.no_grad():
                if TASK_TYPE == 'regression':
                    train_metrics = calculate_regression_metrics(train_pred, train_label)
                    current_train_metric = train_metrics['RMSE']
                    val_metrics = calculate_regression_metrics(val_pred, val_label)
                    current_val_metric = val_metrics['RMSE']
                else:
                    train_metrics = calculate_classification_metrics(train_pred, train_label)
                    current_train_metric = train_metrics['ROC-AUC']
                    val_metrics = calculate_classification_metrics(val_pred, val_label)
                    current_val_metric = val_metrics['ROC-AUC']
    
            if not Path(f'best_{dataset_name}_model.pth').exists():
                torch.save(model.state_dict(), f'best_{dataset_name}_model.pth')
                
            # save best model
            if ((TASK_TYPE == 'regression' and current_val_metric < best_val_metric) or
                (TASK_TYPE != 'regression' and current_val_metric > best_val_metric)):
                best_val_metric = current_val_metric
                torch.save(model.state_dict(), f'best_{dataset_name}_model.pth')
                patience = 0
                print('best_val updated')
            else:
                patience += 1
            
            # early stopping
            if patience >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
            # print progress every 10 epochs
            if epoch % 10 == 0:
                print(f'\nEpoch {epoch}: Loss = {loss.item():.4f}')
                if TASK_TYPE == 'regression':
                    print(f"  Train RMSE: {train_metrics['RMSE']:.4f}, MAE: {train_metrics['MAE']:.4f}, R2: {train_metrics['R2']:.4f}")
                    print(f"  Val RMSE: {val_metrics['RMSE']:.4f}, MAE: {val_metrics['MAE']:.4f}, R2: {val_metrics['R2']:.4f}")
                else:
                    print_per_task_metrics(train_metrics, "Train")
                    print_per_task_metrics(val_metrics, "Validation")
        
        # final evaluation using test set
        model.load_state_dict(torch.load(f'best_{dataset_name}_model.pth'))
        with torch.no_grad():
            model.eval()
            test_pred = model(hyG, v_feat, e_feat, True, True)
            test_pred = torch.stack([p for i, p in enumerate(test_pred) if str(i) in DICT])
            test_label = torch.tensor([label_columns.values[i] for i in range(len(label_columns)) if str(i) in DICT], 
                                    dtype=torch.float32)
            
            if TASK_TYPE == 'regression':
                test_metrics = calculate_regression_metrics(test_pred, test_label)
                print("\nFinal Regression Metrics:")
                print(f"  RMSE: {test_metrics['RMSE']:.4f}")
                print(f"  MAE: {test_metrics['MAE']:.4f}")
                print(f"  R2: {test_metrics['R2']:.4f}")
            else:
                test_metrics = calculate_classification_metrics(test_pred, test_label)
                print("\nFinal Classification Metrics:")
                print_per_task_metrics(test_metrics, "Test")
            
            save_metrics_to_csv(dataset_name, TASK_TYPE, test_metrics, model_name="hypergraph_NN", feature_type='fp default', total_num_graphs=LEN)
    
    
    # get the hypergraph embeddings and save it
    node_embeddings, hyperedge_embeddings = model(hyG, v_feat, e_feat, True, last_layer=False)
    
    embeddings = {
        'node_embeddings': node_embeddings,
        'hyperedge_embeddings': hyperedge_embeddings
    }
    torch.save(embeddings, f'./{dataset_name}/{dataset_name}_hyG_embeddings.pt')
    print(f'combined embeddings saved to {dataset_name}/{dataset_name}_hyG_embeddings.pt')
    
    # 5. Create and load embeddings
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    import pickle
    
    print('Step 5: Load embeddings')
    
    # load the previously saved hypergraph embeddings
    def load_hypergraph_embeddings(dataset_name):
        """Load hypergraph embeddings from .pt file"""
        file_path = f'./{dataset_name}/{dataset_name}_hyG_embeddings.pt'
        try:
            embeddings = torch.load(file_path)
            return embeddings
        except Exception as e:
            print(f"Error loading hypergraph embeddings: {str(e)}")
            return None
    
    # Use a GIN model to create graph embeddings from molecule graphs
    class GINEncoder(nn.Module):
        def __init__(self, node_dim=8, fingerprint_dim=256, hidden_dim=128):
            super().__init__()
            combined_dim = node_dim + fingerprint_dim
            
            self.conv1 = GINConv(
                nn.Sequential(
                    nn.Linear(combined_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
            self.conv2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        
        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return global_add_pool(x, batch)
    
    # load the pretrained chemical LLM model to create text embeddings from SMILES strings
    class ChemBERTaWrapper:
        def __init__(self):
            self.tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-5M-MLM")
            self.model = AutoModel.from_pretrained("DeepChem/ChemBERTa-5M-MLM").to(device)
            self.model.eval()
        
        def embed(self, smiles_list, batch_size=32):
            embeddings = []
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i+batch_size]
                try:
                    inputs = self.tokenizer(
                        batch, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True, 
                        max_length=128
                    ).to(device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu())
                except Exception as e:
                    print(f"Error processing batch {i}: {str(e)}")
                    continue
                    
            return torch.cat(embeddings, dim=0) if embeddings else None
    
    class MolecularEmbedder:
        def __init__(self, dataset_name=None):
            self.gin = GINEncoder().to(device)
            self.chemberta = ChemBERTaWrapper()
            self.dataset_name = dataset_name
            self.hypergraph_embs = None
            if dataset_name:
                self.hypergraph_embs = load_hypergraph_embeddings(dataset_name)
        
        def _mol_to_graph_data(self, mol):
            mol_fp = get_morgan_fingerprint(mol)
            
            atom_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    int(atom.GetHybridization()),
                    int(atom.GetIsAromatic()),
                    atom.GetTotalNumHs(),
                    int(atom.IsInRing()),
                    atom.GetNumRadicalElectrons()
                ]
                
                atom_fp = get_morgan_fingerprint(mol, atom_index=atom.GetIdx())
                combined_features = np.concatenate([features, atom_fp])
                atom_features.append(combined_features)
            
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.extend([[i, j], [j, i]])
            
            return Data(
                x=torch.tensor(atom_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)
            )
        
        def embed_molecules(self, smiles_list, feature_type="graph+text+hypergraph"):
            graph_embs = []
            text_embs = []
            fingerprints = []
            hypergraph_embs = []
            valid_smiles = []
            
            for smiles in tqdm(smiles_list, desc="Embedding molecules"):
                try:
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if mol is None:
                        continue
                    
                    graph_emb = None
                    hg_emb = None
                    fp = None
                    
                    if 'graph' in feature_type:
                        graph_data = self._mol_to_graph_data(mol)
                        graph_emb = self.gin(
                            graph_data.x.to(device),
                            graph_data.edge_index.to(device),
                            torch.zeros(graph_data.num_nodes, dtype=torch.long).to(device)
                        ).cpu().detach().numpy()
                    
                    if 'hypergraph' in feature_type and self.hypergraph_embs:
                        if smiles in self.hypergraph_embs:
                            hg_emb = self.hypergraph_embs[smiles].cpu().numpy().flatten()
                        else:
                            hg_emb = np.zeros(256)
                    
                    if 'fingerprint' in feature_type:
                        fp = get_morgan_fingerprint(mol)
                    
                    valid_smiles.append(smiles)
                    graph_embs.append(graph_emb)
                    hypergraph_embs.append(hg_emb)
                    fingerprints.append(fp)
                    
                except Exception as e:
                    print(f"Error processing {smiles}: {str(e)}")
                    continue
            
            if 'text' in feature_type:
                text_embs = self.chemberta.embed(valid_smiles)
                if text_embs is None:
                    raise ValueError("Failed to generate text embeddings")
            else:
                text_embs = [None] * len(valid_smiles)
            
            combined_features = []
            for i in range(len(valid_smiles)):
                features_to_combine = []
                
                if 'graph' in feature_type and graph_embs[i] is not None:
                    features_to_combine.append(graph_embs[i].flatten())
                
                if 'text' in feature_type and text_embs[i] is not None:
                    features_to_combine.append(text_embs[i].numpy().flatten())
                
                if 'fingerprint' in feature_type and fingerprints[i] is not None:
                    features_to_combine.append(fingerprints[i])
                
                if 'hypergraph' in feature_type and hypergraph_embs[i] is not None:
                    features_to_combine.append(hypergraph_embs[i])
                
                if not features_to_combine:
                    raise ValueError("No valid features were generated")
                    
                combined = np.concatenate(features_to_combine)
                combined_features.append(combined)
            
            return np.array(combined_features), valid_smiles
        
    def get_morgan_fingerprint(mol, radius=2, n_bits=256, atom_index=None):
        try:
            if atom_index is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, fromAtoms=[atom_index])
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        except:
            return np.zeros(n_bits)
    
    def scaffold_split(df, smiles_col='SMILES', random_state=42):
        valid_mask = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
        df = df[valid_mask].copy()
        df['scaffold'] = df[smiles_col].apply(lambda x: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(str(x))))
        scaffolds = df['scaffold'].unique()
        train_scaff, temp_scaff = train_test_split(scaffolds, test_size=0.2, random_state=random_state)
        val_scaff, test_scaff = train_test_split(temp_scaff, test_size=0.5, random_state=random_state)
        return (
            df[df['scaffold'].isin(train_scaff)].drop('scaffold', axis=1).reset_index(drop=True),
            df[df['scaffold'].isin(val_scaff)].drop('scaffold', axis=1).reset_index(drop=True),
            df[df['scaffold'].isin(test_scaff)].drop('scaffold', axis=1).reset_index(drop=True)
        )
    def morgan_fingerprints(smiles_list, n_bits=1024, radius=2):
        fps = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(fp))
            else:
                fps.append(np.zeros(n_bits))
        return np.array(fps)
        
    def load_process_dataset(dataset_name, use_combined_features=True, feature_type="graph+text+hypergraph"):
        dataset = load_dataset(f"scikit-fingerprints/MoleculeNet_{dataset_name}")
        df = pd.DataFrame(dataset['train'])
    
        smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
        if dataset_name in ['ClinTox', 'Tox21', 'SIDER']:
            label_cols = [col for col in df.columns if col != smiles_col]
        else:
            label_cols = ['label'] if 'label' in df.columns else [dataset_name]
    
        print('splitting dataset with scaffold splitting')
        train_df, val_df, test_df = scaffold_split(df, smiles_col=smiles_col)
    
        if use_combined_features:
            print('featurizing molecules with selected embeddings')
            embedder = MolecularEmbedder(dataset_name)
            X_train, train_smiles = embedder.embed_molecules(train_df[smiles_col], feature_type=feature_type)
            X_val, val_smiles = embedder.embed_molecules(val_df[smiles_col], feature_type=feature_type)
            X_test, test_smiles = embedder.embed_molecules(test_df[smiles_col], feature_type=feature_type)
            
            # Align labels with valid smiles
            train_df = train_df[train_df[smiles_col].isin(train_smiles)]
            val_df = val_df[val_df[smiles_col].isin(val_smiles)]
            test_df = test_df[test_df[smiles_col].isin(test_smiles)]
        else:
            print('featurizing molecules with fingerprints by default')
            X_train = morgan_fingerprints(train_df[smiles_col])
            X_val = morgan_fingerprints(val_df[smiles_col])
            X_test = morgan_fingerprints(test_df[smiles_col])
    
        y_train = train_df[label_cols].fillna(0).values.astype(float)
        y_val = val_df[label_cols].fillna(0).values.astype(float)
        y_test = test_df[label_cols].fillna(0).values.astype(float)
    
        if dataset_name in ['ESOL', 'FreeSolv', 'Lipophilicity']:
            task_type = 'regression'
            multilabel = False
        else:
            task_type = 'classification'
            multilabel = len(label_cols) > 1
    
        return X_train, y_train, X_val, y_val, X_test, y_test, task_type, multilabel, label_cols
    
    def calculate_classification_metrics(y_true, y_pred_probs, y_pred=None):
        if y_pred is None:
            y_pred = (y_pred_probs > 0.5).astype(int)
    
        metrics = {
            'Per_Task': [],
            'Macro_ROC_AUC': 0.0,
            'Valid_Tasks': 0,
            'Accuracy': accuracy_score(y_true, y_pred),
            'ROC_AUC': roc_auc_score(y_true, y_pred_probs) if y_pred_probs.ndim == 1 else float('nan')
        }
    
        if y_pred_probs.ndim == 1:
            return metrics
    
        for i in range(y_true.shape[1]):
            try:
                task_auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i]) if len(np.unique(y_true[:, i])) > 1 else float('nan')
                task_acc = accuracy_score(y_true[:, i], y_pred[:, i])
                metrics['Per_Task'].append({'ROC_AUC': task_auc, 'Accuracy': task_acc, 'Num_Positive': np.sum(y_true[:, i]), 'Num_Negative': np.sum(y_true[:, i]== 0)})
                if not np.isnan(task_auc):
                    metrics['Macro_ROC_AUC'] += task_auc
                    metrics['Valid_Tasks'] += 1
            except ValueError:
                metrics['Per_Task'].append({'ROC_AUC': float('nan'), 'Accuracy': float('nan')})
    
        if metrics['Valid_Tasks'] > 0:
            metrics['Macro_ROC_AUC'] /= metrics['Valid_Tasks']
    
        return metrics
    
    def calculate_regression_metrics(y_true, y_pred):
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def print_metrics(metrics, dataset_name, task_type):
        print(f"\n{dataset_name} Metrics:")
        if task_type == 'classification':
            if metrics['Per_Task']:
                print("Per-Task Metrics:")
                for i, task in enumerate(metrics['Per_Task']):
                    print(f"  Task {i}: ROC-AUC: {task['ROC_AUC']:.4f}, Accuracy: {task['Accuracy']:.4f}, Num_Positive: {task['Num_Positive']}, Num_Negative: {task['Num_Negative']}")
                print(f"Macro Average ROC-AUC: {metrics['Macro_ROC_AUC']:.4f} (across {metrics['Valid_Tasks']} tasks)")
            else:
                print(f"ROC-AUC: {metrics['ROC_AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
        else:
            print(f"RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")
    
    def train_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, task_type, multilabel, label_cols, dataset_name, feature_type):
        pipeline = Pipeline([('scaler', StandardScaler())])
        
        if task_type == 'classification':
            models = {
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
                'MLP': MLPClassifier(hidden_layer_sizes=(64,), random_state=42, max_iter=500),
                'Ridge': RidgeClassifier(random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'SGD': SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
            }
        else:
            models = {
                'SVM': SVR(),
                'Ridge': Ridge(random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'MLP': MLPRegressor(hidden_layer_sizes=(64,), random_state=42, max_iter=500),
                'SGD': SGDRegressor(loss='squared_error', max_iter=1000, random_state=42)
            }
    
        if multilabel and task_type == 'classification':
            models = {name: MultiOutputClassifier(model) for name, model in models.items()}
    
        X_train = pipeline.fit_transform(X_train)
        X_val = pipeline.transform(X_val)
        X_test = pipeline.transform(X_test)
    
        X_full = np.vstack([X_train, X_val])
        y_full = np.vstack([y_train, y_val]) if multilabel else np.concatenate([y_train, y_val])
    
        results = {}
        for name, model in models.items():
            print(f"\nTraining {name} on {dataset_name}...")
            model.fit(X_full, y_full)
    
            if task_type == 'classification':
                if hasattr(model, 'predict_proba'):
                    y_pred_probs = model.predict_proba(X_test)
                    if multilabel:
                        y_pred_probs = np.array([p[:, 1] for p in y_pred_probs]).T
                    else:
                        y_pred_probs = y_pred_probs[:, 1]
                    y_pred = (y_pred_probs > 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_probs = y_pred
    
                metrics = calculate_classification_metrics(y_test, y_pred_probs, y_pred)
                results[name] = metrics['ROC_AUC'] if not multilabel else metrics['Macro_ROC_AUC']
            else:
                y_pred = model.predict(X_test)
                metrics = calculate_regression_metrics(y_test, y_pred)
                results[name] = metrics['RMSE']
    
            print_metrics(metrics, dataset_name, task_type)
            save_metrics_to_csv(dataset_name, task_type, metrics, feature_type, name, y_test, total_num_graphs=len(y_full))
        return results

    # Process all feature combinations for this dataset
    for feature_num, feature_type in feature_options.items():
        print(f"\nProcessing {dataset_name} with feature combination: {feature_type}")
        
        try:
            # Try with combined features first
            try:
                X_train, y_train, X_val, y_val, X_test, y_test, task_type, multilabel, label_cols = load_process_dataset(
                    dataset_name, 
                    use_combined_features=True,
                    feature_type=feature_type
                )
            except Exception as e:
                print(f"Failed to use combined features, falling back to fingerprints: {str(e)}")
                X_train, y_train, X_val, y_val, X_test, y_test, task_type, multilabel, label_cols = load_process_dataset(dataset_name, use_combined_features=False)
                feature_type = "fingerprint"
            
            print(f"Using {feature_type} features for {dataset_name}")
            results = train_evaluate(X_train, y_train, X_val, y_val, X_test, y_test,
                                    task_type, multilabel, label_cols, dataset_name, feature_type=feature_type)
            
            print(f"\nResults for {dataset_name} with {feature_type} features:")
            for model, score in results.items():
                metric = "ROC-AUC" if task_type == 'classification' else "RMSE"
                print(f"{model}: {metric} = {score:.4f}")
        except Exception as e:
            print(f"Failed to process {dataset_name} with {feature_type} features: {str(e)}")
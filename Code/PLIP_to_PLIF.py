# NOTE: line tags begining with "## CHANGE" should be updated to contain user-specific information 

import xml.etree.ElementTree as ET 
import pandas as pd
from collections import ChainMap
import csv 
import os


def parse_xml(xmlfile):
    '''
    Parse the XML file and extract tree root.
    
        Args:
            xmlfile (str): Path to the XML file.
        
        Returns:
            tree_root (Element): The root element of the parsed XML tree.
            ligand_id (str): The ligand ID extracted from the XML file.
            atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
            to ligand atom id numbering (values).
    '''
    
    tree_root = ET.parse(xmlfile).getroot()
    ligand_id = tree_root.find('.//longname').text ## CHANGE depending on user naming system

    atom_idx_list = tree_root.find('bindingsite').find('mappings').find('smiles_to_pdb').text.split(',')
    
    atom_idx_map = {}
    for pair in atom_idx_list:
        try:
            k, v = pair.split(':')
            atom_idx_map[v] = k
        except ValueError:
            pass
        
    return tree_root, ligand_id, atom_idx_map


def get_hydrophob_ints(tree_root, atom_idx_map):
    '''
    Extract hydrophobic interaction data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        hydrophob_ints_map (dict): Dictionary mapping hydrophobic interactions
        to specified features.
    '''
    
    hydrophob_ints_map = {}
    hydrophob_count = 0
    raw_hydrophob_tags = ['resnr','restype','dist','ligcarbonidx','protcarbonidx']    
    
    for hydrophob_int in tree_root.findall('./bindingsite/interactions/hydrophobic_interactions/hydrophobic_interaction'):
        hydrophob_ints_map['hydrophob_int'+hydrophob_int.attrib['id']] ={}
        hydrophob_count += 1
        
        for features in hydrophob_int:
            if features.tag in raw_hydrophob_tags:
                if 'idx' in features.tag and features.text in atom_idx_map.keys():
                    atom_idx = atom_idx_map[features.text]
                else:
                    atom_idx = features.text
                hydrophob_ints_map['hydrophob_int'+hydrophob_int.attrib['id']][features.tag] = atom_idx

    hydrophob_ints_map['hydrophob_int_count'] = hydrophob_count
    
    return hydrophob_ints_map


def get_hbonds(tree_root, atom_idx_map):
    '''
    Extract hydorgen bond data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        hbond_map (dict): Dictionary mapping hydrogen bonds
        to specified features.
    '''
    
    hbond_map = {}
    hbond_count = 0
    raw_hbond_tags = ['resnr','restype','sidechain','dist_h-a','dist_d-a','don_angle','protisdon','donoridx','donortype','acceptoridx','acceptortype']    
    
    for hbond in tree_root.findall('./bindingsite/interactions/hydrogen_bonds/hydrogen_bond'):
        hbond_map['hbond'+hbond.attrib['id']] ={}
        hbond_count += 1
        
        for features in hbond:
            if features.tag in raw_hbond_tags:
                if 'idx' in features.tag and features.text in atom_idx_map.keys():
                    atom_idx = atom_idx_map[features.text]
                else:
                    atom_idx = features.text
                hbond_map['hbond'+hbond.attrib['id']][features.tag] = atom_idx
    
    hbond_map['hbond_count'] = hbond_count
    
    return hbond_map


def get_waterbridges(tree_root, atom_idx_map):
    '''
    Extract water bridge data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        waterbridge_map (dict): Dictionary mapping water bridges
        to specified features.
    '''
    
    waterbridge_map = {}
    waterbridge_count = 0
    raw_waterbridge_tags = ['resnr','restype','sidechain','dist_a-w','dist_d-w','don_angle','protisdon','donoridx','donortype','water_idx','water_angle'] 
    
    for waterbridge in tree_root.findall('./bindingsite/interactions/water_bridges/water_bridge'):
        waterbridge_map['waterbridge'+waterbridge.attrib['id']] ={}
        waterbridge_count += 1
        
        for features in waterbridge:
            
            if features.tag in raw_waterbridge_tags:
                if 'idx' in features.tag and features.text in atom_idx_map.keys():
                    atom_idx = atom_idx_map[features.text]
                else:
                    atom_idx = features.text
                waterbridge_map['waterbridge'+waterbridge.attrib['id']][features.tag] = atom_idx                
    
    waterbridge_map['waterbridge_count'] = waterbridge_count
    
    return waterbridge_map


def get_saltbridges(tree_root, atom_idx_map):
    '''
    Extract salt bridge data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        saltbridge_map (dict): Dictionary mapping salt bridges
        to specified features.
    '''
    
    saltbridge_map = {}
    saltbridge_count = 0
    raw_saltbridge_tags = ['resnr','restype','prot_idx_list','dist','protispos','lig_group','lig_idx_list']
    
    for saltbridge in tree_root.findall('./bindingsite/interactions/salt_bridges/salt_bridge'):
        saltbridge_map['saltbridge'+saltbridge.attrib['id']] ={}
        saltbridge_count += 1
        
        for features in saltbridge:

            if features.tag in raw_saltbridge_tags:
                
                if features.tag == 'prot_idx_list' or features.tag == 'lig_idx_list':
                    saltbridge_map['saltbridge'+saltbridge.attrib['id']][features.tag] = {}
                    
                    for idx in features:
                        if idx.text in atom_idx_map.keys():
                            atom_idx = atom_idx_map[idx.text]
                        else:
                            atom_idx = idx.text
                        saltbridge_map['saltbridge'+saltbridge.attrib['id']][features.tag][idx.tag+idx.attrib['id']] = atom_idx    
                else:
                    saltbridge_map['saltbridge'+saltbridge.attrib['id']][features.tag] = features.text
                                
    
    saltbridge_map['saltbridge_count'] = saltbridge_count
    
    return saltbridge_map


def get_pistacks(tree_root, atom_idx_map):
    '''
    Extract pi stacking interaction data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        pistack_map (dict): Dictionary mapping pi stacking interactions
        to specified features.
    '''
    
    pistack_map = {}
    pistack_count = 0
    raw_pistack_tags = ['resnr','restype','prot_idx_list','centdist','angle','offset','type','lig_idx_list'] 
        
    for pistack in tree_root.findall('./bindingsite/interactions/pi_stacks/pi_stack'):
        pistack_map['pistack'+pistack.attrib['id']] ={}
        pistack_count += 1
        
        for features in pistack:

            if features.tag in raw_pistack_tags:
                
                if features.tag == 'prot_idx_list' or features.tag == 'lig_idx_list':
                    pistack_map['pistack'+pistack.attrib['id']][features.tag] = {}
                    
                    for idx in features:
                        if idx.text in atom_idx_map.keys():
                            atom_idx = atom_idx_map[idx.text]
                        else:
                            atom_idx = idx.text
                        pistack_map['pistack'+pistack.attrib['id']][features.tag][idx.tag+idx.attrib['id']] = atom_idx             
                else:
                    pistack_map['pistack'+pistack.attrib['id']][features.tag] = features.text
                                
    
    pistack_map['pistack_count'] = pistack_count
    
    return pistack_map


def get_picat_ints(tree_root, atom_idx_map):
    '''
    Extract pi-cation interaction data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        picat_ints_map (dict): Dictionary mapping pi-cation interactions
        to specified features.
    '''
    
    picat_ints_map = {}
    picat_ints_count = 0
    raw_picat_ints_tags = ['resnr','restype','prot_idx_list','dist','offset','protcharged','lig_group','lig_idx_list'] 
        
    
    for picat_int in tree_root.findall('./bindingsite/interactions/pi_cation_interactions/pi_cation_interaction'):
        picat_ints_map['picat_int'+picat_int.attrib['id']] ={}
        picat_ints_count += 1
        
        for features in picat_int:

            if features.tag in raw_picat_ints_tags:
                
                if features.tag == 'prot_idx_list' or features.tag == 'lig_idx_list':
                    picat_ints_map['picat_int'+picat_int.attrib['id']][features.tag] = {}
                    
                    for idx in features:
                        if idx.text in atom_idx_map.keys():
                            atom_idx = atom_idx_map[idx.text]
                        else:
                            atom_idx = idx.text
                        picat_ints_map['picat_int'+picat_int.attrib['id']][features.tag][idx.tag+idx.attrib['id']] = atom_idx      
                else:
                    picat_ints_map['picat_int'+picat_int.attrib['id']][features.tag] = features.text
                                
    picat_ints_map['picat_ints_count'] = picat_ints_count
    
    return picat_ints_map


def get_halogenbonds(tree_root, atom_idx_map):
    '''
    Extract halogen bond data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        halogenbond_map (dict): Dictionary mapping halogen bonds
        to specified features.
    '''
    
    halogenbond_map = {}
    halogenbond_count = 0
    raw_halogenbond_tags = ['resnr','restype','sidechain','dist','don_angle','acc_angle','don_idx','donortype','acc_idx','acceptortype'] 
        
    
    for halogenbond in tree_root.findall('./bindingsite/interactions/halogen_bonds/halogen_bond'):
        halogenbond_map['halogenbond'+halogenbond.attrib['id']] ={}
        halogenbond_count += 1
        
        for features in halogenbond:

            if features.tag in raw_halogenbond_tags:
                if 'idx' in features.tag and features.text in atom_idx_map.keys():
                    atom_idx = atom_idx_map[features.text]
                else:
                    atom_idx = features.text
                halogenbond_map['halogenbond'+halogenbond.attrib['id']][features.tag] = atom_idx
                                
    halogenbond_map['halogenbond_count'] = halogenbond_count
    
    return halogenbond_map


def get_interactions(tree_root, ligandID, atom_idx_map):
    ''' 
    Extract specific interaction type data from the XML tree.
    
    Args:
        tree_root (Element): Root element of the XML tree.
        ligandID (str): Ligand identifier used to index the final dataframe.
        atom_idx_map (dict): Dictionary mapping PDB atom id numbering (keys) 
        to ligand atom id numbering (values).
    
    Returns:
        df (pd.DataFrame): DataFrame containing interaction data for the ligand, 
        indexed by ligandID, with columns representing interaction features.
    '''
        
    hydrophob_ints_map = get_hydrophob_ints(tree_root, atom_idx_map)
    hbond_map = get_hbonds(tree_root, atom_idx_map)
    waterbridge_map = get_waterbridges(tree_root, atom_idx_map)
    saltbridge_map = get_saltbridges(tree_root, atom_idx_map)
    pistack_map = get_pistacks(tree_root, atom_idx_map)
    picat_ints_map = get_picat_ints(tree_root, atom_idx_map)
    halogenbond_map = get_halogenbonds(tree_root, atom_idx_map)
    
    interaction_map_list = [hydrophob_ints_map, hbond_map, waterbridge_map, saltbridge_map, pistack_map, picat_ints_map, halogenbond_map]
    
    df = pd.json_normalize(dict(ChainMap(*interaction_map_list)))
    df.index = [ligandID]
    
    return df


def main():
    basedir = 'Example_XMLs' ## CHANGE IF NEEDED
    interactions_map = pd.DataFrame()
    xmlfile_suffix = '1B1in-raw.xml' ## CHANGE IF NEEDED
    
    labeling_file = 'label_ligandID_key.csv' ## CHANGE IF NEEDED
    labeling_df = pd.read_csv(labeling_file)

    for root, dirs, files in os.walk(basedir):
        for filename in files:
            if filename.endswith(xmlfile_suffix):
                
                xmlpath = os.path.join(root, filename)
                tree_root, ligandID, atom_idx_map = parse_xml(xmlpath)
                
                temp_interactions_df = get_interactions(tree_root, ligandID, atom_idx_map)

                temp_interactions_df['Class'] = labeling_df[labeling_df['ligandID'] == ligandID]['Class'].values[0]
                interactions_map = pd.concat([interactions_map, temp_interactions_df])
                
    interactions_map.to_csv('1B1in-PLIF.csv',index=True,index_label='ligandID') ## CHANGE IF NEEDED
    
    
if __name__=='__main__':
    main()

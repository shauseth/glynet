# describe.py

import pandas as pd
from glypy.io import iupac
from glypy.algorithms import subtree_search


# Lists

monomers = ['Glc', 'Gal', 'Man', 'Fuc', 'Kdn', 'GlcNAc', 'GalNAc', 'GlcA', 'Neu5Ac', 'Neu5Gc']
terminal = ['[' + monomer for monomer in monomers]
modifications = ['3S', '4S', '6S', '6P']
modified = ['(6P)Man', '(6S)Glc', '(6P)Glc', '(3S)GlcA',
            '(3S)Gal', '(4S)Gal', '4S(3S)Gal', '6S(3S)Gal', '(6S)(4S)Gal', '(6P)Gal',
            '(3S)GalNAc', '(4S)GalNAc', '(6S)GalNAc', '(6S)(4S)GalNAc',
            '(3S)GlcNAc', '(6S)GlcNAc', '(6P)GlcNAc']


# Helper Functions

def pre_process(glycan):
    glycan = glycan.replace('KDN', 'Kdn')
    return glycan

def post_process(glycan, subtree):
    subtree = subtree.replace('2NAc', 'NAc')
    subtree = subtree.replace('Neu5N', 'Neu5')
    subtree = subtree.replace('a-Glc', 'GlcA')
    if '5Ac(a' in glycan or '5Gc(a' in glycan:
        subtree = subtree.replace('?', 'a')
    elif '5Ac(b' in glycan or '5Gc(b' in glycan:
        subtree = subtree.replace('?', 'b')
    return subtree

def is_modified(glycan):
    n = 0
    for x in modifications:
        n += glycan.count(x)
    return True if n > 0 else False

def remove_modification(glycan):
    for item in modifications:
        glycan = glycan.replace('(' + item + ')', '')
        glycan = glycan.replace(item, '')
    return glycan

def get_subtrees(glycan, k=2):
    structure = iupac.loads(glycan, dialect='simple')
    subtrees = []
    for treelet in subtree_search.treelets(structure, k, distinct=False):
        subtree = iupac.dumps(treelet, dialect='simple')
        subtrees.append(post_process(glycan, subtree))
    return subtrees

def count_items(string, items):
    count_dict = {}
    for item in reversed(items):
        count = string.count(item)
        if count > 0:
            string = string.replace(item, '')
            count_dict[item] = count
    return count_dict

def count_subtrees(subtrees):
    count_dict = {}
    for subtree in set(subtrees):
        count = subtrees.count(subtree)
        if count > 0:
            count_dict[subtree] = count
    return count_dict


# Main Functions

def remove_linker(iupac):
    """Removes linker from IUPAC."""
    return '('.join(iupac.split('(')[:-1])

def get_descriptors(glycan, depth=3, use_terminal=True):
    """Get descriptors for CFG glycan with counts.
    depth: Number of monomers in the largest subtree searched. Default: 3
    use_terminal: Use terminal monosaccharide as descriptors. Default: True"""
    glycan = pre_process(glycan)
    descriptors = {}
    descriptors.update(count_items(glycan, monomers + modifications))
    if depth > 1:
        if is_modified(glycan):
            descriptors.update(count_items(glycan, modified))
            glycan = remove_modification(glycan)
        for k in range(2, depth + 1):
            subtrees = get_subtrees(glycan, k)
            descriptors.update(count_subtrees(subtrees))
    if use_terminal:
        glycan = '[' + glycan
        descriptors.update(count_items(glycan, terminal))
    return descriptors

def get_fingerprints(iupacs, depth=3, use_terminal=True):
    """Get fingerprint descriptors for a list of IUPACs.
    depth: Number of monomers in the largest subtree searched. Default: 3
    use_terminal: Use terminal monosaccharide as descriptors. Default: True"""
    all_descs, all_dicts, fingprs = [], [], []
    for iupac in iupacs:
        desc_dict = get_descriptors(iupac, depth, use_terminal)
        all_descs += desc_dict.keys()
        all_dicts.append(desc_dict)
    descs = sorted(set(all_descs))
    for desc_dict in all_dicts:
        fingpr = [desc_dict[desc] if desc in desc_dict else 0 for desc in descs]
        fingprs.append(fingpr)
    fingpr_data = pd.DataFrame(fingprs, columns=descs)
    fingpr_data.insert(0, 'IUPAC', iupacs)
    return fingpr_data

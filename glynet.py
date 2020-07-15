# glynet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import r2_score

import os
import random
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Encoding

monomers = ['Fuc', 'GalNAc', 'Gal', 'GlcNAc', 'GlcA', 'Glc', 'KDN',
            'Man', 'Rha', 'Neu5,9Ac2', 'Neu5Ac', 'Neu5Gc', 'MurNAc']
linkages = ['a', 'b']
occupancies = ['2', '3', '4', '5', '6']
branches = ['[', ']']
sul_phos = ['3S', '4S', '6S', '6P']

def fix(iupac):
    """Fix defective IUPACs."""
    iupac = iupac.replace('GlcNA(cb-S)p(1-4)[Fuc(a1-6)]GlcNAc(b1-Sp19', 'GlcNAc(b1-Sp19')
    iupac = iupac.replace(' ', '')
    iupac = iupac.replace('\xa0', '')
    iupac = iupac.replace('M(an-a)', 'Man')
    iupac = iupac.replace('Ma(n', 'Man(a')
    iupac = iupac.replace('GlcNAc(b1-4)(a1-MDPLys', 'GlcNAc(b1-MDPLys')
    iupac = iupac.replace('GlcNAc(b1-3)Ma(n1-Sp10', 'GlcNAc(b1-3)Man(a1-Sp10')
    iupac = iupac.replace('GlcNA(', 'GlcNAc(')
    iupac = iupac.replace('GlcNac', 'GlcNAc')
    iupac = iupac.replace('GalNac', 'GalNAc')
    iupac = iupac.replace('Ga(l', 'Gal(a')
    return iupac

def encode(iupac):
    """Encode an IUPAC."""
    vector = []
    for unit in iupac.split('-')[:-1]:
        vector += [unit.count(monomer + '(') for monomer in monomers]
        vector += [unit.split('(')[-1].count(linkage) for linkage in linkages]
        vector += [(unit[0] + unit[-1]).count(occupancy) for occupancy in occupancies]
        vector += [unit.count(branch) for branch in branches]
        vector += [unit.count(supo) for supo in sul_phos]
    tensor = torch.tensor(vector).float()
    return tensor.view(-1, 26)

def pad(encoded, size = 40):
    """Zero pad the bottom of an encoded IUPAC."""
    pad = size - encoded.shape[0]
    padded = F.pad(encoded, (0, 0, 0, pad))
    return padded

def plot_encoded(encoded, filename = None):
    """Plot encoded IUPACs."""
    xticks = monomers + linkages + occupancies + branches + sul_phos
    plt.imshow(encoded, cmap = 'Blues')
    plt.xticks(range(len(xticks)), xticks, rotation = 90)
    plt.yticks(range(encoded.shape[0]))
    plt.xlabel('Feature')
    plt.ylabel('Unit')
    plt.savefig(filename + '.pdf', bbox_inches = 'tight') if filename else plt.show()


# Data

def sort_data(data):
    """Sorts the sum of all columns."""
    data['sum'] = data[data.columns[1:]].values.sum(axis = 1)
    data = data.sort_values('sum')
    del data['sum']
    return data

def ten_fold(traindata):
    """Stratififed 10-fold cross validation with random sampling."""
    sections = []
    assert len(traindata) == 600
    for i in range(0, 600, 100):
        section = traindata[i:i + 100]
        sections.append(section)
    for i in range(10):
        held_out = []
        for section in sections:
            random.shuffle(section)
            sample = section[:10]
            del section[:10]
            held_out += sample
        kept_in = [pair for pair in traindata if not pair in held_out]
        yield held_out, kept_in

def get_data(transformation = np.cbrt, cutoff = 1.0, path = 'data/', stdev = False):
    """Get CBP data from CSVs in path."""
    cbps = [cbp for cbp in os.listdir(path) if not cbp.endswith('.csv') and cbp != '.DS_Store']
    data = pd.DataFrame()
    for i, cbp in enumerate(sorted(cbps)):
        cbp_data = pd.read_csv(path + cbp + '_stdev.csv') if stdev else pd.read_csv(path + cbp + '.csv')
        cbp_values = cbp_data[cbp_data.columns[-1]]
        if i == 0:
            data['IUPAC'] = cbp_data['IUPAC'].apply(fix)
            data[cbp] = cbp_values
        else:
            data[cbp] = cbp_values
    index = pd.read_csv('index.csv')
    data['Index'] = index['Index']
    data = data.dropna()
    del data['Index']
    data = transform(data, transformation, cutoff)
    return data

def transform(data, transformation, cutoff):
    """Transform average RFUs with some cutoff."""
    proteins = list(data.columns[1:])
    values = data[proteins].values
    values[values <= cutoff] = cutoff
    values = transformation(values)
    data[proteins] = values
    return data

def detransform(results, transformation = np.cbrt):
    """Remove the transformation from the results."""
    values = results[results.columns[1:]].values
    results[results.columns[1:]] = np.power(values, 3) if transformation == np.cbrt else None
    return results

def prepare_data(data, mode = 'train', batch_size = 16):
    """Prepares inputs and outputs for training or testing."""
    inputs, values = [], []
    for glycan, value in data:
        glycan_tensor = pad(encode(glycan)).view(-1)
        value_tensor = torch.tensor(value).float()
        inputs.append(glycan_tensor)
        values.append(value_tensor)
    if mode == 'train':
        trainset = list(zip(inputs, values))
        return torch.utils.data.DataLoader(trainset, batch_size = batch_size)
    elif mode == 'test':
        inputs_stack = torch.stack(inputs)
        values_stack = torch.stack(values).float()
        return inputs_stack, values_stack
    
def make_results(data, actual, predicted, glycans, experiment, path, proteins, save = True):
    """Makes the results CSV in path with experiment name."""
    data = data.sort_index()
    results = pd.DataFrame()
    results['glycans'] = glycans
    actual_tensor = torch.tensor(actual).float()
    predicted_tensor = torch.tensor(predicted) # .unsqueeze(1) # here
    for i in range(len(proteins)):
        results[proteins[i] + ' actual'] = actual_tensor[:, i]
        results[proteins[i] + ' predicted'] = predicted_tensor[:, i]
    results['index'] = results['glycans'].map(dict(zip(data['IUPAC'], data.index)))
    results = results.sort_values('index')
    del results['index']
    results = detransform(results)
    if save:
        results.to_csv(path + experiment + '.csv', index = False)
    return results


# Visuals

def plot_performance(train_losses, test_losses, r2s):
    """Plots training loss, test loss and the r-squared."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
    ax1.plot(train_losses, label = 'train_loss')
    ax1.plot(test_losses, label = 'test_loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.grid(alpha = 0.2)
    ax2.plot(r2s, label = 'r-squared', color = 'green')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('r-squared')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha = 0.2)
    plt.show()

def plot_all(results, title, proteins, filename = None):
    """Plots the r-squared for all proteins as bar charts."""
    r2_list = []
    fig, ax = plt.subplots(figsize = (12, 4))
    plt.title(title)
    for i, protein in enumerate(proteins):
        r2 = r2_score(results[protein + ' actual'].values, results[protein + ' predicted'].values)
        r2_list.append(r2)
        plt.bar(i, r2, color = color_dict[protein])
    plt.xticks(range(len(proteins)), proteins, rotation = 270)
    plt.xlabel('Proteins')
    plt.ylabel('R-squared')
    plt.ylim(0, 1)
    plt.grid(axis = 'y', alpha = 0.2)
    plt.savefig(filename, bbox_inches = 'tight') if filename else plt.show()

def plot_results(actual, predicted, title = 'title', color = 'cornflowerblue', filename = None):
    """Plots the predictions along with the actuals."""
    plt.figure(figsize = (20, 5))
    plt.title(title + '_r' + str(round(r2_score(actual, predicted), 4)))
    plt.plot(actual, label = 'actual', color = color)
    plt.scatter(range(len(predicted)), predicted, label = 'prediction', color = 'grey')
    plt.xlabel('Glycan')
    plt.ylabel('Normalized Avg. RFU')
    plt.ylim(0)
    plt.legend()
    plt.savefig(filename, bbox_inches = 'tight') if filename else plt.show()


# Other

def norm(values):
    """Normalize between 0 and 1."""
    p = (values - values.min())
    q = values.max() - values.min()
    return p / q


# Color Dictionary
color_dict = {'AAA': 'darkkhaki',
             'AAL': 'darkorange',
             'ABA': 'peru',
             'ACL': 'crimson',
             'AIA': 'olivedrab',
             'AMA': 'limegreen',
             'AOL': 'gold',
             'BPL': 'magenta',
             'CAA': 'yellowgreen',
             'CAL': 'pink',
             'CFL': 'seagreen',
             'CTB': 'saddlebrown',
             'ConA': 'greenyellow',
             'DBA': 'sienna',
             'DSL': 'violet',
             'ECL': 'red',
             'EEL': 'deeppink',
             'G3C': 'peachpuff',
             'GNL': 'forestgreen',
             'GSL': 'darkolivegreen',
             'HAA': 'chocolate',
             'HHA': 'salmon',
             'HPA': 'sandybrown',
             'LCA': 'green',
             'LEL': 'tomato',
             'LFA': 'tan',
             'LTL': 'red',
             'MAL': 'forestgreen',
             'MNA': 'midnightblue',
             'MOA': 'goldenrod',
             'MPL': 'yellowgreen',
             'NPL': 'yellow',
             'PNA': 'gold',
             'PSA': 'yellow',
             'PSL': 'goldenrod',
             'PTL': 'limegreen',
             'RCA': 'seagreen',
             'RPA': 'limegreen',
             'SBA': 'sandybrown',
             'SJA': 'darkseagreen',
             'SNA': 'darkgreen',
             'STA': 'goldenrod',
             'TJA': 'forestgreen',
             'TL': 'lightcoral',
             'UDA': 'limegreen',
             'UEA': 'yellow',
             'VGA': 'orchid',
             'VVL': 'blueviolet',
             'WFL': 'mediumpurple',
             'WGA': 'wheat'}

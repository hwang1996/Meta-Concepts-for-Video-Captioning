# %%
import sys
sys.path.append('../SceneGraphParser')
import sng_parser 

from tqdm import tqdm

# %%
import os
import h5py

label_h5 = os.path.join('../../../msrvtt_captioning/output/metadata', 'msrvtt_train_sequencelabel.h5') 
label_h5 = h5py.File(label_h5, 'r')
vocab = [i for i in label_h5['vocab']]

# %%
import json

train_val_file = json.load(open('../files/train_videodatainfo.json'))
test_file = json.load(open('../files/test_videodatainfo.json'))

# %%
entity_list = []
for i, cap in enumerate(tqdm(train_val_file['sentences'])):
    graph = sng_parser.parse(cap['caption'])
    for entity in graph['entities']:
        if entity['lemma_head'].encode() in vocab:
            entity_list.append(entity['lemma_head'])
        elif entity['head'].encode() in vocab:
            entity_list.append(entity['head'])

# %%
from collections import Counter
from PyDictionary import PyDictionary
dictionary=PyDictionary()

entity_freq = Counter(entity_list)
entity_freq.most_common()

# %%
entity_list_1 = []
for i, entity in enumerate(tqdm(entity_list)):
#     import pdb; pdb.set_trace()
    if len(entity.split(' ')) > 0:
        entity_list_1.append(entity.split(' ')[-1])
    else:
        entity_list_1.append(entity)

# %%
entity_freq = Counter(entity_list_1)
# entity_freq = entity_freq.most_common()
entity_freq_ = entity_freq.most_common(5220)    # freq >= 3
all_ent = [a[0] for a in entity_freq_]

# %%
import spacy
nlp = spacy.load('en_core_web_lg')

# %%
synonym_list = {}
for i, item in enumerate(all_ent):
    cluster = [t.cluster for t in nlp(item)][0]
    if cluster not in synonym_list:
        synonym_list[cluster] = [item]
    else:
        synonym_list[cluster].append(item)

# %%
synonyms = []
synonyms_freq = []
keys = list(synonym_list.keys())
synonyms.append(synonym_list[keys[0]])
synonyms_freq.append([entity_freq_[all_ent.index(ent)][1] for ent in synonym_list[keys[0]]])
synonyms.append(synonym_list[keys[1]])
synonyms_freq.append([entity_freq_[all_ent.index(ent)][1] for ent in synonym_list[keys[1]]])

synonym_all = synonyms[0].copy()
synonym_all.extend([a for a in synonyms[-1]])

# %%
for ent, freq in entity_freq_:
    if len(synonyms) == 60:
        break
    if ent in synonym_all:
        continue
    else:
        try:
            ent_syn = dictionary.synonym(ent) 
            rest_ents = list(set(all_ent_copy)-set(synonym_all))
            intersection = list(set(rest_ents).intersection(set(ent_syn)))
            intersection.append(ent)
            synonyms.append(intersection)
            synonyms_freq.append([entity_freq_[all_ent.index(ent)][1] for ent in intersection])
            synonym_all.extend([a for a in synonyms[-1]])
        except:
            continue            


# %%
synonyms_class = {}
for i, sym_ in enumerate(synonyms):
    for sym in sym_:
        synonyms_class[sym] = i + 1

# %%
synonyms_freq_list = {'synonyms': synonyms, 'synonyms_freq': synonyms_freq, 'synonyms_class': synonyms_class}

import pickle
pickle.dump(synonyms_freq_list, open('../files/synonyms_freq_list_60.pkl', 'wb'))



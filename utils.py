import numpy as np
import keras_bert
from random import randint
from itertools import product
import keras
from logging import info
import collections
from typing import List, OrderedDict, Optional

ALPHABET = 'ACGT'
CLASS_LABELS = OrderedDict([('superkingdom', ['Archaea', 'Bacteria', 'Eukaryota',
    'Viruses', 'unknown']), ('phylum', ['Actinobacteria', 'Apicomplexa', 'Aquificae',
    'Arthropoda', 'Artverviricota', 'Ascomycota', 'Bacillariophyta', 'Bacteroidetes',
    'Basidiomycota', 'Candidatus Thermoplasmatota', 'Chlamydiae', 'Chlorobi',
    'Chloroflexi', 'Chlorophyta', 'Chordata', 'Crenarchaeota', 'Cyanobacteria',
    'Deinococcus-Thermus', 'Euglenozoa', 'Euryarchaeota', 'Evosea', 'Firmicutes',
    'Fusobacteria', 'Gemmatimonadetes', 'Kitrinoviricota', 'Lentisphaerae', 'Mollusca',
    'Negarnaviricota', 'Nematoda', 'Nitrospirae', 'Peploviricota', 'Pisuviricota',
    'Planctomycetes', 'Platyhelminthes', 'Proteobacteria', 'Rhodophyta', 'Spirochaetes',
    'Streptophyta', 'Tenericutes', 'Thaumarchaeota', 'Thermotogae', 'Uroviricota',
    'Verrucomicrobia', 'unknown']), ('genus', ['Acidilobus', 'Acidithiobacillus',
    'Actinomyces', 'Actinopolyspora', 'Acyrthosiphon', 'Aeromonas', 'Akkermansia', 'Anas',
    'Apis', 'Aquila', 'Archaeoglobus', 'Asparagus', 'Aspergillus', 'Astyanax', 'Aythya',
    'Bdellovibrio', 'Beta', 'Betta', 'Bifidobacterium', 'Botrytis', 'Brachyspira',
    'Bradymonas', 'Brassica', 'Caenorhabditis', 'Calypte', 'Candidatus Kuenenia',
    'Candidatus Nitrosocaldus', 'Candidatus Promineofilum', 'Carassius', 'Cercospora',
    'Chanos', 'Chlamydia', 'Chrysemys', 'Ciona', 'Citrus', 'Clupea', 'Coffea',
    'Colletotrichum', 'Cottoperca', 'Crassostrea', 'Cryptococcus', 'Cucumis', 'Cucurbita',
    'Cyanidioschyzon', 'Cynara', 'Cynoglossus', 'Daucus', 'Deinococcus', 'Denticeps',
    'Desulfovibrio', 'Dictyostelium', 'Drosophila', 'Echeneis', 'Egibacter', 'Egicoccus',
    'Elaeis', 'Equus', 'Erpetoichthys', 'Esox', 'Euzebya', 'Fervidicoccus', 'Frankia',
    'Fusarium', 'Gadus', 'Gallus', 'Gemmata', 'Gopherus', 'Gossypium', 'Gouania',
    'Helianthus', 'Ictalurus', 'Ktedonosporobacter', 'Legionella', 'Leishmania',
    'Lepisosteus', 'Leptospira', 'Limnochorda', 'Malassezia', 'Manihot', 'Mariprofundus',
    'Methanobacterium', 'Methanobrevibacter', 'Methanocaldococcus', 'Methanocella',
    'Methanopyrus', 'Methanosarcina', 'Microcaecilia', 'Modestobacter', 'Monodelphis',
    'Mus', 'Musa', 'Myripristis', 'Neisseria', 'Nitrosopumilus', 'Nitrososphaera',
    'Nitrospira', 'Nymphaea', 'Octopus', 'Olea', 'Oncorhynchus', 'Ooceraea',
    'Ornithorhynchus', 'Oryctolagus', 'Oryzias', 'Ostreococcus', 'Papaver', 'Perca',
    'Phaeodactylum', 'Phyllostomus', 'Physcomitrium', 'Plasmodium', 'Podarcis', 'Pomacea',
    'Populus', 'Prosthecochloris', 'Pseudomonas', 'Punica', 'Pyricularia', 'Pyrobaculum',
    'Quercus', 'Rhinatrema', 'Rhopalosiphum', 'Roseiflexus', 'Rubrobacter', 'Rudivirus',
    'Salarias', 'Salinisphaera', 'Sarcophilus', 'Schistosoma', 'Scleropages',
    'Sedimentisphaera', 'Sesamum', 'Solanum', 'Sparus', 'Sphaeramia', 'Spodoptera',
    'Sporisorium', 'Stanieria', 'Streptomyces', 'Strigops', 'Synechococcus', 'Takifugu',
    'Thalassiosira', 'Theileria', 'Thermococcus', 'Thermogutta', 'Thermus', 'Tribolium',
    'Trichoplusia', 'Ustilago', 'Vibrio', 'Vitis', 'Xenopus', 'Xiphophorus',
    'Zymoseptoria', 'unknown'])])

Record = collections.namedtuple('Record', ['id', 'seq'])

def parse_fasta(fasta):
    records = []
    id_ = None
    seq = ''
    with open(fasta) as f:
        for line in f:
            if line.startswith('>'):
                if (id_ is not None):
                    records.append(Record(id_, seq))
                id_ = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
        if (id_ is not None):
            records.append(Record(id_, seq))
    info(f'read in {len(records)} sequences')
    return records


def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (k == 1 and stride == 1):
        # for performance reasons
        return seq
    kmers = []
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i+k]
        if to_upper:
            kmers.append(kmer.upper())
        else:
            kmers.append(kmer)
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers.append(seq[i+k:].ljust(k, 'N'))
    return kmers


def seq2tokens(seq, token_dict, seq_length=250, max_length=None,
               k=3, stride=3, window=True, seq_len_like=None):
    """transforms raw sequence into list of tokens to be used for
    fine-tuning BERT
    NOTE: intended to be used as `custom_encode_sequence` argument for
    DataGenerators"""
    if (max_length is None):
        max_length = seq_length
    if (seq_len_like is not None):
        seq_length = min(max_length, np.random.choice(seq_len_like))
        # open('seq_lens.txt', 'a').write(str(seq_length) + ', ')
    seq = seq2kmers(seq, k=k, stride=stride, pad=True)
    if (window):
        start = randint(0, max(len(seq) - seq_length - 1, 0))
        end = start + seq_length - 1
    else:
        start = 0
        end = seq_length
    indices = [token_dict['[CLS]']] + [token_dict[word]
                                       if word in token_dict
                                       else token_dict['[UNK]']
                                       for word in seq[start:end]]
    if (len(indices) < max_length):
        indices += [token_dict['']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    segments = [0 for _ in range(max_length)]
    return [np.array(indices), np.array(segments)]

def seq_frames(seq: str, frame_len: int, all_frames=False) -> List[str]:
    """returns all frames of seq with a maximum length of `frame_len` and
    step-size 1 if `all_frames` else `frame_len`"""
    iterator = (range(len(seq) - frame_len + 1) if all_frames
                else range(0, len(seq), frame_len))
    return [seq[i:i+frame_len] for i in iterator]

def get_token_dict(alph=ALPHABET, k=3) -> dict:
    """get token dictionary dict generated from `alph` and `k`"""
    token_dict = keras_bert.get_base_dict()
    for word in [''.join(_) for _ in product(alph, repeat=k)]:
        token_dict[word] = len(token_dict)
    return token_dict

def process_bert_tokens_batch(batch_x):
    """when `seq2tokens` is used as `custom_encode_sequence`, batches
    are generated as [[input1, input2], [input1, input2], ...]. In
    order to train, they have to be transformed to [input1s,
    input2s] with this function"""
    return [np.array([_[0] for _ in batch_x]),
            np.array([_[1] for _ in batch_x])]

def load_bert(bert_path, compile_=False):
    """get bert model from path"""
    custom_objects = {'GlorotNormal': keras.initializers.glorot_normal,
                      'GlorotUniform': keras.initializers.glorot_uniform}
    custom_objects.update(keras_bert.get_custom_objects())
    model = keras.models.load_model(bert_path, compile=compile_,
                                    custom_objects=custom_objects)
    return model

def annotate_predictions(preds: List[np.ndarray],
                         overwrite_class_labels:Optional[OrderedDict]=None) -> dict:
    """annotates list of prediction arrays with provided or preset labels"""
    class_labels = (overwrite_class_labels if overwrite_class_labels is not None
                    else CLASS_LABELS)
    return {rank: {l: v.astype(float) for l, v in zip(class_labels[rank], p.transpose())}
            for rank, p in zip(class_labels, preds)}

def best_predictions(preds_annotated: dict):
    result = []
    for rank in preds_annotated:
        best = max(preds_annotated[rank], key=lambda l: preds_annotated[rank][l])
        result.append((best, preds_annotated[rank][best]))
    return result

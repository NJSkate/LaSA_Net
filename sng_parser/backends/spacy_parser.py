#! /usr/bin/env python3

from .. import database
from ..parser import Parser
from .backend import ParserBackend
import json

from tkinter import _flatten

from src.scannet_classes import SCANNET_OBJECTS, UNIQUE_SR3D_OBJECTS, SR3D_OBJECTS, REL_ALIASES, VIEW_DEP_RELS

__all__ = ['SpacyParser']

@Parser.register_backend
class SpacyParser(ParserBackend):
    """
    Scene graph parser based on spaCy.
    """

    __identifier__ = 'spacy'

    def __init__(self, model=None):
        """
        Args:
            model (str): a spec for the spaCy model. (default: en). Please refer to the
            official website of spaCy for a complete list of the available models.
            This option is useful if you are dealing with languages other than English.
        """

        try:
            import spacy
        except ImportError as e:
            raise ImportError('Spacy backend requires the spaCy library. Install spaCy via pip first.') from e

        if spacy.__version__ < '3':
            default_model = 'en'
        else:
            default_model = 'en_core_web_sm'

        self.model = model
        if self.model is None:
            self.model = default_model

        try:
            self.nlp = spacy.load(self.model)
        except OSError as e:
            raise ImportError('Unable to load the English model. Run `python -m spacy download en` first.') from e

        self.mapping_full2rio27 = json.load(open("mapping_full2rio27.json", "r"))
        self.obj_cls = list(self.mapping_full2rio27.keys())

        self.obj_cls = self.obj_cls + SCANNET_OBJECTS + UNIQUE_SR3D_OBJECTS + SR3D_OBJECTS

    def parse(self, sentence, return_doc=False):
        doc = self.nlp(sentence)

        entities = list()
        entity_chunks = list()
        for entity in doc.noun_chunks:
            ent = dict(
                span=entity.text,
                lemma_span=entity.lemma_,
                single_head=entity.root,
                head=entity.root.text,
                head_idx=[entity.root.i],
                ent_id = None,
                head_type='None',
                repeat = False,
                lemma_head=entity.root.lemma_,
                span_bounds=(entity.start, entity.end),
                modifiers=[]
            )

            for x in reversed(tuple(entity.root.children)):

                if x.dep_ == 'det':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})

                elif x.dep_ == 'nummod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})

                elif x.dep_ == 'amod':
                    ent['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})

                    for y in x.children:
                        if y.dep_ == 'amod':
                            ent['modifiers'].append({'dep': y.dep_, 'span': y.text, 'idx': y.i, 'lemma_span': y.lemma_})

                elif x.dep_ == 'compound':

                    ent['head'] = x.text + ' ' + ent['head']
                    ent['lemma_head'] = x.lemma_ + ' ' + ent['lemma_head']

                    ent['head_idx'] = [x.i] + ent['head_idx']

                    for y in x.children:
                        if y.dep_ == 'amod' or y.dep_ == 'compound':
                            ent['modifiers'].append({'dep': y.dep_, 'span': y.text, 'idx': y.i, 'lemma_span': y.lemma_})

            entities.append(ent)
            entity_chunks.append(entity)

        filter_entity_chunks =[]
        filter_entities = []
        for idx, ec in enumerate(entity_chunks):

            head_id_list = sorted(entities[idx]['head_idx'])
            if (ec.root.lemma_ in self.obj_cls) or (database.is_object_noun(ec.root.lemma_)) or\
               (entities[idx]['lemma_head'] in self.obj_cls) or (database.is_object_noun(entities[idx]['lemma_head'])) or\
               (True in [(doc[i].lemma_ in self.obj_cls) or (database.is_object_noun(doc[i].lemma_)) for i in head_id_list]):
                filter_entity_chunks.append(ec)
                entities[idx]['head_type']='Object'
                filter_entities.append(entities[idx])

            elif database.is_scene_noun(ec.root.lemma_):
                filter_entity_chunks.append(ec)
                entities[idx]['head_type']='Scene'
                filter_entities.append(entities[idx])

            elif ec.root.lemma_ in ['this', 'it', 'which', 'there', 'these', 'those']:
                filter_entity_chunks.append(ec)
                entities[idx]['head_type']='Pron'
                filter_entities.append(entities[idx])

            elif ec.root.lemma_ == 'top' and doc[head_id_list[-1] -1].text == 'counter':
                filter_entity_chunks.append(ec)
                entities[idx]['head_type']='Object'
                filter_entities.append(entities[idx])

                ent = entities[idx]
                ent['head'] = doc[head_id_list[-1] -1].text + ' ' + ent['head']
                ent['lemma_head'] = doc[head_id_list[-1] -1].lemma_ + ' ' + ent['lemma_head']

                ent['head_idx'] = [doc[head_id_list[-1] -1].i] + ent['head_idx']

        token_mask = [int(t.dep_ == 'punct') for t in doc]

        main_entity = None
        main_entity_id = 0
        tmp_entity_chunks = []
        tmp_entity = []
        for idx, entity in enumerate(filter_entity_chunks):

            if (main_entity == None) and (filter_entities[idx]['head_type']=='Object')\
                and filter_entities[idx]['lemma_head'] not in ['wall', 'floor']:
                filter_entities[idx]['ent_id'] = 0
                main_entity = filter_entities[idx]
                main_entity_id = idx
                tmp_entity_chunks.append(entity)
                tmp_entity.append(filter_entities[idx])
                break

        for idx, entity in enumerate(filter_entity_chunks):

            token_mask[entity.start:entity.end] = [1]*(entity.end - entity.start)

            if filter_entities[idx]['head_type']=='Pron' and main_entity is not None\
                and idx < main_entity_id:
                continue
            if idx == main_entity_id and main_entity is not None:
                continue

            tmp_entity_chunks.append(entity)
            tmp_entity.append(filter_entities[idx])
        filter_entity_chunks = tmp_entity_chunks
        filter_entities = tmp_entity

        ent_id = 0
        for idx, entity in enumerate(filter_entity_chunks):
            node = None

            if(idx > 0 and filter_entities[idx]['head_type']=='Pron'):

                filter_entities[idx]['ent_id'] = 0

            elif main_entity is not None\
                and filter_entities[idx]['head_type']=='Object'\
                and (filter_entities[idx]['ent_id'] == None)\
                and (filter_entities[idx]['single_head'].text == main_entity['single_head'].text)\
                and (doc[filter_entities[idx]['single_head'].i - 1].text == 'the'):
                    filter_entities[idx]['ent_id'] = 0
                    filter_entities[idx]['repeat'] = True
            else:
                if filter_entities[idx]['ent_id'] == None:
                    ent_id+=1
                    filter_entities[idx]['ent_id'] = ent_id

        graph_dege = []
        for idx_s, entity in enumerate(filter_entity_chunks):
            relation = None

            root_token, root_entity, root_conj, idx_root_ent = self.__find_root(entity, filter_entity_chunks)

            if (root_token != None) and (root_token.dep_ in ['ROOT', 'ccomp']):
                for idx_o, obj_ent in enumerate(filter_entity_chunks):
                    if (obj_ent.root.head == root_token)\
                        and (obj_ent.root.dep_ in ['nsubj', 'nsubjpass', 'attr'])\
                        and (entity != obj_ent):

                        rel, rel_idx, token_mask = self.__find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask)

                        relation = {
                                    'obj_idx': filter_entities[idx_o]['ent_id'],
                                    'object': filter_entities[idx_o]['head'],
                                    'relation': rel,
                                    'relation_idx': rel_idx,
                                    'sub_idx': filter_entities[idx_s]['ent_id'],
                                    'subject': filter_entities[idx_s]['head']
                                    }
                        if(relation['relation'] is not None):
                            graph_dege.append(relation)

            elif (root_conj != None):
                for idx_o, obj_ent in enumerate(filter_entity_chunks):
                    if (obj_ent.root.head == root_conj)\
                        and (obj_ent.root.dep_ in ['nsubj', 'nsubjpass', 'attr'])\
                        and (entity != obj_ent):

                        rel, rel_idx, token_mask = self.__find_rel_by_root(root_conj, entity, filter_entity_chunks, doc, token_mask)

                        relation = {
                                    'obj_idx': filter_entities[idx_o]['ent_id'],
                                    'object': filter_entities[idx_o]['head'],
                                    'relation': rel,
                                    'relation_idx': rel_idx,
                                    'sub_idx': filter_entities[idx_s]['ent_id'],
                                    'subject': filter_entities[idx_s]['head']
                                    }
                        if(relation['relation'] is not None):
                            graph_dege.append(relation)

            elif (root_token == None) and (root_entity != None):
                rel, rel_idx, token_mask = self.__find_rel_by_entity(root_entity, entity, filter_entity_chunks, doc, token_mask)

                relation = {
                            'obj_idx': filter_entities[idx_root_ent]['ent_id'],
                            'object': filter_entities[idx_root_ent]['head'],
                            'relation': rel,
                            'relation_idx': rel_idx,
                            'sub_idx': filter_entities[idx_s]['ent_id'],
                            'subject': filter_entities[idx_s]['head']
                            }
                if(relation['relation'] is not None):
                    graph_dege.append(relation)

            if (root_token != None) and (relation == None):
                if (root_token.i > 0) and (doc[root_token.i - 1].head != root_token) and (root_token.tag_ == 'VBN'):
                    rel, rel_idx, token_mask = self.__find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask)
                    relation = {
                            'obj_idx': 0,
                            'object': filter_entities[0]['head'],
                            'relation': rel,
                            'relation_idx': rel_idx,
                            'sub_idx': filter_entities[idx_s]['ent_id'],
                            'subject': filter_entities[idx_s]['head']
                            }
                    if(relation['relation'] is not None):
                            graph_dege.append(relation)

        for idx, entity in enumerate(filter_entity_chunks):

            i_list = []
            if entity.root.head.dep_ == 'ROOT':
                root_token = entity.root.head
                i_flag = []
                i_list = find_modify(root_token.i, token_mask, doc, i_list, i_flag)

            if (entity.root.i < len(doc)-1):
                if (doc[entity.root.i+1].dep_ == 'prep') and (doc[entity.root.i+1].head == entity.root):
                    root_token = doc[entity.root.i+1]
                    i_list = []
                    i_flag = []
                    i_list = find_modify(root_token.i, token_mask, doc, i_list, i_flag)

            if len(i_list)>1:
                for i in i_list:
                    x = doc[i]
                    filter_entities[idx]['modifiers'].append({'dep': x.dep_, 'span': x.text, 'idx': x.i, 'lemma_span': x.lemma_})\

        nodes = []
        for idx, ent in enumerate(filter_entities):
            if (ent['head_type'] != 'Pron') and (ent['repeat'] == False):
                target_mask = [0] * len(token_mask)
                mod_mask    = [0] * len(token_mask)
                pron_mask   = [0] * len(token_mask)
                target_char_span = []
                mod_char_span = []
                pron_char_span = []

                for id in ent['head_idx']:

                    target_mask[id] = 1

                    span = find_char_span_by_token_idx(id, doc)
                    if span:
                        target_char_span.append(span)

                mod_text = []
                for mod in ent['modifiers']:
                    if doc[mod['idx']].tag_ in ['VBZ', 'DT', 'CC']:
                        continue

                    mod_mask[mod['idx']] = 1
                    mod_text.append(mod['span'])

                    span = find_char_span_by_token_idx(mod['idx'], doc)
                    if span:
                        mod_char_span.append(span)

                node = {
                    'node_id': ent['ent_id'],
                    'target' : ent['head'],
                    'lemma_head': ent['lemma_head'],
                    'node_type': ent['head_type'],
                    'target_mask':target_mask,
                    'mod_text': mod_text,
                    'mod_mask': mod_mask,
                    'pron_mask': pron_mask,
                    'target_char_span': target_char_span,
                    'mod_char_span': mod_char_span,
                    'pron_char_span': pron_char_span
                }

                nodes.append(node)

            elif main_entity is not None and idx > 0 and len(nodes) > 0 and ent['ent_id'] == 0 and\
                 (ent['head_type'] == 'Pron' or (ent['repeat'] == True)):
                pron_mask = nodes[0]['pron_mask']

                for id in ent['head_idx']:
                    pron_mask[id] = 1

                    span = find_char_span_by_token_idx(id, doc)
                    if span:
                        nodes[0]['pron_char_span'].append(span)

                for mod in ent['modifiers']:
                    if doc[mod['idx']].tag_ in ['VBZ', 'DT', 'CC']:
                        continue

                    nodes[0]['mod_mask'][mod['idx']] = 1
                    nodes[0]['mod_text'].append(mod['span'])

                    span = find_char_span_by_token_idx(mod['idx'], doc)
                    if span:
                        nodes[0]['mod_char_span'].append(span)

                nodes[0]['pron_mask'] = pron_mask

        if main_entity is None:

            for token in doc:
                if(token.text == 'trash' and doc[token.i+1].text == 'can') or\
                  (token.text == 'urinal' or token.text == 'cardboard' ):
                    target_mask = [0] * len(token_mask)
                    mod_mask    = [0] * len(token_mask)
                    pron_mask   = [0] * len(token_mask)
                    target_char_span = []
                    mod_char_span = []
                    pron_char_span = []

                    head_name = ''
                    if(token.text == 'trash' and doc[token.i+1].text == 'can'):

                        head_idx = [token.i, token.i+1]
                        for id in head_idx:
                            target_mask[id] = 1

                            span = find_char_span_by_token_idx(id, doc)
                            if span:
                                target_char_span.append(span)
                        head_name = 'trash can'
                    elif (token.text == 'urinal' or token.text == 'cardboard' ):
                        target_mask[token.i] = 1

                        span = find_char_span_by_token_idx(token.i, doc)
                        if span:
                            target_char_span.append(span)
                        head_name = token.text

                    mod_text = []

                    node = {
                        'node_id': 0,
                        'target' : head_name,
                        'lemma_head': head_name,
                        'node_type': 'Object',
                        'target_mask':target_mask,
                        'mod_text': mod_text,
                        'mod_mask': mod_mask,
                        'pron_mask': pron_mask,
                        'target_char_span': target_char_span,
                        'mod_char_span': mod_char_span,
                        'pron_char_span': pron_char_span
                    }

                    nodes.insert(0, node)
                    break

        for idx, node in enumerate(nodes):
            node['rel_char_span'] = []

            if idx == 0 and node['node_id'] == 0:
                rel_id = []
                for edge in graph_dege:
                    if edge['obj_idx'] == 0:
                        rel_id.append(edge['relation_idx'])

                rel_id = list(_flatten(rel_id))

                for id in rel_id:

                    span = find_char_span_by_token_idx(id, doc)
                    if span:
                        node['rel_char_span'].append(span)

        return nodes, graph_dege

    @staticmethod
    def __locate_noun(chunks, i):
        for j, c in enumerate(chunks):
            if c.start <= i < c.end:
                return j
        return None

    @staticmethod
    def __find_root(entity, filter_entity_chunks):
        root_token = None
        root_entity = None
        root_conj = None
        idx = None

        tmp = entity.root
        ent_i_list = [e.root.i for e in filter_entity_chunks]

        if entity.root.dep_ == 'conj' and entity.root.head.i not in ent_i_list:
            return root_token, root_entity, root_conj, idx

        for i in range(10):

            if(tmp.head.i in ent_i_list) and (tmp.dep_ != 'conj'):
                idx = ent_i_list.index(tmp.head.i)
                root_entity = tmp.head
                return root_token, root_entity, root_conj, idx

            if tmp.head.dep_ == 'conj' and tmp.head.tag_ in ['VBZ']:
                idx = tmp.head.i
                root_conj = tmp.head
                return root_token, root_entity, root_conj, idx

            if tmp.dep_ == 'ccomp' and tmp.head.dep_ == 'ROOT':
                root_token = tmp
                return root_token, root_entity, root_conj, idx

            if(tmp.dep_ != 'ROOT'):
                tmp = tmp.head
            else:

                root_token = tmp
                return root_token, root_entity, root_conj, idx

        return root_token, root_entity, root_conj, idx

    @staticmethod
    def __find_rel_by_root(root_token, entity, filter_entity_chunks, doc, token_mask):
        rel = None
        rel_idx = []

        token = entity.root.head

        if(entity.root.dep_ == 'conj'):
            token = token.head

        end = False
        while(not end):
            if(token == root_token):

                if rel == None:
                    rel = token.text
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1
                elif (token.tag_ in ['VBN','VBZ','VBG']):
                    rel = rel
                    token_mask[token.i] = 1
                else:
                    rel = token.text + ' ' + rel
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1

                return rel, rel_idx, token_mask

            else:
                if rel == None:
                    rel = token.text
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1
                elif (token.tag_ in ['VBN','VBZ','VBG']):
                    rel = rel
                else:
                    rel = token.text + ' ' + rel
                    rel_idx = [token.i] + rel_idx
                    token_mask[token.i] = 1

                    rel, rel_idx, token_mask = find_children(rel, rel_idx, token, doc, token_mask)

                token = token.head

        return rel, rel_idx, token_mask

    @staticmethod
    def __find_rel_by_entity(root_entity, sub_entity, filter_entity_chunks, doc, token_mask):
        rel = None
        rel_idx = []

        iter_token = sub_entity.root.head

        end = False
        while(not end):
            if(iter_token == root_entity):
                return rel, rel_idx, token_mask
            else:
                if rel == None:
                    rel = iter_token.text
                    rel_idx = [iter_token.i] + rel_idx
                    token_mask[iter_token.i] = 1
                elif (iter_token.tag_ in ['VBN','VBZ','VBG']):
                    rel = rel
                else:
                    rel = iter_token.text + ' ' + rel
                    rel_idx = [iter_token.i] + rel_idx
                    token_mask[iter_token.i] = 1

                    rel, rel_idx, token_mask = find_children(rel, rel_idx, iter_token, doc, token_mask)

                iter_token = iter_token.head

        return rel, rel_idx, token_mask

def find_children(rel, rel_idx, token, doc, token_mask):
    for i in range(1, 10):
        if doc[token.i - i].head == token:
            rel = doc[token.i - i].text + ' ' + rel
            rel_idx = [token.i - i] + rel_idx
            token_mask[token.i - i] = 1
            continue
        else:
            return rel, rel_idx, token_mask
    return rel, rel_idx, token_mask

def find_modify(first_token_i, token_mask, doc, i_list, i_flag):

    children_list = [c for c in doc[first_token_i].children]
    for child in children_list:
        if (token_mask[child.i] == 0) and (doc[child.i].tag_ not in ['EX']):

            i_list.append(child.i)
            i_flag.append(0)

    for idx, i in enumerate(i_list):
        if i_flag[idx] == 0:
            i_flag[idx] = 1
            find_modify(i, token_mask, doc, i_list, i_flag)

    if token_mask[first_token_i] == 1:
        return sorted(i_list)
    else:
        return sorted([first_token_i]+i_list)

def find_char_span_by_token_idx(id, doc):
    '''
    input:
        doc: full text
        id:  the word idx of the token
    output:
        char span: [char_star, char_end]

    id:    0    1   2  3   4   5    6   7   8   9   10  11 12 13   14  15  17      18
    eg:   the chair is on the left side of the table , and to the left of another chair
    span: 012345678......
    the input idx of the word 'chair' is 1
    the output char span is [4,9]
    '''
    doc_text = doc.text + ' ABCDEF'
    token_list = doc_text.split()

    token_text = doc[id].text

    if id >= len(token_list) or token_text not in token_list[id]:

        return []
    if token_text != token_list[id]:
        return []

    new_sen = ' '.join(token_list[id:])

    char_star = doc_text.find(new_sen)

    lenth_ = len(token_text)

    char_end = char_star + lenth_

    return [char_star, char_end]

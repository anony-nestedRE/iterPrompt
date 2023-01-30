# -*- coding: utf-8 -*-
# @Time    : 
# @Author  : 
# @Site    : 
# @File    : prompt_policy.py
# @Software: PyCharm

# 

import os
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import set_random, compute_metric, calc_running_avg_loss, save_config, compute_metric_by_dist, print_dist
from utils import compute_metric_relation, compute_metric_relation_guided, compute_metric_by_dist_layers, print_config
from data_utils import get_type_info, NDataSet, generate_ne_entity_replace2, generate_ne_entity_replace, save_predict
import datetime
import time
import copy, json, random
from torch.nn.utils import clip_grad_norm_
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import argparse


class PromptModel(nn.Module):
    def __init__(self, bert_path):
        super(PromptModel, self).__init__()
        self.MLM = BertForMaskedLM.from_pretrained(bert_path)

    def forward(self, x, mask):
        # ============================================MaskedLMOutput====================================================
        # ======Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).======
        # =========================================type: torch.FloatTensor==============================================
        # ==========================shape: (batch_size, sequence_length, config.vocab_size)=============================
        out = self.MLM(x, attention_mask=mask)[0]
        # ==========================================================================================================
      
        return out


class PromptAuto(object):
    def __init__(self, config):
        super(PromptAuto, self).__init__()
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        self.rel_type = [x[0] for x in self.relations]

        self.model_dir = os.path.join(config.model_dir, 'prompt-auto', f'template{config.template}')
        os.makedirs(self.model_dir, exist_ok=True)
        # self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

        self.model = PromptModel(config.bert_dir).to(config.device)
        # self.fgm = FGM(self.model, 'word_embeddings.')
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.type2word = {tp: '[unused%d]' % (i + 1) for i, tp in enumerate(self.all_type + ['no'])}
        self.rel_type_word = [
            '[unused%d]' % (len(self.all_type) - len(self.relations) + i + 1) for i in range(len(self.rel_type) + 1)]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)
        self.rel_id_word = ['[unused%d]' % (i + self.config.rel_start) for i in range(1, self.config.rel_elem_num + 1)]
        # self.rel_word2id = [(w, i) for w, i in zip(self.rel_id_word, self.rel_ids)]
        self.rel_type = self.rel_type + ['no']

        self.raw_tmp_file = '%s.json'

    def _setup_train(self, lr=1e-5):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            print('no trained model %s' % model_file)
            return
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def generate_candidate_train(self, layers, elements):
        candidates = []
        all_type, relations = get_type_info(self.config.type_file)
        left_type = []
        right_type = []
        for r in relations:
            for ops in r[1]:
                left_type.extend(ops[0])
                right_type.extend(ops[1])
        left_type = set(left_type)
        right_type = set(right_type)
        for e1 in layers[-1]:
            for l in layers:
                for e2 in l:
                    if e1 != e2 and elements[e1][0] in left_type and elements[e2][0] in right_type:
                        candidates.append((e1, e2))
        for l in layers[:-1]:
            for e1 in l:
                for e2 in layers[-1]:
                    if e1 != e2 and elements[e1][0] in left_type and elements[e2][0] in right_type:
                        candidates.append((e1, e2))
        return list(sorted(set(candidates)))

    def convert_entity(self, words, entities):
        converted_entities = []
        c_idx2w_idx = []  # character idx -> word idx
        for i in range(len(words)):
            c_idx2w_idx.extend([i] * len(words[i]))
        for e in entities:
            converted_entities.append((e[0], (c_idx2w_idx[e[1][0] + 5], c_idx2w_idx[e[1][1] + 5])))
        return converted_entities

    def convert_train_data2prompt(self, data_dir, file_name):
        prompt_file = os.path.join(self.model_dir, 'prompt-%s' % file_name)
        ex = [json.loads(l) for l in open(os.path.join(data_dir, file_name), 'r', encoding='utf-8').readlines()]
        prompt_ex = []
        for e in ex:
            if self.config.rel_mode == 2:
                random.shuffle(self.rel_id_word)
            sentence = ['[CLS]'] + self.tokenizer.tokenize(e['sentence']) + ['[SEP]']
            e['element'] = e['entity'] + e['element'][len(e['entity']):]
            c_entities = self.convert_entity(sentence[:-1], e['entity'])
            layer = [0] * len(e['element'])
            for i in range(len(e['entity']), len(e['element'])):
                layer[i] = 1 + max(layer[e['element'][i][1][0]], layer[e['element'][i][1][1]])
            layer_elem = [[] for _ in range(max(layer) + 1)]
            for i in range(len(layer)):
                layer_elem[layer[i]].append(i)
            c2label = {(elem[1][0], elem[1][1]): elem[0] for elem in e['element'][len(e['entity']):]}
            cur_sentence = copy.deepcopy(sentence)
            for i in range(len(layer_elem)):
                candidates = self.generate_candidate_train(layer_elem[:i+1], e['element'])
                if i != 0:
                    for j in range(len(layer_elem[i])):
                        elem = e['element'][layer_elem[i][j]]
                        if layer_elem[i][j]-len(c_entities) + 1 >= self.config.rel_elem_num:
                            continue
                        type_w = self.type2word[elem[0]]
                        left = ([self.type2word[c_entities[elem[1][0]][0]]]
                                + sentence[c_entities[elem[1][0]][1][0]:c_entities[elem[1][0]][1][1] + 1]) \
                            if elem[1][0] < len(c_entities) \
                            else [self.rel_id_word[(elem[1][0] - len(c_entities) + 1)]]
                        right = ([self.type2word[c_entities[elem[1][1]][0]]]
                                 + sentence[c_entities[elem[1][1]][1][0]:c_entities[elem[1][1]][1][1] + 1]) \
                            if elem[1][1] < len(c_entities) \
                            else [self.rel_id_word[(elem[1][1] - len(c_entities) + 1)]]
                        if self.config.rel_place == 0:  
                            cur_sentence = cur_sentence \
                                           + [self.rel_id_word[(layer_elem[i][j] - len(c_entities) + 1)]] \
                                           + left + [type_w] + right + ['[SEP]']
                        elif self.config.rel_place == 1:  
                            cur_sentence = cur_sentence + left \
                                           + [self.rel_id_word[(layer_elem[i][j] - len(c_entities) + 1)]] \
                                           + [type_w] + right + ['[SEP]']
                        else:  # 
                            cur_sentence = cur_sentence + left + [type_w] + right \
                                           + [self.rel_id_word[(layer_elem[i][j] - len(c_entities) + 1)]] + ['[SEP]']
                tmp_sentence = copy.deepcopy(cur_sentence)
                cc = 0
                tmp_label = copy.deepcopy(cur_sentence)
                for c in candidates:
                    if c[0] - len(e['entity']) + 1 >= self.config.rel_elem_num \
                            or c[1] - len(e['entity']) + 1 >= self.config.rel_elem_num:
                        continue
                    left = ([self.type2word[c_entities[c[0]][0]]]
                            + sentence[c_entities[c[0]][1][0]:c_entities[c[0]][1][1] + 1]) \
                        if layer[c[0]] == 0 else [self.rel_id_word[(c[0] - len(e['entity']) + 1)]]
                    right = ([self.type2word[c_entities[c[1]][0]]]
                             + sentence[c_entities[c[1]][1][0]:c_entities[c[1]][1][1] + 1]) \
                        if layer[c[1]] == 0 else [self.rel_id_word[(c[1] - len(e['entity']) + 1)]]
                    # cur_c = left + ['[MASK]'] + right + ['[SEP]']
                    if self.config.template == 2:
                        cur_c = left + ['对'] + right + ['是', '[MASK]', '关', '系', '[SEP]']
                    else:
                        cur_c = left + ['[MASK]'] + right + ['[SEP]']
                    if len(tmp_sentence) + len(cur_c) > self.config.max_pro_len:
                        cc += 1
                        prompt_ex.append(
                            {'sid': '%s-l-%d-%d' % (e['sid'], i+1, cc), 'sentence': tmp_sentence, 'label': tmp_label})
                        tmp_sentence = copy.deepcopy(cur_sentence)
                        tmp_label = [self.tokenizer.pad_token]*len(tmp_sentence)
                    tmp_sentence = tmp_sentence + cur_c
                    if self.config.template == 2:
                        tmp_label.extend(left + ['对'] + right + ['是', self.type2word[c2label.get(c, 'no')], '关', '系', '[SEP]'])
                    else:
                        tmp_label.extend(left + [self.type2word[c2label.get(c, 'no')]] + right + ['[SEP]'])
                    assert len(tmp_sentence) == len(tmp_label)
                if tmp_label:
                    prompt_ex.append(
                        {'sid': '%s-l-%d-%d' % (e['sid'], i + 1, cc + 1), 'sentence': tmp_sentence, 'label': tmp_label})
        writer = open(prompt_file, 'w', encoding='utf-8', newline='')
        for e in prompt_ex:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()

    def convert_test_prompt(self, sentence, entities, elements):
        c_entities = self.convert_entity(sentence[:-1], entities)
        layer = [0] * len(elements)
        for i in range(len(entities), len(elements)):
            layer[i] = 1 + max(layer[elements[i][1][0]], layer[elements[i][1][1]])
        layer_elem = [[] for _ in range(max(layer)+1)]
        for i in range(len(layer)):
            layer_elem[layer[i]].append(i)
        cur_sentence = copy.deepcopy(sentence)
        for i, l in enumerate(layer):
            if l != 0:
                if i-len(entities)+1 >= self.config.rel_elem_num:
                    continue
                elem = elements[i]
                type_w = self.type2word[elem[0]]
                left = ([self.type2word[c_entities[elem[1][0]][0]]]
                        + sentence[c_entities[elem[1][0]][1][0]:c_entities[elem[1][0]][1][1]+1]) \
                    if elem[1][0] < len(c_entities) \
                    else [self.rel_id_word[(elem[1][0]-len(c_entities)+1)]]
                right = ([self.type2word[c_entities[elem[1][1]][0]]] +
                         sentence[c_entities[elem[1][1]][1][0]:c_entities[elem[1][1]][1][1]+1]) \
                    if elem[1][1] < len(c_entities) \
                    else [self.rel_id_word[(elem[1][1]-len(c_entities)+1)]]
                if self.config.rel_place == 0:  # 
                    cur_sentence = cur_sentence \
                                   + [self.rel_id_word[(i-len(c_entities)+1)]] \
                                   + left + [type_w] + right + ['[SEP]']
                elif self.config.rel_place == 1:  # 
                    cur_sentence = cur_sentence + left \
                                   + [self.rel_id_word[(i-len(c_entities)+1)]] \
                                   + [type_w] + right + ['[SEP]']
                else:  # 
                    cur_sentence = cur_sentence + left + [type_w] + right \
                                   + [self.rel_id_word[(i-len(c_entities)+1)]] + ['[SEP]']
                # cur_sentence = cur_sentence + [self.rel_id_word[(i-len(c_entities)+1)]] \
                #                + left + [type_w] + right + ['[SEP]']
        prompt_ex = []
        candidates = self.generate_candidate_train(layer_elem, elements)
        tmp_sentence = copy.deepcopy(cur_sentence)
        tmp_c = []
        for c in candidates:
            if c[0] - len(entities) + 1 >= self.config.rel_elem_num \
                    or c[1] - len(entities) + 1 >= self.config.rel_elem_num:
                continue
            left = ([self.type2word[c_entities[c[0]][0]]]
                    + sentence[c_entities[c[0]][1][0]:c_entities[c[0]][1][1]+1]) \
                if layer[c[0]] == 0 else [self.rel_id_word[(c[0]-len(c_entities)+1)]]
            right = ([self.type2word[c_entities[c[1]][0]]]
                     + sentence[c_entities[c[1]][1][0]:c_entities[c[1]][1][1]+1]) \
                if layer[c[1]] == 0 else [self.rel_id_word[(c[1]-len(c_entities)+1)]]
            if self.config.template == 2:
                cur_c = left + ['对'] + right + ['是', '[MASK]', '关', '系', '[SEP]']
            else:
                cur_c = left + ['[MASK]'] + right + ['[SEP]']
            # cur_c = left + ['[MASK]'] + right + ['[SEP]']
            if (len(tmp_sentence) + len(cur_c)) > self.config.max_pro_len >= len(tmp_sentence) and len(tmp_c) != 0:
                prompt_ex.append(
                    {'sentence': tmp_sentence+[self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                     'candidates': tmp_c})
                tmp_sentence = copy.deepcopy(cur_sentence)
                tmp_c = []
            tmp_sentence = tmp_sentence + cur_c
            tmp_c.append(c)
        if tmp_c and len(tmp_sentence) <= self.config.max_pro_len:
            prompt_ex.append(
                {'sentence': tmp_sentence+[self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                 'candidates': tmp_c})
        return prompt_ex

    def get_train_loader(self, data_file):
        def collate_fn(batch):
            sentence, word_ids, label, sid = [], [], [], []
            for idx, line in enumerate(batch):
                item = json.loads(line)
                t_words = self.tokenizer.convert_tokens_to_ids(item['sentence'])
                t_label = self.tokenizer.convert_tokens_to_ids(item['label'])
                word_ids.append(t_words[:self.config.max_pro_len]+[
                    self.tokenizer.pad_token_id]*(self.config.max_pro_len-len(t_words)))
                label.append(t_label[:self.config.max_pro_len]+[
                    self.tokenizer.pad_token_id]*(self.config.max_pro_len-len(t_words)))
                sentence.append(''.join(item['sentence']))
                sid.append(item['sid'])
            return {'sentence': sentence, 'ids': word_ids, 'label': label, 'sid': sid}

        _collate_fn = collate_fn
        print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=True, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def get_other_loader(self, data_file):
        def collate_fn(batch):
            sentence, sid, words, entities, elements = [], [], [], [], []
            for idx, line in enumerate(batch):
                item = json.loads(line)
                sentence.append(item['sentence'])
                t_words = ['[CLS]'] + self.tokenizer.tokenize(item['sentence']) + ['[SEP]']
                words.append(t_words[:self.config.max_pro_len])
                sid.append(item['sid'])
                entities.append([(e[0], (e[1][0], e[1][1])) for e in item['entity']])
                elements.append([(e[0], (e[1][0], e[1][1])) for e in item['entity']]
                                +[(e[0], tuple(e[1])) for e in item['element'][len(item['entity']):]])
            return {'sentence': sentence, 'words': words, 'entity': entities, 'element': elements, 'sid': sid}

        _collate_fn = collate_fn
        # print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=False, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def eval(self, op, data_file=None):
        all_predict, all_elem, all_entity = self.predict(op, data_file)
        if op == self.config.test_prefix:
            ex = [json.loads(l) for l in
                  open(os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)]
            all_examples = []
            assert len(ex) == len(all_elem)
            for e, entity, element in zip(ex, all_entity, all_elem):
                all_examples.append({'sid': e['sid'], 'entity': entity, 'element': element})
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, self.rel_type[:-1])
        print_dist(mp, mr, mf1, self.rel_type[:-1],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            g_predict, g_elem, g_entity = self.predict_guided(op, data_file)
            gp, gr, gf1 = compute_metric_by_dist_layers(g_elem, g_predict, self.rel_type[:-1])
            print('#'*5, 'guide result', '#'*5)
            print_dist(gp, gr, gf1, self.rel_type[:-1],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
        return p, r, f1

    def predict(self, op, data_file=None):
        data_iter = self.get_other_loader(
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity = [], [], []
        elem_c = 0
        self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    all_elem.append(batch['element'][i])
                    all_entity.append(batch['entity'][i])
                    cur_elem = copy.deepcopy(batch['entity'][i])
                    layer = 0
                    while layer <= 6:
                        prompt_ex = self.convert_test_prompt(batch['words'][i], batch['entity'][i], cur_elem)
                        flag = True
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            prob = self.model(ids, mask).squeeze(0)
                            cand_index = ids.squeeze(0).eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_prob = prob[cand_index.tolist()][:, self.rel_type_ids[0]:self.rel_type_ids[-1]+1]
                            cand_prob = cand_prob.float().softmax(-1)
                            cand_label = [self.rel_type[x] for x in cand_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_elem) < self.config.rel_elem_num:
                                    cur_elem.append((label, e['candidates'][li]))
                                    flag = False
                        if flag:
                            break
                        else:
                            layer += 1
                    all_predict.append(cur_elem[len(batch['entity'][i]):])
                    elem_c += len(cur_elem) - len(batch['entity'][i])
        return all_predict, all_elem, all_entity

    def predict_prob(self, op, data_file=None):
        data_iter = self.get_other_loader(
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} predict batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity, all_prob = [], [], [], []
        elem_c = 0
        if op == 'train':
            self.model.train()
        else:
            self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    all_elem.append(batch['element'][i])
                    all_entity.append(batch['entity'][i])
                    cur_elem = copy.deepcopy(batch['entity'][i])
                    cur_prob = []
                    layer = 0
                    while layer <= 6:
                        prompt_ex = self.convert_test_prompt(batch['words'][i], batch['entity'][i], cur_elem)
                        flag = True
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            prob = self.model(ids, mask).squeeze(0)
                            cand_index = ids.squeeze(0).eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_prob = prob[cand_index.tolist()][:, self.rel_type_ids[0]:self.rel_type_ids[-1] + 1]
                            cand_prob = cand_prob.float().softmax(-1)
                            cand_label = [self.rel_type[x] for x in cand_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_elem) < self.config.rel_elem_num:
                                    cur_elem.append((label, e['candidates'][li]))
                                    cur_prob.append(cand_prob[li].tolist())
                                    flag = False
                        if flag:
                            break
                        else:
                            layer += 1
                    all_predict.append(cur_elem[len(batch['entity'][i]):])
                    all_prob.append(cur_prob)
                    elem_c += len(cur_elem) - len(batch['entity'][i])
        return all_predict, all_elem, all_entity, all_prob

    def predict_guided(self, op, data_file=None):
        data_iter = self.get_other_loader(
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity = [], [], []
        self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    # all_elem.append(batch['element'][i])
                    all_entity.append(batch['entity'][i])
                    layer = [0] * len(batch['element'][i])
                    for k in range(len(batch['entity'][i]), len(batch['element'][i])):
                        layer[k] = 1+max(layer[batch['element'][i][k][1][0]], layer[batch['element'][i][k][1][1]])
                    layer_elem = [[] for _ in range(max(layer)+1)]
                    for k in range(len(layer)):
                        layer_elem[layer[k]].append(batch['element'][i][k])
                    all_elem.append(layer_elem[1:])
                    cur_gold, cur_predict = [], []
                    for l in range(1, max(layer)+1):
                        cur_predict.append([])
                        cur_gold.extend(layer_elem[l-1])
                        try:
                            prompt_ex = self.convert_test_prompt(
                                batch['words'][i], batch['entity'][i], copy.deepcopy(cur_gold))
                        except:
                            print(batch['sid'][i], l, layer, cur_gold)
                            prompt_ex = []
                        # prompt_ex = self.convert_test_prompt(
                        #     batch['words'][i], batch['entity'][i], copy.deepcopy(cur_gold))
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            prob = self.model(ids, mask).squeeze(0)
                            cand_index = ids.squeeze(0).eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_prob = prob[cand_index.tolist()][:, self.rel_type_ids[0]:self.rel_type_ids[-1] + 1]
                            cand_prob = cand_prob.float().softmax(-1)
                            cand_label = [self.rel_type[x] for x in cand_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_gold) < self.config.rel_elem_num:
                                    cur_predict[-1].append((label, e['candidates'][li]))
                                    # flag = False
                    all_predict.append(cur_predict)
        return all_predict, all_elem, all_entity

    def train(self, data_file=None):
        self._setup_train(self.config.bert_lr)
        self.convert_train_data2prompt(
            self.data_dir, (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)
        train_iter = self.get_train_loader(
            os.path.join(
                self.model_dir,
                'prompt-%s' % ((self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)))
        batch_len = len(train_iter)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            print('[INFO] {} | train epoch = {}|{}'.format(datetime.datetime.now(), e, self.config.train_epoch))
            if self.config.rel_mode == 1:
                print('shuffle relation id words and recreate train file')
                random.shuffle(self.rel_id_word)
            if self.config.rel_mode in [1, 2]:
                print('convert new train file')
                self.convert_train_data2prompt(
                    self.data_dir, (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)
                train_iter = self.get_train_loader(
                    os.path.join(
                        self.model_dir,
                        'prompt-%s' % (
                            (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)))
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                label = t.tensor(batch['label'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob = self.model(ids, mask)
                loss = F.cross_entropy(prob.reshape([-1, prob.shape[-1]]), label.reshape([-1]), reduce=False)
                loss = loss * ids.eq(self.tokenizer.mask_token_id).reshape([-1])
                loss = loss.sum() / ids.eq(self.tokenizer.mask_token_id).sum()
                loss.backward()

                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)


class PromptManual(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        # ============================================rel_type===============================================
        # rel_type = [>, >=, <, <=, =, +, -, /, has, and, @, no]
        self.rel_type = [x[0] for x in self.relations] + ['no']
        # ===========================================entity_type=============================================
        # entity_type = ["Main", "Main-q", "Labor", "Labor-q", "Service", "Place", "Rate", "RateV", "Fund", "FundV",
        # "Time", "TimeV", "Base", "BaseV"]
        self.entity_type = list(filter(lambda x: x not in self.rel_type, self.all_type))

        self.model_dir = os.path.join(config.model_dir, 'prompt-manual', f'template{config.template}')
        os.makedirs(self.model_dir, exist_ok=True)
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

        self.model = PromptModel(config.bert_dir).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        # ===========================entityType2word==================================================
        # entityType2word = {"Main": '[unused1]', "Main-q": '[unused2]', .........}
        self.entityType2word = {tw: '[unused%d]' % (i+1) for i, tw in enumerate(self.entity_type)}
        # ====================relTypeLabelWord=====================================
        self.relTypeLabelWord = {'>': ['>'], '>=': ['≥', '≧'],
                                 '<': ['<'], '<=': ['≤', '≦'], '=': ['='],
                                 '+': ['加'], '-': ['减'], '/': ['除', '率'],
                                 '@': ['属', '之', '的'], 'has': ['有', '含'],
                                 'and': ['与', '和', '并'], 'no': ['无', '非', '没', '不']}
        # ======================================================
        for tp in self.rel_type:
            self.relTypeLabelWord[tp] = self.relTypeLabelWord[tp][:1]
        # =================================all_rel_lb==============================================
        self.all_rel_lb = []
        for tp in self.rel_type:
            self.all_rel_lb.extend(self.relTypeLabelWord[tp])
        #
        self.all_rel_lb_id = self.tokenizer.convert_tokens_to_ids(self.all_rel_lb)
        # =========================================================================
        # relTypeLabelWord = {'>': [0], '>=': [1, 2],
        #                     '<': [3], '<=': [4, 5], '=': [6],
        #                     '+': [7], '-': [8], '/': [9, 10],
        #                     '@': [11, 12, 13], 'has': [14, 15],
        #                     'and': [16, 17, 18], 'no': [19, 20, 21, 22]}
        self.relTypeLabelWordId = {k: [self.all_rel_lb.index(w) for w in v] for k, v in self.relTypeLabelWord.items()}
        # =============================[r_id = k] <---> [unused(rel_start+k)]===================
        # relIdentityWord = [unused101, unused102, ...., unused170]
        # ...
        self.relIdentityWord = ['[unused%d]' % (i+self.config.rel_start) for i in range(1, self.config.rel_elem_num+1)]

        self.raw_tmp_file = '%s.json'

    def _setup_train(self, lr=1e-5):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            print('no trained model %s' % model_file)
            return
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def load_model(self, model_file):
        self._load_model(model_file)

    def convert_entity(self, words, entities):
        """          """
        converted_entities = []
        # c_idx2w_idx = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 13]
        c_idx2w_idx = []  # character idx -> word idx
        for i in range(len(words)):
            c_idx2w_idx.extend([i] * len(words[i]))
        for e in entities:
            converted_entities.append((e[0], (c_idx2w_idx[e[1][0] + 5], c_idx2w_idx[e[1][1] + 5])))  # 5代表[CLS]的长度
        return converted_entities

    def generate_candidate_train(self, layers, elements):
        """ """
        candidates = []
        all_type, relations = get_type_info(self.config.type_file)
        left_type = []
        right_type = []
        for r in relations:
            for ops in r[1]:
                left_type.extend(ops[0])
                right_type.extend(ops[1])
        left_type = set(left_type)
        right_type = set(right_type)
        for e1 in layers[-1]:
            for l in layers:
                for e2 in l:
                    # 
                    if e1 != e2 and elements[e1][0] in left_type and elements[e2][0] in right_type:
                        # 
                        candidates.append((e1, e2))
        for l in layers[:-1]:
            for e1 in l:
                for e2 in layers[-1]:
                    if e1 != e2 and elements[e1][0] in left_type and elements[e2][0] in right_type:
                        candidates.append((e1, e2))
        return list(sorted(set(candidates)))

    def convert_train_data2prompt(self, data_dir, file_name):
        """
        """
        prompt_file = os.path.join(self.model_dir, 'prompt-%s' % file_name)
        ex = [json.loads(l) for l in open(os.path.join(data_dir, file_name), 'r', encoding='utf-8').readlines()]
        prompt_ex = []
        for e in ex:
            # ================================================================================
            if self.config.rel_mode == 2:
                random.shuffle(self.relIdentityWord)
            sentence = ['[CLS]'] + self.tokenizer.tokenize(e['sentence']) + ['[SEP]']
            # ent_len = 3, elem_len = 5
            ent_len, elem_len = len(e['entity']), len(e['element'])
            e['element'] = e['entity'] + e['element'][ent_len:]

            # ============================================c_entities================================
            # c_entities = [("Labor", (1, 2)), ("Labor", (6, 10)), ("RateV", (12, 13))]
            c_entities = self.convert_entity(sentence[:-1], e['entity'])
            # =======================================layer==========================================
            # layer = [0, 0, 0, 0, 0]
            layer = [0] * elem_len
            for i in range(ent_len, elem_len):
                layer[i] = 1 + max(layer[e['element'][i][1][0]], layer[e['element'][i][1][1]])
            # ===========================================layer==========================================================

            # =======================================layer_elem========================================
            # layer_elem = [[], [], []]
            layer_elem = [[] for _ in range(max(layer)+1)]
            for i in range(len(layer)):
                # layer_elem = [[0, 1, 2], [3], [4]]
                layer_elem[layer[i]].append(i)
            # ===========================================layer_elem=====================================================

            # ==================================c2label======================================
            # c2label = {
            #     (0, 1): "/",
            #     (3, 2): ">=",
            # }
            # ==========================================c2label=========================================================
            c2label = {(elem[1][0], elem[1][1]): elem[0] for elem in e['element'][len(e['entity']):]}
            # =================================
            cur_sentence = copy.deepcopy(sentence)

            # ==================================================================================
            for i in range(len(layer_elem)):
                candidates = self.generate_candidate_train(layer_elem[:i+1], e['element'])
                
                # iteratively add the extracted triple r^l = <El_i, yrt, El_j> into T_iter(s) for the next round RE
                if i != 0:
                    # ***********************************************************************
                    for j in range(len(layer_elem[i])):
                        # 
                        if layer_elem[i][j]-len(c_entities) + 1 >= self.config.rel_elem_num:
                            continue
                        # =====================================elem=====================================
                        elem = e['element'][layer_elem[i][j]]
                        # =====================================type_w================================
                        type_w = self.relTypeLabelWord[elem[0]][0]
                        # ==============================================================================================
                        
                        left = ([self.entityType2word[c_entities[elem[1][0]][0]]] 
                                + sentence[c_entities[elem[1][0]][1][0]:c_entities[elem[1][0]][1][1] + 1]) \
                            if elem[1][0] < len(c_entities) else [self.relIdentityWord[(elem[1][0]-ent_len+1)]]
                        # ==============================================================================================
                        
                        # right = ['[unused8]'] + ['30', '%'] = ['[unused8]', '30', '%']
                        right = ([self.entityType2word[c_entities[elem[1][1]][0]]]  # entityType2word["RateV"]
                                 + sentence[c_entities[elem[1][1]][1][0]:c_entities[elem[1][1]][1][1] + 1]) \
                            if elem[1][1] < len(c_entities) else [self.relIdentityWord[(elem[1][1]-ent_len+1)]]

                        if self.config.rel_place == 0:  # 
                            cur_sentence = cur_sentence + [self.relIdentityWord[(layer_elem[i][j]-ent_len+1)]] + left \
                                           + [type_w] + right + ['[SEP]']
                        # 
                        elif self.config.rel_place == 1:  # 
                            cur_sentence = cur_sentence + left + [self.relIdentityWord[(layer_elem[i][j]-ent_len+1)]] \
                                           + [type_w] + right + ['[SEP]']
                        else:  #
                            cur_sentence = cur_sentence + left + [type_w] + right + \
                                           [self.relIdentityWord[(layer_elem[i][j]-ent_len+1)]] + ['[SEP]']
                # ============================================================================================
                tmp_sentence = copy.deepcopy(cur_sentence)
                cc = 0
                # ================================tmp_label================================
                
                tmp_label = []
                # ==========================================================
               
                for c in candidates:
                    if c[0] - len(e['entity']) + 1 >= self.config.rel_elem_num \
                            or c[1] - len(e['entity']) + 1 >= self.config.rel_elem_num:
                        continue
                    # 
                    left = ([self.entityType2word[c_entities[c[0]][0]]]
                            + sentence[c_entities[c[0]][1][0]:c_entities[c[0]][1][1] + 1]) \
                        if layer[c[0]] == 0 else [self.relIdentityWord[c[0]-ent_len+1]]
                    # ============================================================================
                    
                    right = ([self.entityType2word[c_entities[c[1]][0]]]
                             + sentence[c_entities[c[1]][1][0]:c_entities[c[1]][1][1] + 1]) \
                        if layer[c[1]] == 0 else [self.relIdentityWord[c[1]-ent_len+1]]
                    # ===========================================================
                    if self.config.template == 2:
                        cur_c = left + ['对'] + right + ['是', '[MASK]', '关', '系', '[SEP]']  
                    else:  
                        # =================================================
                       
                        cur_c = left + ['[MASK]'] + right + ['[SEP]']  
                    # 
                    if len(tmp_sentence) + len(cur_c) > self.config.max_pro_len and len(tmp_label) > 0:
                        cc += 1
                        # 
                        prompt_ex.append(
                            {'sid': '%s-l-%d-%d' % (e['sid'], i+1, cc), 'sentence': tmp_sentence, 'label': tmp_label})
                        # 
                        tmp_sentence = copy.deepcopy(cur_sentence)
                        tmp_label = []
                    # =============================================================
                    tmp_sentence = tmp_sentence + cur_c
                    # ===============================================================
                    tmp_label.append(c2label.get(c, 'no'))
                if tmp_label:
                    prompt_ex.append(
                        {'sid': '%s-l-%d-%d' % (e['sid'], i + 1, cc + 1), 'sentence': tmp_sentence, 'label': tmp_label})
        writer = open(prompt_file, 'w', encoding='utf-8', newline='')
        for e in prompt_ex:
            writer.write(json.dumps(e, ensure_ascii=False)+'\n')
        writer.close()

    def convert_test_prompt(self, sentence, entities, elements):
        ent_len, elem_len = len(entities), len(elements)
        c_entities = self.convert_entity(sentence[:-1], entities)
        layer = [0] * elem_len
        for i in range(ent_len, elem_len):
            layer[i] = 1 + max(layer[elements[i][1][0]], layer[elements[i][1][1]])
        layer_elem = [[] for _ in range(max(layer)+1)]
        for i in range(elem_len):
            layer_elem[layer[i]].append(i)
        cur_sentence = copy.deepcopy(sentence)
        for i, l_ in enumerate(layer):
            if l_ != 0:
                if i-len(entities)+1 >= self.config.rel_elem_num:
                    continue
                elem = elements[i]
                type_w = self.relTypeLabelWord[elem[0]][0]
                left = ([self.entityType2word[c_entities[elem[1][0]][0]]]
                        + sentence[c_entities[elem[1][0]][1][0]:c_entities[elem[1][0]][1][1]+1]) \
                    if elem[1][0] < len(c_entities) else [self.relIdentityWord[elem[1][0]-ent_len+1]]
                right = ([self.entityType2word[c_entities[elem[1][1]][0]]]
                         + sentence[c_entities[elem[1][1]][1][0]:c_entities[elem[1][1]][1][1]+1]) \
                    if elem[1][1] < len(c_entities) else [self.relIdentityWord[elem[1][1]-ent_len+1]]
                if self.config.rel_place == 0:  # 
                    cur_sentence = cur_sentence + [self.relIdentityWord[(i-ent_len+1)]] \
                                   + left + [type_w] + right + ['[SEP]']
                elif self.config.rel_place == 1:  #
                    cur_sentence = cur_sentence + left + [self.relIdentityWord[(i-ent_len+1)]] \
                                   + [type_w] + right + ['[SEP]']
                else:  # 
                    cur_sentence = cur_sentence + left + [type_w] + right \
                                   + [self.relIdentityWord[(i-ent_len+1)]] + ['[SEP]']
        prompt_ex = []
        candidates = self.generate_candidate_train(layer_elem, elements)
        tmp_sentence = copy.deepcopy(cur_sentence)
        tmp_c = []
        for c in candidates:
            if c[0] - len(entities) + 1 >= self.config.rel_elem_num \
                    or c[1] - len(entities) + 1 >= self.config.rel_elem_num:
                continue
            left = ([self.entityType2word[c_entities[c[0]][0]]]
                    + sentence[c_entities[c[0]][1][0]:c_entities[c[0]][1][1]+1]) \
                if c[0] < len(c_entities) else [self.relIdentityWord[c[0]-ent_len+1]]
            right = ([self.entityType2word[c_entities[c[1]][0]]]
                     + sentence[c_entities[c[1]][1][0]:c_entities[c[1]][1][1]+1]) \
                if c[1] < len(c_entities) else [self.relIdentityWord[c[1]-ent_len+1]]
            if self.config.template == 2:
                cur_c = left + ['对'] + right + ['是', '[MASK]', '关', '系', '[SEP]']
            else:
                cur_c = left + ['[MASK]'] + right + ['[SEP]']

            if len(tmp_sentence)+len(cur_c) > self.config.max_pro_len >= len(tmp_sentence) and len(tmp_c) > 0:
                prompt_ex.append(
                    {'sentence': tmp_sentence+[self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                     'candidates': tmp_c})
                tmp_sentence = copy.deepcopy(cur_sentence)
                tmp_c = []
            tmp_sentence = tmp_sentence + cur_c
            tmp_c.append(c)
        if tmp_c and len(tmp_sentence) <= self.config.max_pro_len:
            prompt_ex.append(
                {'sentence': tmp_sentence+[self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                 'candidates': tmp_c})
        return prompt_ex

    def get_train_loader(self, data_file):
        """
        """
        def collate_fn(batch):
            sentence, word_ids, label, sid = [], [], [], []
            for idx, line in enumerate(batch):
                item = json.loads(line)
                t_words = self.tokenizer.convert_tokens_to_ids(item['sentence'])
                word_ids.append(t_words+[self.tokenizer.pad_token_id]*(self.config.max_pro_len-len(t_words)))
                label.extend([self.rel_type.index(x) for x in item['label']])
                sentence.append(''.join(item['sentence']))
                sid.append(item['sid'])
            return {'sentence': sentence, 'ids': word_ids, 'label': label, 'sid': sid}

        _collate_fn = collate_fn
        print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=True, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def get_other_loader(self, data_file):
        def collate_fn(batch):
            """
           
            """
            sentence, sid, words, entities, elements = [], [], [], [], []
            for idx, line in enumerate(batch):
                item = json.loads(line)
                sentence.append(item['sentence'])
                t_words = ['[CLS]'] + self.tokenizer.tokenize(item['sentence']) + ['[SEP]']
                words.append(t_words)
                sid.append(item['sid'])
                entities.append([(e[0], (e[1][0], e[1][1])) for e in item['entity']])
                elements.append(entities[-1] + [(e[0], tuple(e[1])) for e in item['element'][len(entities[-1]):]])
            return {'sentence': sentence, 'words': words, 'entity': entities, 'element': elements, 'sid': sid}

        _collate_fn = collate_fn
        # print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=False, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def predict(self, op, data_file=None):
        """
       
        """
        data_iter = self.get_other_loader(  # data_file = data/5-1/valid.json
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity = [], [], []
        self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    all_elem.append(batch['element'][i])
                    all_entity.append(batch['entity'][i])
                    cur_elem = copy.deepcopy(batch['entity'][i])
                    layer = 0
                    while layer <= 6:
                        prompt_ex = self.convert_test_prompt(batch['words'][i], batch['entity'][i], cur_elem)
                        flag = True
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            score = self.model(ids, mask)
                            cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_score = score[cand_index.tolist()][:, self.all_rel_lb_id]
                            cand_prob = cand_score.float().softmax(-1)
                            label_prob = t.cat([cand_prob[:, self.relTypeLabelWordId[tp]].mean(-1).unsqueeze(-1)
                                                for tp in self.rel_type], dim=-1)
                            # print(label_prob.shape)
                            cand_label = [self.rel_type[x] for x in label_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_elem) < self.config.rel_elem_num:
                                    cur_elem.append((label, e['candidates'][li]))
                                    flag = False
                        if flag:
                            break
                        else:
                            layer += 1
                    all_predict.append(cur_elem[len(batch['entity'][i]):])
        return all_predict, all_elem, all_entity

    def predict_prob(self, op, data_file=None):
        data_iter = self.get_other_loader(
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity, all_prob = [], [], [], []
        if op == 'train':
            self.model.train()
        else:
            self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    all_elem.append(batch['element'][i])
                    all_entity.append(batch['entity'][i])
                    cur_elem = copy.deepcopy(batch['entity'][i])
                    cur_prob = []
                    layer = 0
                    while layer <= 6:
                        prompt_ex = self.convert_test_prompt(batch['words'][i], batch['entity'][i], cur_elem)
                        flag = True
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            score = self.model(ids, mask)
                            cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_score = score[cand_index.tolist()][:, self.all_rel_lb_id]
                            cand_prob = cand_score.float().softmax(-1)
                            label_prob = t.cat([cand_prob[:, self.relTypeLabelWordId[tp]].mean(-1).unsqueeze(-1)
                                                for tp in self.rel_type], dim=-1)
                            # print(label_prob.shape)
                            cand_label = [self.rel_type[x] for x in label_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_elem) < self.config.rel_elem_num:
                                    cur_elem.append((label, e['candidates'][li]))
                                    cur_prob.append(label_prob[li].tolist())
                                    flag = False
                        if flag:
                            break
                        else:
                            layer += 1
                    all_predict.append(cur_elem[len(batch['entity'][i]):])
                    all_prob.append(cur_prob)
        return all_predict, all_elem, all_entity, all_prob

    def predict_guided(self, op, data_file=None):
        data_iter = self.get_other_loader(
            os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, len(data_iter)))
        print('*' * 20)
        all_predict, all_elem, all_entity = [], [], []
        self.model.eval()
        with t.no_grad():
            for idx, batch in enumerate(data_iter):
                for i in range(len(batch['sid'])):
                    all_entity.append(batch['entity'][i])
                    layer = [0] * len(batch['element'][i])
                    for k in range(len(batch['entity'][i]), len(batch['element'][i])):
                        layer[k] = 1+max(layer[batch['element'][i][k][1][0]], layer[batch['element'][i][k][1][1]])
                    layer_elem = [[] for _ in range(max(layer)+1)]
                    for k in range(len(layer)):
                        layer_elem[layer[k]].append(batch['element'][i][k])
                    all_elem.append(layer_elem[1:])
                    cur_gold, cur_predict = [], []
                    for l_ in range(1, max(layer)+1):
                        cur_predict.append([])
                        cur_gold.extend(layer_elem[l_-1])
                        try:
                            prompt_ex = self.convert_test_prompt(
                                batch['words'][i], batch['entity'][i], copy.deepcopy(cur_gold))
                        except:
                            print(batch['sid'][i], l_, layer, cur_gold)
                            prompt_ex = []
                        for e in prompt_ex:
                            ids = t.tensor(self.tokenizer.convert_tokens_to_ids(e['sentence']),
                                           dtype=t.long, device=self.config.device).unsqueeze(0)
                            mask = ids.ne(self.tokenizer.pad_token_id)
                            score = self.model(ids, mask)
                            cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                            cand_score = score[cand_index.tolist()][:, self.all_rel_lb_id]
                            cand_prob = cand_score.float().softmax(-1)
                            label_prob = t.cat([cand_prob[:, self.relTypeLabelWordId[tp]].mean(-1).unsqueeze(-1)
                                                for tp in self.rel_type], dim=-1)
                            cand_label = [self.rel_type[x] for x in label_prob.argmax(-1)]
                            assert len(cand_label) == len(e['candidates'])
                            for li, label in enumerate(cand_label):
                                if label != 'no' and len(cur_gold) < self.config.rel_elem_num:
                                    cur_predict[-1].append((label, e['candidates'][li]))
                    all_predict.append(cur_predict)
        return all_predict, all_elem, all_entity

    def eval(self, op, data_file=None):
        all_predict, all_elem, all_entity = self.predict(op, data_file)
        if op == self.config.test_prefix:
            ex = [json.loads(l) for l in
                  open(os.path.join(self.data_dir, self.raw_tmp_file % op) if data_file is None else data_file)]
            all_examples = []
            assert len(ex) == len(all_elem)
            for e, entity, element in zip(ex, all_entity, all_elem):
                all_examples.append({'sid': e['sid'], 'entity': entity, 'element': element})
            save_predict(all_predict, all_examples, os.path.join(self.model_dir, 'predict.json'))
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, self.rel_type[:-1])
        print_dist(mp, mr, mf1, self.rel_type[:-1],
                   out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'metric.xlsx'))
        if self.config.guide:
            g_predict, g_elem, g_entity = self.predict_guided(op, data_file)
            gp, gr, gf1 = compute_metric_by_dist_layers(g_elem, g_predict, self.rel_type[:-1])
            print('#'*5, 'guide result', '#'*5)
            print_dist(gp, gr, gf1, self.rel_type[:-1],
                       out_file=None if op != self.config.test_prefix else os.path.join(self.model_dir, 'guide.xlsx'))
        return p, r, f1

    def train(self, data_file=None):
        self._setup_train(self.config.bert_lr)
        # self.convert_train_data2prompt(self.data_dir, self.raw_tmp_file % self.config.train_prefix)
        # train_iter = self.get_train_loader(
        #     os.path.join(self.model_dir, 'prompt-%s' % (self.raw_tmp_file % self.config.train_prefix)))
        self.convert_train_data2prompt(
            self.data_dir, (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)
        train_iter = self.get_train_loader(
            os.path.join(
                self.model_dir,
                'prompt-%s' % ((self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)))
        batch_len = len(train_iter)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            if self.config.rel_mode == 1:
                # ==================================================================
                print('shuffle relation id words and recreate train file')
                # 
                random.shuffle(self.relIdentityWord)
            if self.config.rel_mode in [1, 2]:
                # 
                print('convert new train file')
                self.convert_train_data2prompt(
                    self.data_dir, (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)
                train_iter = self.get_train_loader(
                    os.path.join(
                        self.model_dir,
                        'prompt-%s' % (
                            (self.raw_tmp_file % self.config.train_prefix) if data_file is None else data_file)))
                # 
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['ids'], dtype=t.long, device=self.config.device)
                label = t.tensor(batch['label'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                score = self.model(ids, mask)
                cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                cand_score = score[cand_index.tolist()][:, self.all_rel_lb_id]
                cand_prob = cand_score.float().softmax(-1)
                label_prob = t.cat([cand_prob[:, self.relTypeLabelWordId[tp]].mean(-1).unsqueeze(-1)
                                    for tp in self.rel_type], dim=-1)
                loss = F.nll_loss(t.log(label_prob), label)
                loss.backward()

                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            # =======================================================================
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)


class PromptAutoFKH(PromptAuto):
    """
   
    """
    def __init__(self, config):
        super(PromptAutoFKH, self).__init__(config)
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.rel_type = sorted(list(n_label))
        self.all_type = self.all_type[:-len(self.relations)] + self.rel_type

        self.model_dir = os.path.join(config.model_dir, 'prompt-auto-fkh')
        os.makedirs(self.model_dir, exist_ok=True)
        save_config(config, os.path.join(self.model_dir, 'config.txt'))

        self.model = PromptModel(config.bert_dir).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.type2word = {tp: '[unused%d]' % (i + 1) for i, tp in enumerate(self.all_type + ['no'])}
        self.rel_type_word = [
            '[unused%d]' % (len(self.all_type) - len(self.rel_type) + i + 1) for i in range(len(self.rel_type) + 1)]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)
        self.rel_id_word = ['[unused%d]' % (i + self.config.rel_start) for i in range(1, self.config.rel_elem_num + 1)]
        self.rel_type = self.rel_type + ['no']

        self.raw_tmp_file = '%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, '%s')


class PromptManualFKH(PromptManual):
    """
    
    """
    def __init__(self, config):
        super(PromptManualFKH, self).__init__(config)
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.rel_type = sorted(list(n_label)) + ['no']
        self.entity_type = self.all_type[:-len(self.relations)]

        self.model_dir = os.path.join(config.model_dir, 'prompt-manual-fkh')
        os.makedirs(self.model_dir, exist_ok=True)
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

        self.model = PromptModel(config.bert_dir).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.entityType2word = {tw: '[unused%d]' % (i+1) for i, tw in enumerate(self.entity_type)}
        relTypeLabelWord = {'>': ['>'], '>=': ['≥', '≧'],
                            '<': ['<'], '<=': ['≤', '≦'], '=': ['='],
                            '+': ['加'], '-': ['减'], '/': ['除', '率'],
                            '@': ['属', '之', '的'], 'has': ['有', '含'],
                            'and': ['与', '和', '并'], 'no': ['无', '非', '没', '不']}
        self.relTypeLabelWord = {tp: relTypeLabelWord[tp] for tp in self.rel_type}
        for tp in self.rel_type:
            self.relTypeLabelWord[tp] = self.relTypeLabelWord[tp][:1]

        #
        self.all_rel_lb_id = self.tokenizer.convert_tokens_to_ids(self.all_rel_lb)
        # 
        self.relTypeLabelWordId = {k: [self.all_rel_lb.index(w) for w in v] for k, v in self.relTypeLabelWord.items()}
        self.relIdentityWord = ['[unused%d]' % (i + self.config.rel_start) for i in
                                range(1, self.config.rel_elem_num + 1)]

        self.raw_tmp_file = '%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, '%s')


class PromptAutoFKL(PromptAuto):
    """
    
    """
    def __init__(self, config):
        super(PromptAutoFKL, self).__init__(config)
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i + config.few_rel_idx][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.target_relations = sorted(list(n_label))

        self.model_dir = os.path.join(config.model_dir, 'prompt-auto-fkl')
        os.makedirs(self.model_dir, exist_ok=True)
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.model = PromptModel(config.bert_dir).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.type2word = {tp: '[unused%d]' % (i + 1) for i, tp in enumerate(self.all_type + ['no'])}
        self.rel_type_word = [
            '[unused%d]' % (len(self.all_type) - len(self.relations) + i + 1) for i in range(len(self.rel_type) + 1)]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)
        self.rel_id_word = ['[unused%d]' % (i + self.config.rel_start) for i in range(1, self.config.rel_elem_num + 1)]
        self.rel_type = self.rel_type + ['no']

        self.raw_tmp_file = 'low-%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, '%s')

    def eval(self, op, data_file=None):
        all_predict, all_elem, all_entity = self.predict(op, data_file)
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} all result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, self.rel_type[:-1])
        print_dist(mp, mr, mf1, self.rel_type[:-1])

        g_predict, g_elem, g_entity = self.predict_guided(op, data_file)
        p, r, f1 = compute_metric_relation_guided(g_elem, g_predict, g_entity, self.target_relations)
        print('[INFO] {} | {} target result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class PromptManualFKL(PromptManual):
    """
    """
    def __init__(self, config):
        super(PromptManualFKL, self).__init__(config)
        self.config = config
        self.data_dir = config.data_dir
        self.all_type, self.relations = get_type_info(config.type_file)
        raw_train_file = os.path.join(config.data_dir, 'train.json')
        examples = [json.loads(line) for line in open(raw_train_file, 'r', encoding='utf-8').readlines()]
        label2num = {r[0]: 0 for r in self.relations}
        for e in examples:
            for r in e['element'][len(e['entity']):]:
                label2num[r[0]] += 1
        label2num = sorted(label2num.items(), key=lambda x: x[1], reverse=True)
        label_count = {label2num[i + config.few_rel_idx][0]: 0 for i in range(config.few_n)}
        n_label = set(label_count.keys())
        self.target_relations = sorted(list(n_label))
        self.rel_type = [x[0] for x in self.relations] + ['no']
        self.entity_type = self.entity_type = list(filter(lambda x: x not in self.rel_type, self.all_type))

        self.model_dir = os.path.join(config.model_dir, 'prompt-manual-fkl')
        os.makedirs(self.model_dir, exist_ok=True)
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))

        self.model = PromptModel(config.bert_dir).to(config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_dir)
        self.entityType2word = {tw: '[unused%d]' % (i + 1) for i, tw in enumerate(self.entity_type)}
        relTypeLabelWord = {'>': ['>'], '>=': ['≥', '≧'],
                            '<': ['<'], '<=': ['≤', '≦'], '=': ['='],
                            '+': ['加'], '-': ['减'], '/': ['除', '率'],
                            '@': ['属', '之', '的'], 'has': ['有', '含'],
                            'and': ['与', '和', '并'], 'no': ['无', '非', '没', '不']}
        self.relTypeLabelWord = {tp: relTypeLabelWord[tp] for tp in self.rel_type}
        for tp in self.rel_type:
            self.relTypeLabelWord[tp] = self.relTypeLabelWord[tp][:1]
        # 
        self.all_rel_lb_id = self.tokenizer.convert_tokens_to_ids(self.all_rel_lb)
        # 
        self.relTypeLabelWordId = {k: [self.all_rel_lb.index(w) for w in v] for k, v in self.relTypeLabelWord.items()}
        self.relIdentityWord = ['[unused%d]' % (i + self.config.rel_start) for i in
                                range(1, self.config.rel_elem_num + 1)]

        self.raw_tmp_file = 'low-%d-way-%d-shot-%s.json' % (self.config.few_n, self.config.few_k, '%s')

    def eval(self, op, data_file=None):
        all_predict, all_elem, all_entity = self.predict(op, data_file)
        p, r, f1 = compute_metric(all_elem, all_predict, all_entity)
        print('[INFO] {} | {} all result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        mp, mr, mf1 = compute_metric_by_dist(all_elem, all_predict, all_entity, self.rel_type[:-1])
        print_dist(mp, mr, mf1, self.rel_type[:-1])

        g_predict, g_elem, g_entity = self.predict_guided(op, data_file)
        p, r, f1 = compute_metric_relation_guided(g_elem, g_predict, g_entity, self.target_relations)
        print('[INFO] {} | {} target result: p = {}, r = {}, f1 = {}'.format(
            datetime.datetime.now(), op, p, r, f1))
        return p, r, f1


class FlatPromptAuto(object):
    def __init__(self, config):
        self.config = config
        # ==========================================data/flat-relation/kbp37============================================
        # ==========================================data/flat-relation/semeval2010======================================
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        # ===============================================bert-base-uncased==============================================
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        # ======================================model_dir===================================================
        # =========================================model/prompt-auto-kbp37==============================================
        # ======================================model/prompt-auto-semeval2010===========================================
        self.model_dir = os.path.join(config.model_dir, 'prompt-auto-%s' % config.flat_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        # =================================================semeval2010=============================================
        # self.relation = [
        # "Cause-Effect",
        # "Component-Whole",
        # "Content-Container",
        # "Entity-Destination",
        # "Entity-Origin",
        # "Instrument-Agency",
        # "Member-Collection",
        # "Message-Topic",
        # "Other",
        # "Product-Producer"
        # ]
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        # =========================================================================================
        # self.rel2id = {
        # "Cause-Effect(e1,e2)": 0,
        # "Cause-Effect(e2,e1)": 1,
        # ...
        # }
        self.rel2id = {}
        # =======================================================================================
        # self.id2rel = [
        # "Cause-Effect(e1,e2)",
        # "Cause-Effect(e2,e1)",
        # ...
        # ]
        self.id2rel = []
        self.neg_rel = ['no_relation', 'Other']
        for r in self.relation:
            if r not in self.neg_rel:
                self.rel2id[r + '(e1,e2)'] = len(self.rel2id)
                self.id2rel.append(r+'(e1,e2)')
                self.rel2id[r + '(e2,e1)'] = len(self.rel2id)
                self.id2rel.append(r + '(e2,e1)')
            else:
                self.rel2id[r] = len(self.rel2id)
                self.id2rel.append(r)
        print('%d relation types' % len(self.rel2id))
        print(self.id2rel)
        # 
        self.model = PromptModel(config.bert_en).to(config.device)
        # self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        # =============================================================================================
        # self.rel_type_word = [
        # [unused0],
        # [unused1],
        # ...
        # ]
        self.rel_type_word = ['[unused%d]' % i for i in range(len(self.rel2id))]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)

    def _setup_train(self, lr=1e-5):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            return
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def get_loader(self, op='train'):
        from data_utils import NDataSet
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            sentence, words, label, p_label = [], [], [], []
            for idx, line in enumerate(batch):
                
                e = eval(line)
                # 
                r_entity = [(x['name'], x['pos']) for x in [e['h'], e['t']]]
                
                t_word = ['[CLS]'] + self.tokenizer.tokenize(' '.join(e['token'])) + [
                    '[SEP]'] + self.tokenizer.tokenize(
                    ' '.join(e['token'][r_entity[0][1][0]:r_entity[0][1][1]])) + ['[MASK]'] + self.tokenizer.tokenize(
                    ' '.join(e['token'][r_entity[1][1][0]:r_entity[1][1][1]])) + ['[SEP]']
                # 
               
                t_label = ['[CLS]'] + self.tokenizer.tokenize(' '.join(e['token'])) + [
                    '[SEP]'] + self.tokenizer.tokenize(
                    ' '.join(e['token'][r_entity[0][1][0]:r_entity[0][1][1]])) + [
                    self.rel_type_word[self.rel2id[e['relation']]]] + self.tokenizer.tokenize(
                    ' '.join(e['token'][r_entity[1][1][0]:r_entity[1][1][1]])) + ['[SEP]']
                # 
                t_word_ids = self.tokenizer.convert_tokens_to_ids(t_word)
                t_label_ids = self.tokenizer.convert_tokens_to_ids(t_label)
                # assert len(t_label_ids) == len(t_word_ids) <= self.config.max_flat_len
                if not len(t_label_ids) == len(t_word_ids) <= self.config.max_flat_len:
                    print(len(t_label_ids), len(t_word_ids))
                # 
                t_word_ids = t_word_ids[:self.config.max_flat_len] + [
                    self.tokenizer.pad_token_id] * (self.config.max_flat_len-len(t_word_ids))
                t_label_ids = t_label_ids[:self.config.max_flat_len] + [
                    self.tokenizer.pad_token_id] * (self.config.max_flat_len-len(t_label_ids))
                sentence.append(e['token'])  # 
                words.append(t_word_ids)
                p_label.append(t_label_ids)
                label.append(e['relation'])  # 
            return {'sentence': sentence, 'words': words, 'p-label': p_label, 'label': label}

        _collate_fn = collate_fn
        file = os.path.join(self.data_dir, '%s.txt' % op)
        data_iter = DataLoader(dataset=NDataSet(file), batch_size=self.config.batch_size,
                               shuffle=(op == 'train'), collate_fn=_collate_fn)
        return data_iter

    def compute_metric(self, label, predict, op='valid'):
        if op == 'test':
            res_file = os.path.join(self.model_dir, 'result.csv')
            writer = open(res_file, 'w', encoding='utf-8')
            for i in range(len(label)):
                writer.write(','.join([label[i], predict[i]]) + '\n')
            writer.close()
        rel2id = self.rel2id
        TP, FP, FN = [0] * len(self.rel2id), [0] * len(self.rel2id), [0] * len(self.rel2id)
        for i in range(len(label)):
            if label[i] not in self.neg_rel:
                TP[rel2id[label[i]]] += int(label[i] == predict[i])
                FP[rel2id[label[i]]] += int(label[i] != predict[i])
            if predict[i] not in self.neg_rel:
                FN[rel2id[predict[i]]] += int(label[i] != predict[i])
        tp, fp, fn = sum(TP), sum(FP), sum(FN)
        p = tp / (tp + fp) if tp != 0 else 0
        r = tp / (tp + fn) if tp != 0 else 0
        f1 = 2 * p * r / (p + r) if tp != 0 else 0
        return p, r, f1

    def eval(self, op):
        valid_iter = self.get_loader(op=op)
        batch_len = len(valid_iter)
        print('*' * 50 + 'Validation' + '*' * 50)
        print('[INFO] {} | {} batch len = {}'.format(datetime.datetime.now(), op, batch_len))
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                prob = self.model(ids, mask)
                cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                cand_prob = prob[cand_index.tolist()][:, self.rel_type_ids[0]:self.rel_type_ids[-1]+1]
                cand_prob = cand_prob.float().softmax(-1)
                cand_label = [self.id2rel[x] for x in cand_prob.argmax(-1)]
                predict.extend(cand_label)
                label.extend(batch['label'])
        print('*' * 50 + 'Before compute metric' + '*' * 50)
        print(f'label len: {len(label)}')
        print(f'predict len: {len(predict)}')
        print(label[:10])
        print(predict[:10])
        p, r, f1 = self.compute_metric(label, predict, op=op)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1

    def train(self):
        self._setup_train(self.config.bert_lr)
        train_iter = self.get_loader()
        batch_len = len(train_iter)
        print('*' * 50 + 'Training' + '*' * 50)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                # ============================ids.shape=(batch_size, seq_len),ids.dtype=long================
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                # ===========================label.shape=(batch_size, seq_len),label.dtype=long===========
                label = t.tensor(batch['p-label'], dtype=t.long, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                # prob.shape = (batch_size, seq_len, config.vocab_size)
                prob = self.model(ids, mask)
                # 
                loss = F.cross_entropy(prob.reshape([-1, prob.shape[-1]]), label.reshape([-1]), reduce=False)
                # 
                loss = loss * ids.eq(self.tokenizer.mask_token_id).reshape([-1])
                # 
                loss = loss.sum() / ids.eq(self.tokenizer.mask_token_id).sum()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)


class FlatPromptManual(object):
    """
    """
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.flat_dir, config.flat_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'prompt-manual-%s' % config.flat_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        self.neg_rel = ['no_relation', 'Other']
        for r in self.relation:
            if r not in self.neg_rel:
                self.rel2id[r + '(e1,e2)'] = len(self.rel2id)
                self.id2rel.append(r + '(e1,e2)')
                self.rel2id[r + '(e2,e1)'] = len(self.rel2id)
                self.id2rel.append(r + '(e2,e1)')
            else:
                self.rel2id[r] = len(self.rel2id)
                self.id2rel.append(r)
        print('%d relation types' % len(self.rel2id))
        print(self.id2rel)
        self.model = PromptModel(config.bert_en).to(config.device)
        self.rel_type_word = ['[unused%d]' % i for i in range(len(self.rel2id))]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            return
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))


class OverlapPromptAuto(object):
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config.overlap_dir, config.overlap_name)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.model_dir = os.path.join(config.model_dir, 'prompt-auto-%s' % config.overlap_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.config.model_dir = self.model_dir
        save_config(self.config, os.path.join(self.model_dir, 'config.txt'))
        self.relation = json.loads(
            open(os.path.join(self.data_dir, 'rel.txt'), 'r', encoding='utf-8').readlines()[0])['relation']
        self.rel2id = {}
        self.id2rel = []
        for r in self.relation:
            self.rel2id[r] = len(self.rel2id)
            self.id2rel.append(r)
        print('%d relation types' % len(self.rel2id))
        print(self.id2rel)
        self.model = PromptModel(config.bert_en).to(config.device)
        # self.tokenizer = BertTokenizer.from_pretrained(config.bert_en)
        self.rel_type_word = ['[unused%d]' % i for i in range(len(self.rel2id))]
        self.rel_type_ids = self.tokenizer.convert_tokens_to_ids(self.rel_type_word)

    def _setup_train(self, lr=1e-5):
        params = self.model.parameters()
        self.optimizer = t.optim.Adam([
            {'params': params, 'lr': lr}
        ])

    def save_model(self, epoch, f1):
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'val f1': f1
        }
        print('save a model at epoch %d' % epoch)
        t.save(state, os.path.join(self.model_dir, 'model.pt'))

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            return
        state = t.load(model_file, map_location=lambda storage, location: storage)
        self.model.load_state_dict(state['model'])
        print('load model from epoch = %d' % (state['epoch']))

    def convert_train_data2prompt(self, data_dir, file_name):
        prompt_file = os.path.join(self.model_dir, 'prompt-%s' % file_name)
        ex = [eval(line) for line in open(os.path.join(data_dir, file_name), 'r', encoding='utf-8').readlines()]
        prompt_ex = []
        for idx, e in enumerate(ex):
            sentence = ['[CLS]'] + self.tokenizer.tokenize(' '.join(e['tokens'])) + ['[SEP]']
            relations = e['relation']
            entity = []
            for r in relations:
                entity.extend([tuple(r['e1'].values()), tuple(r['e2'].values())])
            entity = sorted(list(set(entity)))
            ep2label = {}
            for r in relations:
                entity_pair = (entity.index(tuple(r['e1'].values())), entity.index(tuple(r['e2'].values())))
                ep2label[entity_pair] = ep2label.get(entity_pair, []) + [r['label']]
            candidate, label = [], []
            for left in range(len(entity)):
                for right in range(len(entity)):
                    candidate.append((left, right))
                    c_label = [0] * len(self.id2rel)
                    for l_ in ep2label.get((left, right), []):
                        c_label[self.rel2id[l_]] = 1
                    label.append(c_label)
            tmp_sentence = copy.deepcopy(sentence)
            tmp_label = []
            cc = 0
            for c in candidate:
                left, right = self.tokenizer.tokenize(entity[c[0]][0]), self.tokenizer.tokenize(entity[c[1]][0])
                cur_c = left + ['[MASK]'] + right + ['[SEP]']
                if len(tmp_sentence) + len(cur_c) > self.config.max_pro_len:
                    prompt_ex.append({'id': '%d-%d' % (idx, cc), 'sentence': tmp_sentence, 'label': tmp_label})
                    tmp_sentence = copy.deepcopy(sentence)
                    tmp_label = []
                tmp_sentence = tmp_sentence + cur_c
                t_label = [0] * len(self.id2rel)
                for l_ in ep2label.get(c, []):
                    t_label[self.rel2id[l_]] = 1
                tmp_label.append(t_label)
            if tmp_label:
                prompt_ex.append({'id': '%d-%d' % (idx, cc), 'sentence': tmp_sentence, 'label': tmp_label})
        writer = open(prompt_file, 'w', encoding='utf-8')
        for e in prompt_ex:
            writer.write(json.dumps(e, ensure_ascii=False) + '\n')
        writer.close()

    def convert_test_data2prompt(self, sentence, entities):
        candidate = []
        for left in range(len(entities)):
            for right in range(len(entities)):
                candidate.append((left, right))
        prompt_ex = []
        tmp_sentence = copy.deepcopy(sentence)
        tmp_c = []
        for c in candidate:
            left, right = self.tokenizer.tokenize(entities[c[0]][0]), self.tokenizer.tokenize(entities[c[1]][0])
            cur_c = left + ['[MASK]'] + right + ['[SEP]']
            if len(tmp_sentence) + len(cur_c) > self.config.max_pro_len >= len(tmp_sentence):
                prompt_ex.append(
                    {'sentence': tmp_sentence + [self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                     'candidates': tmp_c})
                tmp_sentence = copy.deepcopy(sentence)
                tmp_c = []
            tmp_sentence = tmp_sentence + cur_c
            tmp_c.append(c)
        if tmp_c and len(tmp_sentence) <= self.config.max_pro_len:
            prompt_ex.append(
                {'sentence': tmp_sentence + [self.tokenizer.pad_token]*(self.config.max_pro_len-len(tmp_sentence)),
                 'candidates': tmp_c})
        return prompt_ex

    def get_train_loader(self, data_file):
        def collate_fn(batch):
            sentence, word_ids, label, ids = [], [], [], []
            for idx, line in enumerate(batch):
                item = json.loads(line)
                t_words = self.tokenizer.convert_tokens_to_ids(item['sentence'])
                t_words = t_words + [self.tokenizer.pad_token_id] * (self.config.max_pro_len-len(t_words))
                ids.append(item['id'])
                word_ids.append(t_words)
                sentence.append(''.join(item['sentence']))
                label.extend(item['label'])
            return {'sentence': sentence, 'ids': ids, 'words': word_ids, 'label': label}
        _collate_fn = collate_fn
        print('shuffle data set...')
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=True, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def get_other_loader(self, data_file):
        def collate_fn(batch):
            return {'ex': [eval(x) for x in batch]}
        _collate_fn = collate_fn
        data_iter = DataLoader(dataset=NDataSet(data_file), batch_size=self.config.batch_size,
                               shuffle=False, collate_fn=_collate_fn, num_workers=0)
        return data_iter

    def compute_metric(self, label, predict):
        assert len(label) == len(predict)
        cnt, c1, c2 = 0, 0, 0
        for i in range(len(label)):
            # cur_predict = [(self.id2rel[r[0]], r[1]) for r in predict[i]]
            cnt += len(set(label[i]) & set(predict[i]))
            c1 += len(set(label[i]))
            c2 += len(set(predict[i]))
        p = cnt / c2 if cnt != 0 else 0
        r = cnt / c1 if cnt != 0 else 0
        f1 = 2*p*r / (p+r) if cnt != 0 else 0
        return p, r, f1

    def eval(self, op='valid'):
        valid_iter = self.get_other_loader(os.path.join(self.data_dir, '%s.txt' % op))
        self.model.eval()
        predict, label = [], []
        with t.no_grad():
            for idx, batch in enumerate(valid_iter):
                ex = batch['ex']
                for e in ex:
                    sentence = ['[CLS]'] + self.tokenizer.tokenize(' '.join(e['tokens'])) + ['[SEP]']
                    relations = e['relation']
                    entity, triple = [], []
                    for r in relations:
                        entity.extend([tuple(r['e1'].values()), tuple(r['e2'].values())])
                        triple.append((r['label'], (r['e1']['entity'], r['e2']['entity'])))
                    entity = sorted(list(set(entity)))
                    t_predict = []
                    prompt_ex = self.convert_test_data2prompt(sentence, entity)
                    for te in prompt_ex:
                        ids = t.tensor(self.tokenizer.convert_tokens_to_ids(te['sentence']),
                                       dtype=t.long, device=self.config.device).unsqueeze(0)
                        mask = ids.ne(self.tokenizer.pad_token_id)
                        score = self.model(ids, mask)
                        cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                        cand_score = score[cand_index.tolist()][:, self.rel_type_ids]
                        cand_prob = t.sigmoid(cand_score)
                        cand_label = cand_prob.ge(0.5).long().tolist()
                        cand_label = [[c[0] for c in filter(lambda x: x[1] > 0.5, list(zip(self.id2rel, cl)))]
                                      for cl in cand_label]
                        # print(len(cand_label), len(te['candidates']))
                        assert len(cand_label) == len(te['candidates'])
                        for ci in range(len(te['candidates'])):
                            for cl in cand_label[ci]:
                                t_predict.append(
                                    (cl, (entity[te['candidates'][ci][0]][0], entity[te['candidates'][ci][1]][0])))
                    predict.append(t_predict)
                    label.append(triple)
        p, r, f1 = self.compute_metric(label, predict)
        print('[INFO] {} | {} result: p = {}, r = {}, f1 = {}'.format(datetime.datetime.now(), op, p, r, f1))
        return p, r, f1

    def train(self):
        self._setup_train(self.config.bert_lr)
        self.convert_train_data2prompt(self.data_dir, '%s.txt' % self.config.train_prefix)
        train_iter = self.get_train_loader(os.path.join(self.model_dir, 'prompt-%s.txt' % self.config.train_prefix))
        batch_len = len(train_iter)
        print('[INFO] {} | train batch len = {}'.format(datetime.datetime.now(), batch_len))
        print('*' * 20)
        running_avg_loss = None
        max_f1, stop_step = -1, 0
        for e in range(1, self.config.train_epoch + 1):
            for idx, batch in enumerate(train_iter):
                self.model.train()
                self.optimizer.zero_grad()
                ids = t.tensor(batch['words'], dtype=t.long, device=self.config.device)
                label = t.tensor(batch['label'], dtype=t.float, device=self.config.device)
                mask = ids.ne(self.tokenizer.pad_token_id)
                score = self.model(ids, mask)
                cand_index = ids.eq(self.tokenizer.mask_token_id).nonzero().t()
                cand_score = score[cand_index.tolist()][:, self.rel_type_ids]
                cand_prob = F.sigmoid(cand_score)
                loss = F.binary_cross_entropy(cand_prob, label)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_avg_loss = calc_running_avg_loss(running_avg_loss, loss.item())
                if (idx + 1) % self.config.p_step == 0:
                    print('[INFO] {} | Epoch : {}/{} | process:{}/{} | train_loss : {}'.format(
                        datetime.datetime.now(), e, self.config.train_epoch, idx + 1,
                        batch_len, round(running_avg_loss, 5)
                    ))
            p, r, f1 = self.eval(op=self.config.valid_prefix)
            if f1 > max_f1:
                self.save_model(e, f1)
                max_f1 = f1
                stop_step = 0
            else:
                stop_step += 1
            if stop_step >= self.config.early_stop:
                print('[INFO] {} | early stop at epoch {}'.format(datetime.datetime.now(), e))
                break
        self._load_model(os.path.join(self.model_dir, 'model.pt'))
        self.eval(op=self.config.test_prefix)


def my_job():
    from config import Config
    nest_model = [('Auto', PromptAuto), ('Manual', PromptManual)]
    # 
    bs = [4, 8, 16]
    lr = [1e-5, 2e-5]
    rel_place = [0, 1, 2]
    rel_num = [60]
    do_whole = True
    for bs_idx in range(len(bs)):
        for lr_idx in range(len(lr)):
            for model_idx in range(len(nest_model)):
                for num_idx in range(len(rel_num)):
                    for p_idx in range(len(rel_place)):
                        for k in range(5):
                            if not do_whole:
                                continue
                            if model_idx != 1:
                                continue
                            config = Config()

                            config.random_seed = 111
                            config.template = 2
                            config.rel_mode = 0

                            config.train_epoch = 40
                            config.batch_size = bs[bs_idx]
                            config.bert_lr = lr[lr_idx]
                            config.rel_place = rel_place[p_idx]
                            config.rel_elem_num = rel_num[num_idx]
                            config.cont_weight = 0
                            config.model_dir = 'Prompt'
                            # config.model_dir = os.path.join(config.model_dir, 'bs-%d' % bs[bs_idx], 'rel-place-%d' % rel_place[p_idx])
                            config.device = 'cuda:1'
                            # =====================================data/5-1=============================================
                            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k+1))
                            # ======================================model/5-1===========================================
                            # config.model_dir = 'Prompt'
                            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k+1))
                            set_random(config.random_seed)
                            print('train prompt %s with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d, rel_mode=%d' % (
                                nest_model[model_idx][0], config.bert_lr, config.batch_size,
                                config.rel_place, config.rel_elem_num, k+1, config.rel_mode))
                            print_config(config)
                            model = nest_model[model_idx][1](config)
                            model.train()

    # 
    bs = [4, 8, 16]
    lr = [1e-5]
    rel_place = [0]
    rel_num = [60]
    ratio = [10, 20, 30, 50]
    random_seeds = [66, 1234, 42, 3678, 99999]
    do_fr = False
    for bs_idx in range(len(bs)):
        for lr_idx in range(len(lr)):
            for model_idx in range(len(nest_model)):
                for num_idx in range(len(rel_num)):
                    for p_idx in range(len(rel_place)):
                        for ratio_idx in range(len(ratio)):
                            for random_seed in random_seeds:
                                if not do_fr:
                                    continue
                                # if model_idx != 1:
                                #     continue
                                config = Config()

                                config.random_seed = random_seed
                                config.template = 2

                                config.train_epoch = 40
                                config.batch_size = bs[bs_idx]
                                config.bert_lr = lr[lr_idx]
                                config.rel_place = rel_place[p_idx]
                                config.rel_elem_num = rel_num[num_idx]
                                config.few_ratio = ratio[ratio_idx]
                                config.cont_weight = 0
                                config.device = 'cuda:0'
                                config.model_dir = 'Prompt'
                                config.model_dir = os.path.join(config.model_dir, 'FS/%d' % config.few_ratio)
                                set_random(config.random_seed)
                                print('train few %d prompt %s with lr = %f, bs = %d, random = %d, rel_num = %d' % (
                                    config.few_ratio, nest_model[model_idx][0], config.bert_lr, config.batch_size,
                                    config.random_seed, config.rel_elem_num))
                                print_config(config)
                                model = nest_model[model_idx][1](config)
                                model.train(data_file='%s-%d.json' % (config.few_labeled_prefix, ratio[ratio_idx]))

    # 小
    bs = [2, 4, 8, 16]
    lr = [1e-5, 5e-6, 2e-5, 3e-5, 4e-5, 5e-5]
    rel_num = [60]
    rel_place = [0, 1, 2]
    few_k = [1, 5, 10]
    high_model = [('Auto', PromptAutoFKH), ('Manual', PromptManualFKH)]
    do_fh = [True, False][1]
    for bs_idx in range(len(bs)):
        for lr_idx in range(len(lr)):
            for num_idx in range(len(rel_num)):
                for p_idx in range(len(rel_place)):
                    for k_idx in range(len(few_k)):
                        for model_idx in range(len(high_model)):
                            for random_seed in random_seeds:
                                if not do_fh:
                                    continue
                                # if model_idx != 1:
                                #     continue
                                config = Config()
                                config.template = 2

                                config.random_seed = random_seed

                                config.train_epoch = 40
                                config.batch_size = bs[bs_idx]
                                config.bert_lr = lr[lr_idx]
                                config.rel_elem_num = rel_num[num_idx]
                                config.rel_place = rel_place[p_idx]
                                config.few_k = few_k[k_idx]
                                config.cont_weight = 0
                                config.device = 'cuda:1'
                                config.model_dir = 'Prompt'
                                config.model_dir = os.path.join(
                                    config.model_dir,
                                    'FS/high-%d-%d' % (config.few_n, config.few_k))
                                set_random(config.random_seed)
                                print(
                                    'train high %d way %d shot prompt %s with lr = %f, bs = %d, random = %d, rel_num = %d' % (
                                        config.few_n, config.few_k, high_model[model_idx][0], config.bert_lr,
                                        config.batch_size, config.random_seed, config.rel_elem_num))
                                print_config(config)
                                model = high_model[model_idx][1](config)
                                model.train()

    # 
    bs = [2, 4, 8, 16]
    lr = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    rel_num = [60]
    rel_place = [0, 1, 2]
    few_k = [1, 5, 10]
    low_model = [('Auto', PromptAutoFKL), ('Manual', PromptManualFKL)]
    do_fl = [True, False][1]
    for bs_idx in range(len(bs)):
        for lr_idx in range(len(lr)):
            for num_idx in range(len(rel_num)):
                for p_idx in range(len(rel_place)):
                    for k_idx in range(len(few_k)):
                        for model_idx in range(len(low_model)):
                            for random_seed in random_seeds:
                                if not do_fl:
                                    continue
                                # if model_idx != 1:
                                #     continue
                                config = Config()

                                config.random_seed = random_seed

                                config.train_epoch = 40
                                config.batch_size = bs[bs_idx]
                                config.bert_lr = lr[lr_idx]
                                config.rel_elem_num = rel_num[num_idx]
                                config.rel_place = rel_place[p_idx]
                                config.few_k = few_k[k_idx]
                                config.cont_weight = 0
                                config.device = 'cuda:0'
                                config.model_dir = 'Prompt'
                                config.model_dir = os.path.join(
                                    config.model_dir,
                                    'FS/low-%d-%d' % (config.few_n, config.few_k))
                                set_random(config.random_seed)
                                print(
                                    'train low %d way %d shot prompt %s with lr = %f, bs = %d, random = %d, rel_num = %d' % (
                                        config.few_n, config.few_k, low_model[model_idx][0], config.bert_lr,
                                        config.batch_size, config.random_seed, config.rel_elem_num))
                                print_config(config)
                                model = low_model[model_idx][1](config)
                                model.train()

    # 
    bs = [8, 16]  # 
    lr = [2e-5, 1e-5, 5e-6]  # 
    flat_name = ['kbp37', 'semeval2010']
    flat_model = [('Auto', FlatPromptAuto)]
    do_flat = [True, False][1]
    for flat_idx in range(len(flat_name)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for model_idx in range(len(flat_model)):
                    for random_seed in random_seeds:
                        if not do_flat:
                            continue
                        # if model_idx != 1:
                        #     continue
                        config = Config()

                        config.random_seed = random_seed

                        config.train_epoch = 40  #
                        config.batch_size = bs[bs_idx]
                        config.bert_lr = lr[lr_idx]
                        config.device = 'cuda:1'
                        config.model_dir = 'Prompt'
                        config.model_dir = os.path.join(config.model_dir, 'flat')
                        config.flat_name = flat_name[flat_idx]
                        set_random(config.random_seed)
                        print('train prompt %s, flat = %s, bs = %d, lr = %f, random_seed = %d' % (
                            flat_model[model_idx][0], flat_name[flat_idx], config.batch_size, config.bert_lr, config.random_seed))
                        print_config(config)
                        model = flat_model[model_idx][1](config)
                        model.train()

    # overlap 
    bs = [8, 16]
    lr = [2e-5, 1e-5, 5e-6]
    overlap_name = ['NYT', 'WebNLG']
    overlap_model = [('Auto', OverlapPromptAuto)]
    do_overlap = False
    for overlap_idx in range(len(overlap_name)):
        for bs_idx in range(len(bs)):
            for lr_idx in range(len(lr)):
                for model_idx in range(len(overlap_model)):
                    for random_seed in random_seeds:
                        if not do_overlap:
                            continue
                        if overlap_idx == 0:
                            continue
                        config = Config()

                        config.random_seed = random_seed

                        config.train_epoch = 40
                        config.batch_size = bs[bs_idx]
                        config.bert_lr = lr[lr_idx]
                        config.device = 'cuda:0'
                        config.model_dir = os.path.join(config.model_dir, 'overlap')
                        config.overlap_name = overlap_name[overlap_idx]
                        set_random(config.random_seed)
                        print('train prompt %s, overlap = %s, bs = %d, lr = %f, random = %d' % (
                            overlap_model[model_idx][0], overlap_name[overlap_idx], config.batch_size, config.bert_lr, config.random_seed))
                        print_config(config)
                        model = overlap_model[model_idx][1](config)
                        model.train()


def ablation_job():
    from config import Config
    template = [1, 2]
    # auto model
    for t_idx in range(1, len(template)):
        for k in range(1, 6):
            config = Config()
            config.template = template[t_idx]
            config.bert_lr = 1e-5
            config.batch_size = 16
            config.rel_mode = 1
            config.rel_place = 1
            config.rel_elem_num = 60
            config.train_epoch = 30
            config.model_dir = os.path.join('template', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
            config.device = 'cuda:0'
            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
            set_random(config.random_seed)
            print('train prompt auto with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
                config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k))
            print_config(config)
            model = PromptAuto(config)
            model.train()
    # manual model
    for t_idx in range(len(template)):
        for k in range(1, 6):
            config = Config()
            config.template = template[t_idx]
            config.bert_lr = 1e-5
            config.batch_size = 16
            config.rel_mode = 1
            config.rel_place = 0
            config.rel_elem_num = 60
            config.train_epoch = 30
            config.model_dir = os.path.join('template', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
            config.device = 'cuda:0'
            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
            set_random(config.random_seed)
            print('train prompt manual with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
                config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k))
            print_config(config)
            model = PromptManual(config)
            model.train()
    # 
    for i in range(2):
        for k in range(1, 6):
            config = Config()
            config.template = template[i]
            config.bert_lr = 1e-5
            config.batch_size = 16
            config.rel_mode = 0
            config.rel_place = [1, 0][i]
            config.rel_elem_num = 60
            config.train_epoch = 30
            config.model_dir = os.path.join('ablation', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
            config.device = 'cuda:0'
            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
            set_random(config.random_seed)
            print('train static-prompt %s with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
                ['auto', 'manual'][i], config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k + 1))
            print_config(config)
            model = [PromptAuto, PromptManual][i](config)
            model.train()


def num_job():
    from config import Config
    rel_num = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70][8:]
    do_auto = False
    for num_idx in range(len(rel_num)):
        for k in range(1, 6):
            if not do_auto or rel_num[num_idx] == 60:
                continue
            config = Config()
            config.bert_lr = 1e-5
            config.batch_size = 16
            config.template = 1
            config.rel_mode = 1
            config.rel_place = 1
            config.rel_elem_num = rel_num[num_idx]
            config.train_epoch = 30
            config.model_dir = os.path.join('num', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
            config.device = 'cuda:0'
            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
            set_random(config.random_seed)
            print('train prompt auto with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
                config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k + 1))
            print_config(config)
            model = PromptAuto(config)
            model.train()
    do_manual = True
    for num_idx in range(len(rel_num)):
        for k in range(1, 6):
            if not do_manual or rel_num[num_idx] == 60:
                continue
            config = Config()
            config.bert_lr = 1e-5
            config.batch_size = 16
            config.template = 1
            config.rel_mode = 1
            config.rel_place = 0
            config.rel_elem_num = rel_num[num_idx]
            config.train_epoch = 30
            config.model_dir = os.path.join('num', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
            config.device = 'cuda:0'
            config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
            config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
            set_random(config.random_seed)
            print('train prompt manual with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
                config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k + 1))
            print_config(config)
            model = PromptManual(config)
            model.train()


def deal_log():
    log_dir = 'E:\master\python\RE\\relation\PLM\prom-out'
    files = list(filter(lambda x: os.path.isfile(os.path.join(log_dir, x)) and '.out' in x, os.listdir(log_dir)))
    raw_files = list(filter(lambda x: 'new' not in x, files))
    new_files = list(filter(lambda x: 'new' in x, files))
    # dirs = list(filter(lambda x: os.path.isdir(os.path.join(log_dir, x)), os.listdir(log_dir)))
    for idx in range(len(raw_files)):
        out_file = 'new-%s' % raw_files[idx]
        if out_file in new_files:
            continue
        lines = open(os.path.join(log_dir, raw_files[idx]), 'r', encoding='utf-8').readlines()
        print(raw_files[idx], len(lines))
        flag = False
        writer = open(os.path.join(log_dir, out_file), 'w', encoding='utf-8')
        for line in lines:
            if 'bs' in line:
                writer.write(line)
                flag = False
            elif ' test ' in line:
                writer.write(line)
                flag = True
            elif flag:
                writer.write(line)
        writer.close()


def deal_inn_result(in_file):
    """"""
    import re
    bk = '[\s]+'
    log_dir = 'E:\master\python\RE\\relation\PLM\prom-out'
    layer_gold_num = [394, 209, 61, 7]
    layer_g = [392, 198, 52, 5]
    out_file = os.path.join(log_dir, '%s.csv' % (in_file.split('.')[0]))
    writer = open(out_file, 'w', encoding='utf-8')
    lines = open(os.path.join(log_dir, in_file), 'r', encoding='utf-8').readlines()
    idx = 0
    whole_cnt, ratio_cnt, high_cnt, low_cnt, flat_cnt, op_cnt = [0] * 6
    while idx < len(lines):
        if 'bs' in lines[idx]:
            if 'ratio' in lines[idx]:  # ratio
                ratio_cnt += 1
                words = lines[idx].strip().split(' ')
                name, ratio, bs, lr = words[2][:-1], words[7][:-1], words[10][:-1], words[13]
                print(name, ratio, bs, lr)
                idx += 7
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                idx += 2  # precision
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                p1, p2, p3, p4 = [re.split(bk, lines[idx+l_+2].strip())[-1] for l_ in range(4)]
                idx += 13  # recall
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                r1, r2, r3, r4 = [re.split(bk, lines[idx+l_+2].strip())[-1] for l_ in range(4)]
                idx += 13  # f-score
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                f1, f2, f3, f4 = [re.split(bk, lines[idx+l_+2].strip())[-1] for l_ in range(4)]
                idx += 11
                while idx+1 < len(lines) and ('bs' not in lines[idx+1] and 'result' not in lines[idx+1]):
                    idx += 1
                gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4, gf1, gf2, gf3, gf4 = [0] * 12
                if idx+1 < len(lines) and 'bs' not in lines[idx+1]:  # guide result
                    idx += 3  # precision
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gp1, gp2, gp3, gp4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    idx += 13  # recall
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gr1, gr2, gr3, gr4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    idx += 13  # f-score
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gf1, gf2, gf3, gf4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                print(p, r, f, '\n', p1, p2, p3, p4, '\n', r1, r2, r3, r4, '\n', f1, f2, f3, f4,
                      '\n', gp1, gp2, gp3, gp4, '\n', gr1, gr2, gr3, gr4, '\n', gf1, gf2, gf3, gf4)
                if ratio_cnt == 1:
                    writer.write('result on ratio\n')
                    writer.write(','.join(['model', 'ratio', 'bs', 'lr', 'p', 'r', 'f',
                                           'p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'r4', 'f1', 'f2', 'f3', 'f4',
                                           'gp1', 'gp2', 'gp3', 'gp4', 'gr1', 'gr2', 'gr3', 'gr4',
                                           'gf1', 'gf2', 'gf3', 'gf4'])+'\n')
                writer.write(','.join([name, ratio, bs, lr, p, r, f, p1, p2, p3, p4, r1, r2, r3, r4, f1, f2, f3, f4,
                                       gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4, gf1, gf2, gf3, gf4]) + '\n')
            elif 'flat' in lines[idx]:  # flat
                words = lines[idx].strip().split(' ')
                name, flat, bs, lr = words[2][:-1], words[5][:-1], words[8][:-1], words[11]
                print(name, flat, bs, lr)
                flat_cnt += 1
                idx += 4 if flat == 'kbp37' else 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if flat_cnt == 1:
                    writer.write('result on flat\n')
                    writer.write(','.join(['model', 'data', 'bs', 'lr', 'p', 'r', 'f'])+'\n')
                writer.write(','.join([name, flat, bs, lr, p, r, f])+'\n')
            elif 'overlap' in lines[idx]:  # overlap
                words = lines[idx].strip().split(' ')
                name, overlap, bs, lr = words[2][:-1], words[5][:-1], words[8][:-1], words[11]
                print(name, overlap, bs, lr)
                op_cnt += 1
                idx += 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if flat_cnt == 1:
                    writer.write('result on overlap\n')
                    writer.write(','.join(['model', 'data', 'bs', 'lr', 'p', 'r', 'f']) + '\n')
                writer.write(','.join([name, overlap, bs, lr, p, r, f])+'\n')
            elif 'low' in lines[idx]:  # n way k shot low
                low_cnt += 1
                words = lines[idx].strip().split(' ')
                name, mode, k, bs, lr = words[2], words[3], words[7], words[11][:-1], words[14]
                print(name, mode, k, bs, lr)
                idx += 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if low_cnt == 1:
                    writer.write('result on low\n')
                    writer.write(','.join(['model', 'mode', 'k', 'bs', 'lr', 'p', 'r', 'f'])+'\n')
                writer.write(','.join([name, mode, k, bs, lr, p, r, f]) + '\n')
            elif 'high' in lines[idx]:  # n way k shot high
                high_cnt += 1
                words = lines[idx].strip().split(' ')
                name, mode, k, bs, lr = words[2], words[3], words[7], words[11][:-1], words[14]
                print(name, mode, k, bs, lr)
                idx += 5
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if high_cnt == 1:
                    writer.write('result on high\n')
                    writer.write(','.join(['model', 'mode', 'k', 'bs', 'lr', 'p', 'r', 'f'])+'\n')
                writer.write(','.join([name, mode, k, bs, lr, p, r, f]) + '\n')
            else:  # whole
                whole_cnt += 1
                words = lines[idx].strip().split(' ')
                name, bs, lr, fold = words[2][:-1], words[5][:-1], words[8][:-1], words[11]
                print(name, bs, lr, fold)
                idx += 7
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                idx += 2  # precision
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                p1, p2, p3, p4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # recall
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                r1, r2, r3, r4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # f-score
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                f1, f2, f3, f4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 11
                while idx + 1 < len(lines) and ('bs' not in lines[idx + 1] and 'result' not in lines[idx + 1]):
                    idx += 1
                gp, gp1, gp2, gp3, gp4, gr, gr1, gr2, gr3, gr4, gf, gf1, gf2, gf3, gf4 = [0] * 15
                if idx + 1 < len(lines) and 'bs' not in lines[idx + 1]:  # guide result
                    idx += 3  # precision
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gp1, gp2, gp3, gp4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gp = re.split(bk, lines[idx+11].strip())[-1]
                    idx += 13  # recall
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gr1, gr2, gr3, gr4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gr = re.split(bk, lines[idx+11].strip())[-1]
                    idx += 13  # f-score
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gf1, gf2, gf3, gf4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gf = re.split(bk, lines[idx+11].strip())[-1]
                cor34 = float(r3)*layer_gold_num[2]+float(r4)*layer_gold_num[3]
                gcor34 = float(gr3)*layer_g[2]+float(gr4)*layer_g[3]
                pre3 = 0 if float(p3) == 0 else float(r3)*layer_gold_num[2]/float(p3)
                pre4 = 0 if float(p4) == 0 else float(r4)*layer_gold_num[3]/float(p4)
                pre34 = pre3 + pre4
                gpre3 = 0 if float(gp3) == 0 else float(gr3)*layer_g[2]/float(gp3)
                gpre4 = 0 if float(gp4) == 0 else float(gr4)*layer_g[3]/float(gp4)
                gpre34 = gpre3 + gpre4
                f34, gf34 = 2*cor34/(pre34+sum(layer_gold_num[2:])), 2*gcor34/(gpre34+sum(layer_g[2:]))
                print(p, r, f, '\n', p1, p2, p3, p4, '\n', r1, r2, r3, r4, '\n', f1, f2, f3, f4,
                      '\n', gp, gr, gf, '\n', gp1, gp2, gp3, gp4, '\n', gr1, gr2, gr3, gr4, '\n',
                      gf1, gf2, gf3, gf4, '\n', f34, gf34)
                if whole_cnt == 1:
                    writer.write('result on whole\n')
                    writer.write(','.join(['model', 'bs', 'lr', 'fold', 'p', 'r', 'f',
                                           'p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'r4', 'f1', 'f2', 'f3', 'f4',
                                           'gp', 'gr', 'gf',
                                           'gp1', 'gp2', 'gp3', 'gp4', 'gr1', 'gr2', 'gr3', 'gr4',
                                           'gf1', 'gf2', 'gf3', 'gf4', 'f34', 'gf34'])+'\n')
                writer.write(','.join([name, bs, lr, fold, p, r, f, p1, p2, p3, p4, r1, r2, r3, r4, f1, f2, f3, f4,
                                       gp, gr, gf, gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4,
                                       gf1, gf2, gf3, gf4, str(f34), str(gf34)]) + '\n')
                if int(fold) == 5:
                    writer.write('avg\n')
        idx += 1
    writer.close()


def deal_prompt_result(in_file):
    import re
    bk = r'[\s]+'
    log_dir = 'E:\master\python\RE\\relation\PLM\prom-out'
    layer_gold_num = [394, 209, 61, 7]
    layer_g = [392, 198, 52, 5]
    out_file = os.path.join(log_dir, '%s.csv' % (in_file.split('.')[0]))
    writer = open(out_file, 'w', encoding='utf-8')
    lines = open(os.path.join(log_dir, in_file), 'r', encoding='utf-8').readlines()
    idx = 0
    whole_cnt, ratio_cnt, high_cnt, low_cnt, flat_cnt, op_cnt = [0] * 6
    while idx < len(lines):
        if 'bs' in lines[idx]:
            if 'fold' in lines[idx]:  # whole
                whole_cnt += 1
                words = lines[idx].strip().split(' ')
                name, lr, bs = words[2], words[6][:-1], words[9][:-1]
                rel_p, rel_n, fold = words[12][:-1], words[15][:-1], words[16][-1:]
                print(name, lr, bs, rel_p, rel_n, fold)
                idx += 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                idx += 2  # precision
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                p1, p2, p3, p4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # recall
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                r1, r2, r3, r4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # f-score
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                f1, f2, f3, f4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 11
                while idx + 1 < len(lines) and ('bs' not in lines[idx + 1] and 'result' not in lines[idx + 1]):
                    idx += 1
                gp, gr, gf, gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4, gf1, gf2, gf3, gf4 = [0] * 15
                if idx + 1 < len(lines) and 'bs' not in lines[idx + 1]:  # guide result
                    idx += 3  # precision
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gp1, gp2, gp3, gp4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gp = re.split(bk, lines[idx+11].strip())[-1]
                    idx += 13  # recall
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gr1, gr2, gr3, gr4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gr = re.split(bk, lines[idx+11].strip())[-1]
                    idx += 13  # f-score
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gf1, gf2, gf3, gf4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    gf = re.split(bk, lines[idx+11].strip())[-1]
                cor34 = float(r3) * layer_gold_num[2] + float(r4) * layer_gold_num[3]
                gcor34 = float(gr3) * layer_g[2] + float(gr4) * layer_g[3]
                pre3 = 0 if float(p3) == 0 else float(r3) * layer_gold_num[2] / float(p3)
                pre4 = 0 if float(p4) == 0 else float(r4) * layer_gold_num[3] / float(p4)
                pre34 = pre3 + pre4
                gpre3 = 0 if float(gp3) == 0 else float(gr3) * layer_g[2] / float(gp3)
                gpre4 = 0 if float(gp4) == 0 else float(gr4) * layer_g[3] / float(gp4)
                gpre34 = gpre3 + gpre4
                f34, gf34 = 2 * cor34 / (pre34 + sum(layer_gold_num[2:])), 2 * gcor34 / (gpre34 + sum(layer_g[2:]))
                print(p, r, f, '\n', p1, p2, p3, p4, '\n', r1, r2, r3, r4, '\n', f1, f2, f3, f4,
                      '\n', gp1, gp2, gp3, gp4, '\n', gr1, gr2, gr3, gr4, '\n', gf1, gf2, gf3, gf4, '\n', f34, gf34)
                if whole_cnt == 1:
                    writer.write('result on whole\n')
                    writer.write(','.join(['model', 'bs', 'lr', 'rel_p', 'rel_n', 'fold', 'p', 'r', 'f',
                                           'p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'r4', 'f1', 'f2', 'f3', 'f4',
                                           'gp', 'gr', 'gf', 'gp1', 'gp2', 'gp3', 'gp4', 'gr1', 'gr2', 'gr3', 'gr4',
                                           'gf1', 'gf2', 'gf3', 'gf4', 'f34', 'gf34']) + '\n')
                writer.write(','.join([name, bs, lr, rel_p, rel_n, fold, p, r, f, p1, p2, p3, p4, r1, r2, r3, r4,
                                       f1, f2, f3, f4, gp, gr, gf, gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4,
                                       gf1, gf2, gf3, gf4, str(f34), str(gf34)]) + '\n')
                if int(fold) == 6:
                    writer.write('avg\n')
            elif 'few' in lines[idx]:
                ratio_cnt += 1
                words = lines[idx].strip().split(' ')
                name, ratio, bs, lr = words[4], words[2], words[11][:-1], words[8][:-1]
                rel_p, rel_n = words[-4][:-1], words[-1]
                print(name, ratio, bs, lr, rel_p, rel_n)
                idx += 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                idx += 2  # precision
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                p1, p2, p3, p4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # recall
                if len(re.split(bk, lines[idx].strip())) < 12:
                    idx += 13
                r1, r2, r3, r4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 13  # f-score
                f1, f2, f3, f4 = [re.split(bk, lines[idx + l_ + 2].strip())[-1] for l_ in range(4)]
                idx += 11
                while idx+1 < len(lines) and ('bs' not in lines[idx+1] and 'result' not in lines[idx+1]):
                    idx += 1
                gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4, gf1, gf2, gf3, gf4 = [0] * 12
                if idx + 1 < len(lines) and 'bs' not in lines[idx + 1]:  # guide result
                    idx += 3  # precision
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gp1, gp2, gp3, gp4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    idx += 13  # recall
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gr1, gr2, gr3, gr4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                    idx += 13  # f-score
                    if len(re.split(bk, lines[idx].strip())) < 12:
                        idx += 13
                    gf1, gf2, gf3, gf4 = [re.split(bk, lines[idx + l_ + 1].strip())[-1] for l_ in range(4)]
                print(p, r, f, '\n', p1, p2, p3, p4, '\n', r1, r2, r3, r4, '\n', f1, f2, f3, f4,
                      '\n', gp1, gp2, gp3, gp4, '\n', gr1, gr2, gr3, gr4, '\n', gf1, gf2, gf3, gf4)
                if ratio_cnt == 1:
                    writer.write('result on ratio\n')
                    writer.write(','.join(['model', 'ratio', 'bs', 'lr', 'rel_p', 'rel_n', 'p', 'r', 'f',
                                           'p1', 'p2', 'p3', 'p4', 'r1', 'r2', 'r3', 'r4', 'f1', 'f2', 'f3', 'f4',
                                           'gp1', 'gp2', 'gp3', 'gp4', 'gr1', 'gr2', 'gr3', 'gr4',
                                           'gf1', 'gf2', 'gf3', 'gf4']) + '\n')
                writer.write(','.join([name, ratio, bs, lr, rel_p, rel_n, p, r, f, p1, p2, p3, p4, r1, r2, r3, r4, f1, f2, f3, f4,
                                       gp1, gp2, gp3, gp4, gr1, gr2, gr3, gr4, gf1, gf2, gf3, gf4]) + '\n')
            elif 'flat' in lines[idx]:  # flat
                words = lines[idx].strip().split(' ')
                name, flat, bs, lr = words[2][:-1], words[5][:-1], words[8][:-1], words[11]
                print(name, flat, bs, lr)
                flat_cnt += 1
                idx += 4 if flat == 'kbp37' else 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if flat_cnt == 1:
                    writer.write('result on flat\n')
                    writer.write(','.join(['model', 'data', 'bs', 'lr', 'p', 'r', 'f']) + '\n')
                writer.write(','.join([name, flat, bs, lr, p, r, f]) + '\n')
            elif 'overlap' in lines[idx]:  # overlap
                words = lines[idx].strip().split(' ')
                name, overlap, bs, lr = words[2][:-1], words[5][:-1], words[8][:-1], words[11]
                print(name, overlap, bs, lr)
                op_cnt += 1
                idx += 1
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if flat_cnt == 1:
                    writer.write('result on overlap\n')
                    writer.write(','.join(['model', 'data', 'bs', 'lr', 'p', 'r', 'f']) + '\n')
                writer.write(','.join([name, overlap, bs, lr, p, r, f]) + '\n')
            elif 'high' in lines[idx]:  # n way k shot high
                high_cnt += 1
                words = lines[idx].strip().split(' ')
                name, bs, lr, mode, k = words[7], words[14][:-1], words[11][:-1], words[1], words[4]
                print(name, bs, lr, mode, k)
                idx += 3
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if high_cnt == 1:
                    writer.write('result on high\n')
                    writer.write(','.join(['model', 'mode', 'k', 'bs', 'lr', 'p', 'r', 'f']) + '\n')
                writer.write(','.join([name, mode, k, bs, lr, p, r, f]) + '\n')
            elif 'low' in lines[idx]:  # n way k shot low
                low_cnt += 1
                words = lines[idx].strip().split(' ')
                name, bs, lr, mode, k = words[7], words[14][:-1], words[11][:-1], words[1], words[4]
                print(name, bs, lr, mode, k)
                while idx + 1 < len(lines) and 'target result' not in lines[idx+1]:
                    idx += 1
                if idx + 1 < len(lines):
                    idx += 1
                words = lines[idx].strip().split(' ')
                p, r, f = words[-7][:-1], words[-4][:-1], words[-1]
                if low_cnt == 1:
                    writer.write('result on low\n')
                    writer.write(','.join(['model', 'mode', 'k', 'bs', 'lr', 'p', 'r', 'f']) + '\n')
                writer.write(','.join([name, mode, k, bs, lr, p, r, f]) + '\n')
        idx += 1

    writer.close()


def ablation():
    from config import Config
    for k in range(1, 6):
        config = Config()
        config.bert_lr = 1e-5
        config.batch_size = 16
        config.template = 2
        config.rel_place = 0
        config.rel_elem_num = 55
        config.rel_mode = 1
        config.device = 'cuda:0'
        config.train_epoch = 30
        config.model_dir = os.path.join('ablation', 'rp-%d-rn-%d' % (config.rel_place, config.rel_elem_num))
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
        set_random(config.random_seed)
        print('train static-prompt %s with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
            'raw', config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k
        ))
        model = PromptAuto(config)
        model.train()


def ablation2():
    from config import Config
    do_auto = False
    for k in range(1, 6):
        if not do_auto:
            continue
        config = Config()
        config.bert_lr = 1e-5
        config.batch_size = 16
        config.template = 2
        config.rel_mode = 0
        config.rel_place = 0
        config.rel_elem_num = 60
        config.train_epoch = 30
        config.model_dir = os.path.join(
            'ablation', 'p%dn%dt%d' % (config.rel_place, config.rel_elem_num, config.template))
        config.device = 'cuda:0'
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
        set_random(config.random_seed)
        print('train prompt auto with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
            config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k+1
        ))
        print_config(config)
        model = PromptAuto(config)
        model.train()

    do_manual = True
    for k in range(1, 6):
        if not do_manual:
            continue
        config = Config()
        config.bert_lr = 1e-5
        config.batch_size = 16
        config.template = 1
        config.rel_mode = 0
        config.rel_place = 0
        config.rel_elem_num = 60
        config.train_epoch = 30
        config.model_dir = os.path.join(
            'ablation2', 'p%dn%dt%d' % (config.rel_place, config.rel_elem_num, config.template))
        config.device = 'cuda:0'
        config.data_dir = os.path.join(config.data_dir, '%d-%d' % (5, k))
        config.model_dir = os.path.join(config.model_dir, '%d-%d' % (5, k))
        set_random(config.random_seed)
        print('train prompt manual with lr = %f, bs = %d, rel_place = %d, rel_num = %d, fold=%d' % (
            config.bert_lr, config.batch_size, config.rel_place, config.rel_elem_num, k + 1
        ))
        print_config(config)
        model = PromptManual(config)
        model.train()

if __name__ == '__main__':
    my_job()
    # ablation_job()
    # ablation()
    # ablation2()
    # num_job()
    # deal_log()
    # deal_inn_result('new-INN-few.out')
    # deal_prompt_result('new-prompt-ab-man.out')
    # deal_prompt_result('new-prompt-job1-0116.out')

import json
import numpy as np
import os


def is_normal_triple(triples, is_relation_first=False):
    entities = set()
    for i, e in enumerate(triples):
        key = 0 if is_relation_first else 2
        if i % 3 != key:
            entities.add(e)
    return len(entities) == 2 * int(len(triples) / 3)


def is_multi_label(triples, is_relation_first=False):
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(int(len(triples) / 3))]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    # if is multi label, then, at least one entity pair appeared more than once
    return len(entity_pair) != len(set(entity_pair))


def is_over_lapping(triples, is_relation_first=False):
    if is_normal_triple(triples, is_relation_first):
        return False
    if is_relation_first:
        entity_pair = [tuple(triples[3 * i + 1: 3 * i + 3]) for i in range(int(len(triples) / 3))]
    else:
        entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    # remove the same entity_pair, then, if one entity appear more than once, it's overlapping
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)


def load_data(in_file, word_dict, rel_dict, out_file, normal_file, epo_file, seo_file):
    with open(in_file, 'r') as f1, open(out_file, 'w') as f2:  # , open(normal_file, 'w') as f3, open(epo_file, 'w') as f4, open(seo_file, 'w') as f5:
        cnt_normal = 0
        cnt_epo = 0
        cnt_seo = 0
        lines = f1.readlines()
        all_rel = []
        max_sent_len = 0
        for line in lines:
            line = json.loads(line)
            print(len(line))
            sents, spos = line[-2], line[-1]
            print(len(spos))
            print(len(sents))
            for i in range(len(sents)):
                new_line = dict()
                tokens = [word_dict[t] for t in sents[i]]
                # sent = ' '.join(tokens)
                new_line['tokens'] = tokens
                max_sent_len = max(max_sent_len, len(tokens))
                triples = np.reshape(spos[i], (-1, 3))
                # entity = []
                relationMentions = []
                for triple in triples:
                    rel = dict()
                    rel['e1'] = {'entity': tokens[triple[0]], 'pos': int(triple[0])}
                    rel['e2'] = {'entity': tokens[triple[1]], 'pos': int(triple[1])}
                    # entity.extend([triple[0], triple[1]])
                    rel['label'] = rel_dict[triple[2]]
                    all_rel.append(rel['label'])
                    relationMentions.append(rel)
                    # if rel['em1Text'] == rel['em2Text']:
                    #     print(i+1, sent, rel)
                new_line['relation'] = relationMentions
                f2.write(json.dumps(new_line) + '\n')
                # if is_normal_triple(spos[i]):
                #     f3.write(json.dumps(new_line) + '\n')
                # if is_multi_label(spos[i]):
                #     f4.write(json.dumps(new_line) + '\n')
                # if is_over_lapping(spos[i]):
                #     f5.write(json.dumps(new_line) + '\n')
        print('max sentence length = ', max_sent_len)
        return all_rel


def deal_raw(data_dir='NYT'):
    file_mode = ['train', 'valid', 'test']
    all_rel = []
    for m in file_mode:
        raw_file = os.path.join(data_dir, 'raw/%s.json' % m)
        print('deal with file %s' % raw_file)
        out_file = os.path.join(data_dir, '%s.txt' % m)
        rel_file = os.path.join(data_dir, 'raw/relations2id.json')
        word_file = os.path.join(data_dir, 'raw/words2id.json')
        with open(rel_file, 'r') as f1, open(word_file, 'r') as f2:
            rel2id = json.load(f1)
            words2id = json.load(f2)
        rel_dict = {j: i for i, j in rel2id.items()}
        word_dict = {j: i for i, j in words2id.items()}
        tmp_rel = load_data(raw_file, word_dict, rel_dict, out_file, None, None, None)
        if m != 'test':
            all_rel.extend(tmp_rel)
    all_rel = list(sorted(set(all_rel)))
    with open(os.path.join(data_dir, 'rel.txt'), 'w') as f:
        f.write(json.dumps({'relation': all_rel}))


if __name__ == '__main__':
    deal_raw('WebNLG')
    deal_raw()

import os
import re
import random
import codecs
import logging
from torch.utils.data import Dataset

TOKEN_PAD = '[PAD]'
TOKEN_UNK = '[UNK]'
TOKEN_CLS = '[CLS]'
TOKEN_SEP = '[SEP]'
TOKEN_EDGES_START = '<s>'
TOKEN_EDGES_END = '</s>'
SEG_TAG_TO_ID = {'B': 0, 'M': 1, 'E': 2, 'S': 3}


class BaseDataset(Dataset):
    def __init__(self, data_path, char_or_word, max_sent_len=0, max_word_len=0,
                 do_pad=False, pad_token=TOKEN_PAD, do_to_id=False, tag_to_id=None,
                 char_tokenizer=None, add_bigram_feature=False, bigram_tokenizer=None,
                 word_tokenizer=None, add_char_feature=False,
                 for_bert=False, do_sort=False, debug=False):
        super(BaseDataset, self).__init__()
        self.read_file(data_path, char_or_word, max_sent_len, max_word_len, do_pad, pad_token, do_to_id, tag_to_id,
                       char_tokenizer, add_bigram_feature, bigram_tokenizer, word_tokenizer, add_char_feature,
                       for_bert, do_sort, debug)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read_file(self, data_path, char_or_word, max_sent_len=0, max_word_len=0,
                  do_pad=False, pad_token=TOKEN_PAD, do_to_id=False, tag_to_id=None,
                  char_tokenizer=None, add_bigram_feature=False, bigram_tokenizer=None,
                  word_tokenizer=None, add_char_feature=False,
                  for_bert=False, do_sort=False, debug=False):
        self.data = []
        sent = []
        seg_tag = []
        pos_tag = []

        with codecs.open(data_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    if char_or_word == 'char':
                        sent, bigram, seg_tag, pos_tag, mask, sent_len = self.process_char_seq(
                            sent, seg_tag, pos_tag, max_sent_len, do_pad, pad_token,
                            do_to_id, tag_to_id, char_tokenizer, add_bigram_feature, bigram_tokenizer, for_bert)
                        self.data.append([sent, bigram, seg_tag, pos_tag, mask, sent_len])
                    elif char_or_word == 'word':
                        sent, chars, pos_tag, mask, sent_len = self.process_word_seq(
                            sent, pos_tag, max_sent_len, max_word_len, do_pad, pad_token,
                            do_to_id, tag_to_id, word_tokenizer, add_char_feature, char_tokenizer, for_bert)
                        self.data.append([sent, chars, pos_tag, mask, sent_len])

                    sent = []
                    seg_tag = []
                    pos_tag = []

                    if debug:
                        if len(self.data) >= 10:
                            break
                else:
                    if char_or_word == 'char':
                        word, seg, pos = line.split()
                        sent.append(word)
                        seg_tag.append(seg)
                        pos_tag.append(pos)
                    else:
                        word, pos = line.split()
                        sent.append(word)
                        pos_tag.append(pos)

            if len(sent) > 0:
                if char_or_word == 'char':
                    sent, bigram, seg_tag, pos_tag, mask, sent_len = self.process_char_seq(
                        sent, seg_tag, pos_tag, max_sent_len, do_pad, pad_token,
                        do_to_id, tag_to_id, char_tokenizer, add_bigram_feature, bigram_tokenizer, for_bert)
                    self.data.append([sent, bigram, seg_tag, pos_tag, mask, sent_len])
                elif char_or_word == 'word':
                    sent, chars, pos_tag, mask, sent_len = self.process_word_seq(
                        sent, pos_tag, max_sent_len, max_word_len, do_pad, pad_token,
                        do_to_id, tag_to_id, word_tokenizer, add_char_feature, char_tokenizer, for_bert)
                    self.data.append([sent, chars, pos_tag, mask, sent_len])

        if do_sort:
            if char_or_word == 'char':
                self.data = sorted(self.data, key=lambda x: x[5], reverse=True)
            elif char_or_word == 'word':
                self.data = sorted(self.data, key=lambda x: x[4], reverse=True)

    def process_char_seq(self, sent, seg_tag, pos_tag, max_sent_len, do_pad, pad_token,
                         do_to_id, tag_to_id, char_tokenizer, add_bigram_feature, bigram_tokenizer, for_bert):
        sent_len = len(sent)
        sent_len = max_sent_len if sent_len > max_sent_len else sent_len

        if for_bert:
            sent = [TOKEN_CLS] + sent
            seg_tag = ['S'] + seg_tag
            pos_tag = ['S-X'] + pos_tag

        sent = sent[:max_sent_len]
        seg_tag = seg_tag[:max_sent_len]
        pos_tag = pos_tag[:max_sent_len]
        bigram = []

        if add_bigram_feature:
            bigram = [TOKEN_EDGES_START] + sent + [TOKEN_EDGES_END]
            bigram = [[bigram[i - 1] + bigram[i]] + [bigram[i] + bigram[i + 1]] for i in range(1, len(bigram) - 1)]

        if do_pad:
            mask = [1] * len(sent) + [0] * (max_sent_len - len(sent))
            seg_tag = seg_tag + ['S'] * (max_sent_len - len(sent))
            pos_tag = pos_tag + ['S-X'] * (max_sent_len - len(sent))

            if add_bigram_feature:
                bigram = bigram + [[pad_token, pad_token]] * (max_sent_len - len(sent))

            sent = sent + [pad_token] * (max_sent_len - len(sent))
        else:
            mask = [1] * len(sent)

        if do_to_id:
            sent = char_tokenizer.convert_tokens_to_ids(sent)

            if add_bigram_feature:
                bigram = bigram_tokenizer.convert_tokens_to_ids(bigram)

            seg_tag = [SEG_TAG_TO_ID.get(tag) for tag in seg_tag]
            pos_tag = [tag_to_id.get(tag) for tag in pos_tag]

        return sent, bigram, seg_tag, pos_tag, mask, sent_len

    def process_word_seq(self, sent, pos_tag, max_sent_len, max_word_len, do_pad, pad_token,
                         do_to_id, tag_to_id, word_tokenizer, add_char_feature, char_tokenizer, for_bert):
        sent_len = len(sent)
        sent_len = max_sent_len if sent_len > max_sent_len else sent_len

        if for_bert:
            sent = [TOKEN_CLS] + sent
            pos_tag = ['X'] + pos_tag

        sent = sent[:max_sent_len]
        pos_tag = pos_tag[:max_sent_len]
        chars = []

        if add_char_feature:
            chars = [[c for c in sent[i]] + [pad_token] * (max_word_len - len(sent[i])) for i in range(len(sent))]

        if do_pad:
            mask = [1] * len(sent) + [0] * (max_sent_len - len(sent))
            pos_tag = pos_tag + ['X'] * (max_sent_len - len(sent))

            if add_char_feature:
                chars = chars + [[pad_token] * (max_word_len if max_word_len > 0 else 1)] * (max_sent_len - len(sent))

            sent = sent + [pad_token] * (max_sent_len - len(sent))
        else:
            mask = [1] * len(sent)

        if do_to_id:
            sent = word_tokenizer.convert_tokens_to_ids(sent)

            if add_char_feature:
                chars = char_tokenizer.convert_tokens_to_ids(chars)

            pos_tag = [tag_to_id.get(tag) for tag in pos_tag]

        return sent, chars, pos_tag, mask, sent_len

    @staticmethod
    def dbc_to_sbc(ustring):
        rstring = ''
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if not (0x0021 <= inside_code and inside_code <= 0x7e):
                rstring += uchar
                continue
            rstring += chr(inside_code)
        return rstring

    @staticmethod
    def save_data(filepath, tags, train_sents, dev_sents, test_sents, postfix):
        logging.info('writing data to: %s' % filepath)
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with codecs.open('%s/tags_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for tag in tags:
                fout.write('%s\n' % tag)

        with codecs.open('%s/train_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for sent in train_sents:
                for item in sent:
                    line = '\t'.join(item)
                    fout.write('%s\n' % line)
                fout.write('\n')
        with codecs.open('%s/dev_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for sent in dev_sents:
                for item in sent:
                    line = '\t'.join(item)
                    fout.write('%s\n' % line)
                fout.write('\n')
        with codecs.open('%s/test_%s.txt' % (filepath, postfix), 'w', 'utf-8') as fout:
            for sent in test_sents:
                for item in sent:
                    line = '\t'.join(item)
                    fout.write('%s\n' % line)
                fout.write('\n')

        logging.info('complete writed')

    @staticmethod
    def statistics(datapath):
        sent_len_dict = {100: 0, 150: 0, 200: 0, 250: 0, 300: 0, 350: 0, 1000: 0}

        data_granularity = ['word', 'char']
        data_split = ['train', 'dev', 'test']
        for dg in data_granularity:
            logging.info('%s level data statistics' % dg)
            for ds in data_split:
                with codecs.open('%s/%s_%s.txt' % (datapath, ds, dg), 'r', 'utf-8') as fin:
                    sent_len = 0
                    for line in fin:
                        line = line.strip()
                        if line == '':
                            for sl in sent_len_dict:
                                if sent_len <= sl:
                                    sent_len_dict[sl] = sent_len_dict[sl] + 1
                                    break
                            sent_len = 0
                        else:
                            sent_len += 1
                logging.info('%s data length: %s' % (ds, str(sent_len_dict)))

                for sl in sent_len_dict:
                    sent_len_dict[sl] = 0


class CTB8Dataset(BaseDataset):
    def __init__(self):
        pass

    @staticmethod
    def transform(src_path, tgt_path):
        train_index, dev_index, test_index = CTB8Dataset.get_data_split()

        train_sents_word = []
        dev_sents_word = []
        test_sents_word = []
        tags_word = set()

        train_sents_char = []
        dev_sents_char = []
        test_sents_char = []
        tags_char = set()

        filenames = os.listdir(src_path)
        for filename in filenames:
            idx = int(filename[filename.index('_') + 1:filename.index('.')])
            with codecs.open('%s/%s' % (src_path, filename), 'r', 'utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line == '':
                        continue
                    if re.match(r'<[^>]+>', line):
                        continue

                    line = BaseDataset.dbc_to_sbc(line)

                    words = []
                    chars = []
                    line = line.split()
                    for term in line:
                        sep = term.rindex('_')

                        word = term[:sep]
                        tag = term[sep + 1:]
                        if '-' in tag:
                            tag = tag[:tag.index('-')]

                        words.append([word, tag])
                        tags_word.add(tag)

                        chs = [ch for ch in word]
                        if len(chs) == 1:
                            seg_t = ['S']
                            pos_t = ['S-%s' % tag]
                        else:
                            seg_t = ['B'] + ['M'] * (len(chs) - 2) + ['E']
                            pos_t = ['B-%s' % tag] + ['M-%s' % tag] * (len(chs) - 2) + ['E-%s' % tag]

                        assert len(chs) == len(seg_t) == len(pos_t)

                        for i in range(len(chs)):
                            chars.append([chs[i], seg_t[i], pos_t[i]])
                            tags_char.add(pos_t[i])

                    if idx in train_index:
                        train_sents_word.append(words)
                        train_sents_char.append(chars)
                    elif idx in dev_index:
                        dev_sents_word.append(words)
                        dev_sents_char.append(chars)
                    elif idx in test_index:
                        test_sents_word.append(words)
                        test_sents_char.append(chars)
                    else:
                        train_sents_word.append(words)
                        train_sents_char.append(chars)

        tags_word = list(tags_word)
        tags_word.sort()
        tags_char = list(tags_char)
        tags_char.sort()

        logging.info('num of train sents: %s, dev setns: %s, test sents: %s' % (
            len(train_sents_word), len(dev_sents_word), len(test_sents_word)))
        logging.info('num of word tags: %s, char tags: %s' % (len(tags_word), len(tags_char)))

        BaseDataset.save_data(tgt_path, tags_word, train_sents_word, dev_sents_word, test_sents_word, 'word')
        BaseDataset.save_data(tgt_path, tags_char, train_sents_char, dev_sents_char, test_sents_char, 'char')

    @staticmethod
    def get_data_split():
        train_index = [i for i in range(44, 144)]
        train_index.extend([i for i in range(170, 271)])
        train_index.extend([i for i in range(400, 900)])
        train_index.extend([i for i in range(1001, 1018)])
        train_index.extend([1019])
        train_index.extend([i for i in range(1021, 1036)])
        train_index.extend([i for i in range(1037, 1044)])
        train_index.extend([i for i in range(1045, 1060)])
        train_index.extend([i for i in range(1062, 1072)])
        train_index.extend([i for i in range(1073, 1118)])
        train_index.extend([i for i in range(1120, 1132)])
        train_index.extend([i for i in range(1133, 1141)])
        train_index.extend([i for i in range(1143, 1148)])
        train_index.extend([i for i in range(1149, 1152)])
        train_index.extend([i for i in range(2000, 2916)])
        train_index.extend([i for i in range(4051, 4100)])
        train_index.extend([i for i in range(4112, 4181)])
        train_index.extend([i for i in range(4198, 4369)])
        train_index.extend([i for i in range(5000, 5447)])
        train_index = set(train_index)

        dev_index = [i for i in range(301, 327)]
        dev_index.extend([i for i in range(2916, 3031)])
        dev_index.extend([i for i in range(4100, 4107)])
        dev_index.extend([i for i in range(4181, 4190)])
        dev_index.extend([i for i in range(4369, 4391)])
        dev_index.extend([i for i in range(5447, 5493)])
        dev_index = set(dev_index)

        test_index = [i for i in range(1, 44)]
        test_index.extend([i for i in range(144, 170)])
        test_index.extend([i for i in range(271, 301)])
        test_index.extend([i for i in range(900, 932)])
        test_index.extend([1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132, 1141, 1142, 1148])
        test_index.extend([i for i in range(3031, 3146)])
        test_index.extend([i for i in range(4107, 4112)])
        test_index.extend([i for i in range(4190, 4198)])
        test_index.extend([i for i in range(4391, 4412)])
        test_index.extend([i for i in range(5493, 5559)])
        test_index = set(test_index)

        return train_index, dev_index, test_index


class PFRDataset(BaseDataset):
    def __init__(self):
        pass

    @staticmethod
    def transform(src_path, tgt_path):
        sents_word = []
        tags_word = set()

        sents_char = []
        tags_char = set()

        with codecs.open(src_path, 'r', 'gbk') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue

                line = BaseDataset.dbc_to_sbc(line)
                line = PFRDataset.merge_ner(line)
                line = PFRDataset.remove_polyphone(line)

                line = line.split()[1:]
                line = PFRDataset.merge_name(line)
                line = PFRDataset.merge_time(line)

                words = []
                chars = []
                for term in line:
                    sep = term.rindex('/')

                    word = term[:sep]
                    tag = term[sep + 1:]

                    word = PFRDataset.get_word(word)
                    tag = PFRDataset.get_tag(tag)

                    words.append([word, tag])
                    tags_word.add(tag)

                    chs = [ch for ch in word]
                    if len(chs) == 1:
                        seg_t = ['S']
                        pos_t = ['S-%s' % tag]
                    else:
                        seg_t = ['B'] + ['M'] * (len(chs) - 2) + ['E']
                        pos_t = ['B-%s' % tag] + ['M-%s' % tag] * (len(chs) - 2) + ['E-%s' % tag]

                    assert len(chs) == len(seg_t) == len(pos_t)

                    for i in range(len(chs)):
                        chars.append([chs[i], seg_t[i], pos_t[i]])
                        tags_char.add(pos_t[i])

                sents_word.append(words)
                sents_char.append(chars)

        train_idx, dev_idx, test_idx = PFRDataset.get_data_split(len(sents_word))

        train_sents_word = []
        dev_sents_word = []
        test_sents_word = []

        train_sents_char = []
        dev_sents_char = []
        test_sents_char = []

        for i in range(len(sents_word)):
            if i in train_idx:
                train_sents_word.append(sents_word[i])
                train_sents_char.append(sents_char[i])
            elif i in dev_idx:
                dev_sents_word.append(sents_word[i])
                dev_sents_char.append(sents_char[i])
            elif i in test_idx:
                test_sents_word.append(sents_word[i])
                test_sents_char.append(sents_char[i])
            else:
                train_sents_word.append(sents_word[i])
                train_sents_char.append(sents_char[i])

        tags_word = list(tags_word)
        tags_word.sort()
        tags_char = list(tags_char)
        tags_char.sort()

        logging.info('num of train sents: %s, dev setns: %s, test sents: %s' % (
            len(train_sents_word), len(dev_sents_word), len(test_sents_word)))
        logging.info('num of word tags: %s, char tags: %s' % (len(tags_word), len(tags_char)))

        BaseDataset.save_data(tgt_path, tags_word, train_sents_word, dev_sents_word, test_sents_word, 'word')
        BaseDataset.save_data(tgt_path, tags_char, train_sents_char, dev_sents_char, test_sents_char, 'char')

    @staticmethod
    def merge_ner(line):
        line = re.sub(r'\[(.*?)\](.*?)\s+',
                      lambda x: ''.join(re.split('/\w+', x.group(1))).replace(' ', '') \
                                + '/' + x.group(2) + '  ', line)
        line = re.sub(r'\[(.*?)\](.*?)$',
                      lambda x: ''.join(re.split('/\w+', x.group(1))).replace(' ', '') \
                                + '/' + x.group(2) + '  ', line)
        return line

    @staticmethod
    def remove_polyphone(line):
        line = re.sub(r'\{(\w+?)\}', '', line)
        return line

    @staticmethod
    def merge_name(token_list):
        tgt_token_list = []
        tokens_need_merge = []
        for token in token_list:
            word = token[:token.rindex('/')]
            pos = token[token.rindex('/') + 1:]
            if pos == 'nrf':
                tokens_need_merge.append(word)
            elif pos == 'nrg':
                tokens_need_merge.append(word)

                tgt_token_list.append(''.join([w for w in tokens_need_merge]) + '/nr')

                tokens_need_merge = []
            else:
                tgt_token_list.append(token)

        if len(tokens_need_merge) > 0:
            tgt_token_list.append(''.join([w for w in tokens_need_merge]) + '/nr')

        return tgt_token_list

    @staticmethod
    def merge_time(token_list):
        tgt_token_list = []
        idx = 0
        while idx < len(token_list):
            token = token_list[idx]
            token_next = token_list[idx + 1] if idx < len(token_list) - 1 else None
            token_next_two = token_list[idx + 2] if idx < len(token_list) - 2 else None

            if re.match(r'(.*年)\/t', token):
                text = re.match(r'(.*年)\/t', token).group(1)

                if (token_next is not None) and re.match(r'(.*月)\/t', token_next):
                    text += re.match(r'(.*月)\/t', token_next).group(1)
                    idx += 1

                    if (token_next_two is not None) and re.match(r'(.*日)\/t', token_next_two):
                        text += re.match(r'(.*日)\/t', token_next_two).group(1)
                        idx += 1

                tgt_token_list.append('%s/t' % text)

            elif re.match(r'(.*月)\/t', token):
                text = re.match(r'(.*月)\/t', token).group(1)

                if re.match(r'(.*日)\/t', token_next):
                    text += re.match(r'(.*日)\/t', token_next).group(1)
                    idx += 1

                tgt_token_list.append('%s/t' % text)

            else:
                tgt_token_list.append(token)

            idx += 1

        return tgt_token_list

    @staticmethod
    def get_word(word):
        ignores = ['Pp_ZhY_q#', 'Pp_zai_q#']

        if 'Pp_' in word:
            for ig in ignores:
                word = word.replace(ig, '')

        return word

    @staticmethod
    def get_tag(tag):
        ignores = ['s!B#Pp_ZhY_h', '3#Pp_ZhY_h', 't#Pp_zai_h', 't#Pp_ZhY_h', 's#Pp_ZhY_h',
                   '!B_td', '!B', '!1', 'ys-pis', 'h_jsh_db', ']']
        tag_dict = {'Ag': 'a', 'a': 'a', 'ad': 'ad', 'an': 'an',
                    'Bg': 'b', 'b': 'b', 'c': 'c', 'Dg': 'd', 'd': 'd', 'dc': 'd', 'df': 'd',
                    'e': 'e', 'f': 'f', 'h': 'h',
                    'i': 'i', 'ia': 'a', 'ib': 'b', 'id': 'd', 'im': 'm', 'in': 'n', 'iv': 'v',
                    'j': 'j', 'ja': 'a', 'jb': 'b', 'jd': 'd', 'jm': 'm', 'jn': 'n', 'jv': 'v',
                    'k': 'k', 'kn': 'k', 'kv': 'k',
                    'l': 'l', 'la': 'a', 'lb': 'b', 'ld': 'd', 'lm': 'm', 'ln': 'n', 'lv': 'v',
                    'Mg': 'm', 'm': 'm', 'mq': 'm',
                    'Ng': 'n', 'n': 'n', 'nr': 'nr', 'ns': 'ns', 'nt': 'nt', 'nx': 'nx', 'nz': 'nz',
                    'o': 'o', 'p': 'p', 'Qg': 'q', 'q': 'q', 'qb': 'q', 'qc': 'q', 'qd': 'q', 'qe': 'q',
                    'qj': 'q', 'ql': 'q', 'qr': 'q', 'qt': 'q', 'qv': 'q', 'qz': 'q',
                    'Rg': 'r', 'r': 'r', 'rr': 'r', 'ry': 'r', 'ryw': 'r', 'rz': 'r', 'rzw': 'r',
                    's': 's', 'Tg': 't', 't': 't', 'tt': 'tt',
                    'Ug': 'u', 'u': 'u', 'ud': 'u', 'ue': 'u', 'ui': 'u', 'ul': 'u', 'uo': 'u', 'us': 'u', 'uz': 'u',
                    'Vg': 'v', 'v': 'v', 'vd': 'vd', 'vi': 'v', 'vl': 'v', 'vn': 'vn', 'vq': 'v', 'vu': 'v', 'vx': 'v',
                    'vt': 'v',
                    'w': 'w', 'wd': 'w', 'wf': 'w', 'wj': 'w', 'wk': 'w', 'wky': 'w', 'wkz': 'w', 'wm': 'w', 'wp': 'w',
                    'ws': 'w', 'wt': 'w', 'wu': 'w', 'ww': 'w', 'wy': 'w', 'wyy': 'w', 'wyz': 'w', 'y': 'y', 'z': 'z'}

        if ('_' in tag) or ('-' in tag) or ('!' in tag) or (']' in tag):
            for ig in ignores:
                tag = tag.replace(ig, '')

        tgt = tag_dict[tag]
        tgt = tgt.upper()
        return tgt

    @staticmethod
    def get_data_split(data_size):
        train_idx = list(range(data_size))

        dev_idx = random.sample(train_idx, 1500)
        for idx in train_idx:
            if idx in dev_idx:
                train_idx.remove(idx)

        test_idx = random.sample(train_idx, 2000)
        for idx in train_idx:
            if idx in test_idx:
                train_idx.remove(idx)

        return train_idx, dev_idx, test_idx


class UDDataset(BaseDataset):
    def __init__(self):
        pass

    @staticmethod
    def transform(src_path, tgt_path):
        train_sents_word, tags_word, train_sents_char, tags_char = UDDataset.parse_data(
            '%s/zh_gsd-ud-train.conllu' % src_path)
        dev_sents_word, _, dev_sents_char, _ = UDDataset.parse_data(
            '%s/zh_gsd-ud-dev.conllu' % src_path)
        test_sents_word, _, test_sents_char, _ = UDDataset.parse_data(
            '%s/zh_gsd-ud-test.conllu' % src_path)

        tags_word = list(tags_word)
        tags_word.sort()
        tags_char = list(tags_char)
        tags_char.sort()

        logging.info('num of train sents: %s, dev setns: %s, test sents: %s' % (
            len(train_sents_word), len(dev_sents_word), len(test_sents_word)))
        logging.info('num of word tags: %s, char tags: %s' % (len(tags_word), len(tags_char)))

        BaseDataset.save_data(tgt_path, tags_word, train_sents_word, dev_sents_word, test_sents_word, 'word')
        BaseDataset.save_data(tgt_path, tags_char, train_sents_char, dev_sents_char, test_sents_char, 'char')

    @staticmethod
    def parse_data(src_path):
        from opencc import OpenCC
        convertor = OpenCC('t2s')

        sents_word = []
        tags_word = set()

        sents_char = []
        tags_char = set()

        item_word = []
        item_char = []

        with codecs.open(src_path, 'r', 'utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    sents_word.append(item_word)
                    item_word = []

                    sents_char.append(item_char)
                    item_char = []
                    continue

                if line.startswith('#'):
                    continue

                line = BaseDataset.dbc_to_sbc(line)
                line = convertor.convert(line)
                line = line.split()
                word = line[1]
                pos_tag = line[3]

                item_word.append([word, pos_tag])
                tags_word.add(pos_tag)

                chs = [ch for ch in word]
                if len(chs) == 1:
                    seg_t = ['S']
                    pos_t = ['S-%s' % pos_tag]
                else:
                    seg_t = ['B'] + ['M'] * (len(chs) - 2) + ['E']
                    pos_t = ['B-%s' % pos_tag] + ['M-%s' % pos_tag] * (len(chs) - 2) + ['E-%s' % pos_tag]

                assert len(chs) == len(seg_t) == len(pos_t)
                for i in range(len(chs)):
                    item_char.append([chs[i], seg_t[i], pos_t[i]])
                    tags_char.add(pos_t[i])

        return sents_word, tags_word, sents_char, tags_char


class Tokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    def convert_tokens_to_ids(self, tokens, unk_token=TOKEN_UNK):
        ids = []
        for token in tokens:
            if isinstance(token, str):
                ids.append(self.token_to_id.get(token, self.token_to_id[unk_token]))
            else:
                ids.append([self.token_to_id.get(t, self.token_to_id[unk_token]) for t in token])
        return ids

    def convert_ids_to_tokens(self, ids, max_sent_len):
        tokens = [self.id_to_token[i] for i in ids]
        if max_sent_len > 0:
            tokens = tokens[:max_sent_len]
        return tokens


def load_tag_dict(file_path):
    tag_to_id = {}
    with codecs.open(file_path, 'r', 'utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                tag_to_id[line] = len(tag_to_id)

    id_to_tag = {v: k for k, v in tag_to_id.items()}
    return tag_to_id, id_to_tag


def convert_ids_to_tags(id_to_tag, ids, max_sent_len):
    tags = [id_to_tag[i] for i in ids]
    if max_sent_len > 0:
        tags = tags[:max_sent_len]
    return tags


def load_pretrain_embedding(filepath, has_meta=False,
                            add_pad=False, pad_token=TOKEN_PAD, add_unk=False, unk_token=TOKEN_UNK, debug=False):
    with codecs.open(filepath, 'r', 'utf-8', errors='ignore') as fin:
        token_to_id = {}
        embed = []

        if has_meta:
            meta_info = fin.readline().strip().split()

        first_line = fin.readline().strip().split()
        embed_size = len(first_line) - 1

        if add_pad:
            token_to_id[pad_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        if add_unk:
            token_to_id[unk_token] = len(token_to_id)
            embed.append([0.] * embed_size)

        token_to_id[first_line[0]] = len(token_to_id)
        embed.append([float(x) for x in first_line[1:]])

        for line in fin:
            line = line.split()

            if len(line) != embed_size + 1:
                continue
            if line[0] in token_to_id:
                continue

            token_to_id[line[0]] = len(token_to_id)
            embed.append([float(x) for x in line[1:]])

            if debug:
                if len(embed) >= 1000:
                    break

    return token_to_id, embed

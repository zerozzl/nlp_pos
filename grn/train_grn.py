import os
import logging
from argparse import ArgumentParser
import codecs
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from grn.model import CharPOS, WordPOS
from utils.utils import setup_seed
from utils.dataloader import load_pretrain_embedding, load_tag_dict, convert_ids_to_tags, Tokenizer, SEG_TAG_TO_ID, \
    BaseDataset
from utils import modelloader, evaluator


def get_dataset(args, tag_to_id, char_tokenizer, bigram_tokenizer, word_tokenizer):
    train_dataset = BaseDataset('%s/%s/train_%s.txt' % (args.data_path, args.task, args.use_char_or_word),
                                args.use_char_or_word, args.max_sent_len, args.max_word_len,
                                do_pad=True, do_to_id=True, tag_to_id=tag_to_id, char_tokenizer=char_tokenizer,
                                add_bigram_feature=args.use_bigram_embed, bigram_tokenizer=bigram_tokenizer,
                                word_tokenizer=word_tokenizer, add_char_feature=args.use_char_embed,
                                debug=args.debug)
    dev_dataset = BaseDataset('%s/%s/dev_%s.txt' % (args.data_path, args.task, args.use_char_or_word),
                              args.use_char_or_word, args.max_sent_len, args.max_word_len,
                              do_pad=True, do_to_id=True, tag_to_id=tag_to_id, char_tokenizer=char_tokenizer,
                              add_bigram_feature=args.use_bigram_embed, bigram_tokenizer=bigram_tokenizer,
                              word_tokenizer=word_tokenizer, add_char_feature=args.use_char_embed,
                              debug=args.debug)
    test_dataset = BaseDataset('%s/%s/test_%s.txt' % (args.data_path, args.task, args.use_char_or_word),
                               args.use_char_or_word, args.max_sent_len, args.max_word_len,
                               do_pad=True, do_to_id=True, tag_to_id=tag_to_id, char_tokenizer=char_tokenizer,
                               add_bigram_feature=args.use_bigram_embed, bigram_tokenizer=bigram_tokenizer,
                               word_tokenizer=word_tokenizer, add_char_feature=args.use_char_embed,
                               debug=args.debug)

    return train_dataset, dev_dataset, test_dataset


def data_collate_fn(data):
    data = np.array(data)
    data_shape = np.shape(data)

    if data_shape[1] == 6:  # char
        sents = torch.LongTensor(np.array(data[:, 0].tolist()))
        bigram = torch.LongTensor(np.array(data[:, 1].tolist()))
        seg_tags = torch.LongTensor(np.array(data[:, 2].tolist()))
        pos_tag = torch.LongTensor(np.array(data[:, 3].tolist()))
        masks = torch.BoolTensor(np.array(data[:, 4].tolist()))
        sents_len = data[:, 5].tolist()

        return sents, bigram, seg_tags, pos_tag, masks, sents_len
    if data_shape[1] == 5:
        sents = torch.LongTensor(np.array(data[:, 0].tolist()))
        chars = torch.LongTensor(np.array(data[:, 1].tolist()))
        pos_tag = torch.LongTensor(np.array(data[:, 2].tolist()))
        masks = torch.BoolTensor(np.array(data[:, 3].tolist()))
        sents_len = data[:, 4].tolist()

        return sents, chars, pos_tag, masks, sents_len


def train(args, dataset, dataloader, model, optimizer, lr_scheduler):
    model.train()
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        optimizer.zero_grad()

        if args.use_char_or_word == 'char':
            sents, bigram, seg_tags, pos_tag, masks, _ = data

            sents = sents.cpu() if args.use_cpu else sents.cuda()
            bigram = bigram.cpu() if args.use_cpu else bigram.cuda()
            seg_tags = seg_tags.cpu() if args.use_cpu else seg_tags.cuda()
            pos_tag = pos_tag.cpu() if args.use_cpu else pos_tag.cuda()
            masks = masks.cpu() if args.use_cpu else masks.cuda()

            loss = model(sents, masks, bigram, seg_tags, decode=False, tags=pos_tag)

        if args.use_char_or_word == 'word':
            sents, chars, pos_tag, masks, _ = data

            sents = sents.cpu() if args.use_cpu else sents.cuda()
            chars = chars.cpu() if args.use_cpu else chars.cuda()
            pos_tag = pos_tag.cpu() if args.use_cpu else pos_tag.cuda()
            masks = masks.cpu() if args.use_cpu else masks.cuda()

            loss = model(sents, masks, chars, decode=False, tags=pos_tag)

        loss = loss.mean()
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    loss_sum = loss_sum / len(dataset)
    return loss_sum


def evaluate(args, dataloader, id_to_tag, model):
    gold_answers = []
    pred_answers = []

    model.eval()

    if args.multi_gpu:
        model = model.module

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            if args.use_char_or_word == 'char':
                sents, bigram, seg_tags, pos_tag, masks, sents_len = data

                sents = sents.cpu() if args.use_cpu else sents.cuda()
                bigram = bigram.cpu() if args.use_cpu else bigram.cuda()
                seg_tags = seg_tags.cpu() if args.use_cpu else seg_tags.cuda()
                pos_tag = pos_tag.cpu() if args.use_cpu else pos_tag.cuda()
                masks = masks.cpu() if args.use_cpu else masks.cuda()

                preds = model(sents, masks, bigram, seg_tags)
                tags = pos_tag.cpu().numpy()

                tags_gold = [convert_ids_to_tags(id_to_tag, tags[i], sents_len[i]) for i in range(len(sents_len))]
                tags_pred = [convert_ids_to_tags(id_to_tag, preds[i], sents_len[i]) for i in range(len(sents_len))]

            if args.use_char_or_word == 'word':
                sents, chars, pos_tag, masks, sents_len = data

                sents = sents.cpu() if args.use_cpu else sents.cuda()
                chars = chars.cpu() if args.use_cpu else chars.cuda()
                pos_tag = pos_tag.cpu() if args.use_cpu else pos_tag.cuda()
                masks = masks.cpu() if args.use_cpu else masks.cuda()

                preds = model(sents, masks, chars)
                tags = pos_tag.cpu().numpy()

                tags_gold = []
                tags_pred = []
                for i in range(len(sents_len)):
                    tags_gold.extend([tag for tag in tags[i][:sents_len[i]]])
                    tags_pred.extend([tag for tag in preds[i][:sents_len[i]]])

            gold_answers.extend(tags_gold)
            pred_answers.extend(tags_pred)

    _, pre, rec, f1 = evaluator.evaluate(gold_answers, pred_answers, args.use_char_or_word)
    return pre, rec, f1


def main(args):
    if args.debug:
        args.batch_size = 3

    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

    if args.multi_gpu:
        logging.info("run on multi GPU")
        torch.distributed.init_process_group(backend="nccl")

    setup_seed(0)

    output_path = '%s/%s/%s' % (args.output_path, args.task, args.use_char_or_word)
    if args.use_seg_embed:
        output_path += '_seg'
    if args.use_bigram_embed:
        output_path += '_bigram'
    if args.use_char_embed:
        output_path += '_char'
    if args.use_crf:
        output_path += '_crf'
    if args.embed_freeze:
        output_path += '_embfix'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.info("loading pretrained embedding")

    char_to_id = {}
    pretrain_char_embed = []
    char_tokenizer = None
    if (args.use_char_or_word == 'char') or args.use_char_embed:
        char_to_id, pretrain_char_embed = load_pretrain_embedding(args.pretrained_char_emb_path,
                                                                  add_pad=True, add_unk=True, debug=args.debug)
        char_tokenizer = Tokenizer(char_to_id)

    bigram_to_id = {}
    pretrain_bigram_embed = []
    bigram_tokenizer = None
    if args.use_bigram_embed:
        bigram_to_id, pretrain_bigram_embed = load_pretrain_embedding(args.pretrained_bigram_emb_path,
                                                                      add_pad=True, add_unk=True, debug=args.debug)
        bigram_tokenizer = Tokenizer(bigram_to_id)

    word_to_id = {}
    pretrain_word_embed = []
    word_tokenizer = None
    if args.use_char_or_word == 'word':
        word_to_id, pretrain_word_embed = load_pretrain_embedding(args.pretrained_word_emb_path, has_meta=True,
                                                                  add_pad=True, add_unk=True, debug=args.debug)
        word_tokenizer = Tokenizer(word_to_id)

    tag_to_id, id_to_tag = load_tag_dict('%s/%s/tags_%s.txt' % (args.data_path, args.task, args.use_char_or_word))

    logging.info("loading dataset")
    train_dataset, dev_dataset, test_dataset = get_dataset(
        args, tag_to_id, char_tokenizer, bigram_tokenizer, word_tokenizer)

    if args.multi_gpu:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      sampler=DistributedSampler(train_dataset))
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                    sampler=DistributedSampler(dev_dataset))
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     sampler=DistributedSampler(test_dataset))
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                      shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                    shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_collate_fn,
                                     shuffle=False)

    best_f1 = 0
    epoch = 0

    if args.pretrained_model_path is not None:
        logging.info("loading pretrained model")
        model, optimizer, epoch, best_f1 = modelloader.load(args.pretrained_model_path)
        model = model.cpu() if args.use_cpu else model.cuda()
    else:
        logging.info("creating model")
        if args.use_char_or_word == 'char':
            model = CharPOS(len(tag_to_id), len(char_to_id), args.char_embed_size, args.input_dropout_rate,
                            args.hidden_dim, args.hidden_dropout_rate,
                            args.use_seg_embed, len(SEG_TAG_TO_ID), args.seg_embed_size,
                            args.use_bigram_embed, len(bigram_to_id), args.bigram_embed_size,
                            args.use_crf, args.embed_freeze)
        elif args.use_char_or_word == 'word':
            model = WordPOS(len(tag_to_id), len(word_to_id), args.word_embed_size, args.input_dropout_rate,
                            args.char_cnn_kernel_size, args.hidden_dim, args.hidden_dropout_rate,
                            args.use_char_embed, len(char_to_id), args.char_embed_size,
                            args.use_crf, args.embed_freeze)

        model = model.cpu() if args.use_cpu else model.cuda()

        if (args.use_char_or_word == 'char') or args.use_char_embed:
            model.init_char_embedding(np.array(pretrain_char_embed))

        if args.use_bigram_embed:
            model.init_bigram_embedding(np.array(pretrain_bigram_embed))

        if args.use_char_or_word == 'word':
            model.init_word_embedding(np.array(pretrain_word_embed))

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.learning_rate_momentum)

    if args.multi_gpu:
        model = DistributedDataParallel(model, find_unused_parameters=True)

    num_train_steps = int(len(train_dataset) / args.batch_size * args.epoch_size)
    num_warmup_steps = int(num_train_steps * args.lr_warmup_proportion)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_warmup_steps, gamma=args.lr_decay_gamma)

    logging.info("begin training")
    while epoch < args.epoch_size:
        epoch += 1

        train_loss = train(args, train_dataset, train_dataloader, model, optimizer, lr_scheduler)

        dev_pre, dev_rec, dev_f1 = evaluate(args, dev_dataloader, id_to_tag, model)

        logging.info('epoch[%s/%s], train_loss: %s' % (
            epoch, args.epoch_size, train_loss))
        logging.info('epoch[%s/%s], val_precision: %s, val_recall: %s, val_f1: %s' % (
            epoch, args.epoch_size, dev_pre, dev_rec, dev_f1))

        modelloader.save(output_path, 'last.pth', model, optimizer, epoch, dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1

            test_pre, test_rec, test_f1 = evaluate(args, test_dataloader, id_to_tag, model)
            logging.info('epoch[%s/%s], test_precision: %s, test_recall: %s, test_f1: %s' % (
                epoch, args.epoch_size, test_pre, test_rec, test_f1))

            modelloader.save(output_path, 'best.pth', model, optimizer, epoch, best_f1)

            with codecs.open('%s/best_score.txt' % output_path, 'w', 'utf-8') as fout:
                fout.write('Dev precision: %s, recall: %s, f1: %s\n' % (dev_pre, dev_rec, dev_f1))
                fout.write('Test precision: %s, recall: %s, f1: %s\n' % (test_pre, test_rec, test_f1))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--task', dest='task',
                        default='ctb8')
    parser.add_argument('--data_path', dest='data_path',
                        default='../data/datasets/')
    parser.add_argument('--pretrained_char_emb_path', dest='pretrained_char_emb_path',
                        default='../data/embeddings/gigaword_chn.all.a2b.uni.ite50.vec')
    parser.add_argument('--pretrained_bigram_emb_path', dest='pretrained_bigram_emb_path',
                        default='../data/embeddings/gigaword_chn.all.a2b.bi.ite50.vec')
    parser.add_argument('--pretrained_word_emb_path', dest='pretrained_word_emb_path',
                        default='../data/embeddings/news_tensite.pku.words.w2v50')
    parser.add_argument('--pretrained_model_path', dest='pretrained_model_path',
                        default=None)
    parser.add_argument('--output_path', dest='output_path',
                        default='../runtime/grn/')
    parser.add_argument('--use_char_or_word', dest='use_char_or_word',
                        default='char')
    parser.add_argument('--use_crf', dest='use_crf', type=bool,
                        default=False)
    parser.add_argument('--use_seg_embed', dest='use_seg_embed', type=bool,
                        default=False)
    parser.add_argument('--use_bigram_embed', dest='use_bigram_embed', type=bool,
                        default=False)
    parser.add_argument('--use_char_embed', dest='use_char_embed', type=bool,
                        default=False)
    parser.add_argument('--embed_freeze', dest='embed_freeze', type=bool,
                        default=False)
    parser.add_argument('--max_sent_len', dest='max_sent_len', type=int,
                        default=150)
    parser.add_argument('--max_word_len', dest='max_word_len', type=int,
                        default=8)
    parser.add_argument('--char_embed_size', dest='char_embed_size', type=int,
                        default=50)
    parser.add_argument('--bigram_embed_size', dest='bigram_embed_size', type=int,
                        default=50)
    parser.add_argument('--word_embed_size', dest='word_embed_size', type=int,
                        default=50)
    parser.add_argument('--seg_embed_size', dest='seg_embed_size', type=int,
                        default=5)
    parser.add_argument('--input_dropout_rate', dest='input_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--hidden_dropout_rate', dest='hidden_dropout_rate', type=float,
                        default=0.5)
    parser.add_argument('--char_cnn_kernel_size', dest='char_cnn_kernel_size', type=int,
                        default=3)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int,
                        default=400)
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=16)
    parser.add_argument('--epoch_size', dest='epoch_size', type=int,
                        default=200)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                        help='0.5 without crf, 0.0005 with crf',
                        default=0.5)
    parser.add_argument('--learning_rate_momentum', dest='learning_rate_momentum', type=float,
                        default=0.9)
    parser.add_argument('--lr_warmup_proportion', dest='lr_warmup_proportion', type=float,
                        default=0.1)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float,
                        default=0.9)
    parser.add_argument('--use_cpu', dest='use_cpu', type=bool,
                        default=False)
    parser.add_argument('--multi_gpu', dest='multi_gpu', type=bool, help='run with: -m torch.distributed.launch',
                        default=True)
    parser.add_argument('--local_rank', dest='local_rank', type=int,
                        default=0)
    parser.add_argument('--debug', dest='debug', type=bool,
                        default=False)

    args = parser.parse_args()

    main(args)

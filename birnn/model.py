import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from torchcrf import CRF


class CharPOS(nn.Module):
    def __init__(self, num_tags, char_vocab_size, char_embed_size, input_dropout_rate,
                 hidden_layers, hidden_dim, hidden_dropout_rate,
                 use_seg_embed, seg_vocab_size, seg_embed_size,
                 use_bigram_embed, bigram_vocab_size, bigram_embed_size,
                 use_crf, embed_freeze):
        super(CharPOS, self).__init__()
        self.use_seg_embed = use_seg_embed
        self.use_bigram_embed = use_bigram_embed
        self.use_crf = use_crf

        embed_size = char_embed_size
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_size)

        if use_seg_embed:
            embed_size += seg_embed_size
            self.seg_embedding = nn.Embedding(seg_vocab_size, seg_embed_size)

        if use_bigram_embed:
            embed_size += bigram_embed_size * 2
            self.bigram_embedding = nn.Embedding(bigram_vocab_size, bigram_embed_size)

        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_dim,
                          num_layers=hidden_layers,
                          batch_first=True,
                          dropout=hidden_dropout_rate,
                          bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        if embed_freeze:
            for param in self.char_embedding.parameters():
                param.requires_grad = False
            if use_bigram_embed:
                for param in self.bigram_embedding.parameters():
                    param.requires_grad = False

    def init_char_embedding(self, pretrained_embeddings):
        self.char_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def init_bigram_embedding(self, pretrained_embeddings):
        self.bigram_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, masks, bigram=None, seg_tags=None, decode=True, tags=None):
        tokens_len = torch.sum(masks, 1).cpu().numpy()

        out = self.char_embedding(tokens)

        if self.use_bigram_embed:
            bi_emb = torch.cat([self.bigram_embedding(bigram[:, :, i]) for i in range(bigram.size()[2])], dim=2)
            out = torch.cat((out, bi_emb), dim=2)
        if self.use_seg_embed:
            s_emb = self.seg_embedding(seg_tags)
            out = torch.cat((out, s_emb), dim=2)

        out = self.in_dropout(out)

        out = rnn.pack_padded_sequence(out, tokens_len, batch_first=True)
        out, _ = self.rnn(out)
        out, _ = rnn.pad_packed_sequence(out, batch_first=True)

        out = self.linear(out)

        if decode:
            if self.use_crf:
                pred = self.crf.decode(out, masks)
            else:
                out = F.softmax(out, dim=2)
                out = torch.argmax(out, dim=2)
                pred = out.cpu().numpy()
            return pred
        else:
            out_shape = out.size()
            masks = masks[:, :out_shape[1]]
            tags = tags[:, :out_shape[1]]

            if self.use_crf:
                loss = -self.crf(out, tags, masks)
            else:
                loss = self.ce_loss(out.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                    tags.reshape(out_shape[0] * out_shape[1]))
            return loss


class WordPOS(nn.Module):
    def __init__(self, num_tags, word_vocab_size, word_embed_size, input_dropout_rate,
                 hidden_layers, hidden_dim, hidden_dropout_rate,
                 use_crf, embed_freeze):
        super(WordPOS, self).__init__()
        self.use_crf = use_crf

        embed_size = word_embed_size
        self.word_embedding = nn.Embedding(word_vocab_size, word_embed_size)

        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=hidden_dim,
                          num_layers=hidden_layers,
                          batch_first=True,
                          dropout=hidden_dropout_rate,
                          bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, num_tags)

        self.in_dropout = nn.Dropout(input_dropout_rate)

        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        if embed_freeze:
            for param in self.word_embedding.parameters():
                param.requires_grad = False

    def init_word_embedding(self, pretrained_embeddings):
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    def forward(self, tokens, masks, decode=True, tags=None):
        tokens_len = torch.sum(masks, 1).cpu().numpy()

        out = self.word_embedding(tokens)
        out = self.in_dropout(out)

        out = rnn.pack_padded_sequence(out, tokens_len, batch_first=True)
        out, _ = self.rnn(out)
        out, _ = rnn.pad_packed_sequence(out, batch_first=True)

        out = self.linear(out)

        if decode:
            if self.use_crf:
                pred = self.crf.decode(out, masks)
            else:
                out = F.softmax(out, dim=2)
                out = torch.argmax(out, dim=2)
                pred = out.cpu().numpy()
            return pred
        else:
            out_shape = out.size()
            masks = masks[:, :out_shape[1]]
            tags = tags[:, :out_shape[1]]

            if self.use_crf:
                loss = -self.crf(out, tags, masks)
            else:
                loss = self.ce_loss(out.reshape(out_shape[0] * out_shape[1], out_shape[2]),
                                    tags.reshape(out_shape[0] * out_shape[1]))
            return loss

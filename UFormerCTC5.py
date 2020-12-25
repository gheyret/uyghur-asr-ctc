import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

import numpy as np
import math

from BaseModel import BaseModel
from data import melfuture
from uyghur import uyghur_latin

class UFormerCTC5(BaseModel):
    def __init__(self, num_features_input, load_best=False):
        super(UFormerCTC5, self).__init__('UFormerCTC5')
        num_layers = 5      #'Number of layers'
        num_heads  = 8      #'Number of heads'
        dim_model  = 512    #'Model dimension'
        dim_key    = 64     #'Key dimension'
        dim_value  = 64     #'Value dimension'
        dim_inner  = 1024   #'Inner dimension'
        dim_emb    = 512    #'Embedding dimension'
        src_max_len = 2500  #'Source max length'
        tgt_max_len = 1000  #'Target max length'
        dropout    = 0.1
        emb_trg_sharing = False
        #self.future_len = num_features_input
        self.flayer = UDS2W2L8(num_features_input)
        self.encoder  = Encoder(num_layers, num_heads=num_heads, dim_model=dim_model, dim_key=dim_key, dim_value=dim_value, dim_inner=dim_inner, src_max_length=src_max_len, dropout=dropout)
        self.decoder  = Decoder(num_layers=num_layers, num_heads=num_heads, dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout, emb_trg_sharing=emb_trg_sharing)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.ctcOut = None
        self.ctcLen = None

        self.checkpoint = "results/" + self.ModelName
        self._load(load_best)
        #self._loadfrom("results/UFormerCTC1_last.pth")
        #self.flayer.load()

        print("          Model Name:", self.ModelName)
        print(f'The model has {self.parameters_count(self):,} trainable parameters')
        print(f'  Future  has {self.parameters_count(self.flayer):,} trainable parameters')
        print(f'  Encoder has {self.parameters_count(self.encoder):,} trainable parameters')
        print(f'  Decoder has {self.parameters_count(self.decoder):,} trainable parameters')


    def forward(self, padded_input, input_lengths, padded_target):
        padded_input,self.ctcOut, self.ctcLen = self.flayer(padded_input,input_lengths)        
        #input must be #B x T x F format
        encoder_padded_outputs, _ = self.encoder(padded_input, self.ctcLen) # BxTxH or #B x T x F
        seq_in_pad, gold = self.preprocess(padded_target)
        pred = self.decoder(seq_in_pad, encoder_padded_outputs, self.ctcLen)
        return pred, gold

    def greedydecode(self, pred, len=0):
        _, pred = torch.topk(pred, 1, dim=2)
        preds = pred.squeeze(2)
        strs_pred = [uyghur_latin.decode(pred_id) for pred_id in preds]
        return strs_pred
    
    def predict(self,wavfile, device):
        self.eval()
        spec  = melfuture(wavfile).unsqueeze(0).to(device)
        spec_len = torch.tensor([spec.shape[2]], dtype=torch.int)
        padded_input,self.ctcOut, self.ctcLen = self.flayer(spec,spec_len)
        encoder_padded_outputs, _ = self.encoder(padded_input, self.ctcLen) # BxTxH or #B x T x F
        strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)        
        return strs_hyps


class ResB(nn.Module):
    def __init__(self, num_filters, kernel, pad, d = 0.4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size = kernel, stride = 1 , padding=pad, bias=False),
            nn.BatchNorm1d(num_filters)
            )

        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm1d(num_filters)
        self.drop =nn.Dropout(d)

    def forward(self, x):
        identity = x
        out  = self.conv(x)
        out += identity
        out  = self.bn(out)
        out  = self.relu(out)
        out  = self.drop(out)
        return out


class UDS2W2L8(nn.Module):
    def __init__(self, num_features_input):
        super(UDS2W2L8, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5),bias=False),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )
        self.lstm1 = nn.GRU(1024, 256, num_layers=1 , batch_first=True, bidirectional=True)
        self.cnn1  = nn.Sequential(
            nn.Conv1d(256, 256, 11, 2, 5,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2),
            ResB(256,11,5,0.2)
        )
        self.lstm2 = nn.GRU(256, 384, num_layers=1 , batch_first=True, bidirectional=True)
        self.cnn2 = nn.Sequential(
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            ResB(384,13,6,0.2),
            nn.Conv1d(384, 512, 17, 1,8,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResB(512,17,8,0.3),
            ResB(512,17,8,0.3),
            nn.Conv1d(512, 512, 1, 1,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResB(512,1,0,0.0),
        )
        self.outlayer = nn.Conv1d(512, uyghur_latin.vocab_size, 1, 1)
        self.softMax = nn.LogSoftmax(dim=1)

    def forward(self, x, lengths):
        out_lens = lengths//2

        x.unsqueeze_(1)
        out = self.conv(x)

        b, c, h, w = out.size()
        out = out.view(b, c*h, w).contiguous() #.permute(0,2,1)

        out = out.permute(0,2,1)
        #out = nn.utils.rnn.pack_padded_sequence(out, out_lens, batch_first=True)
        out, _ = self.lstm1(out)        
        #out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = (out[:, :, :self.lstm1.hidden_size] + out[:, :, self.lstm1.hidden_size:]).contiguous()
        out = self.cnn1(out.permute(0,2,1))

        out_lens = out_lens//2
        out = out.permute(0,2,1)
        #out = nn.utils.rnn.pack_padded_sequence(out, out_lens, batch_first=True)
        out,_ = self.lstm2(out)
        #out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = (out[:, :, :self.lstm2.hidden_size] + out[:, :, self.lstm2.hidden_size:]).contiguous()
        out = self.cnn2(out.permute(0,2,1))
        outctc = self.softMax(self.outlayer(out))
        return out.contiguous().permute(0,2,1), outctc, out_lens

    def load(self):
        pack = torch.load('results/UDS2W2L8_last.pth', map_location='cpu')
        sdict = pack['st_dict']        
        news_dict  = self.state_dict()
        filtered_dict = {k: v for k, v in sdict.items() if k in news_dict and v.size() == news_dict[k].size()}
        news_dict.update(filtered_dict)
        self.load_state_dict(news_dict)


class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_inner, dropout=0.1, src_max_length=2500):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner

        self.src_max_length = src_max_length

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.positional_encoding = PositionalEncoding(dim_model, src_max_length)

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        return:
            output: B x T x H
        """
        encoder_self_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # B x T x D
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len)  # B x T x T
        pos =  self.positional_encoding(padded_input)
        encoder_output = padded_input + pos

        for layer in self.layers:
            encoder_output, self_attn = layer(encoder_output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]

        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, self_attn


class Decoder(nn.Module):
    """
    Decoder Layer Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_emb, dim_model, dim_inner, dim_key, dim_value, dropout=0.1, trg_max_length=1000, emb_trg_sharing=False):
        super(Decoder, self).__init__()
        self.num_trg_vocab = uyghur_latin.vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_emb = dim_emb
        self.dim_model = dim_model
        self.dim_inner = dim_inner
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.dropout_rate = dropout
        self.emb_trg_sharing = emb_trg_sharing

        self.trg_max_length = trg_max_length

        self.trg_embedding = nn.Embedding(self.num_trg_vocab, dim_emb, padding_idx=uyghur_latin.pad_idx)
        self.positional_encoding = PositionalEncoding(dim_model, trg_max_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, dim_inner, num_heads,dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(dim_model, self.num_trg_vocab, bias=False)
        nn.init.xavier_normal_(self.output_linear.weight)

        if emb_trg_sharing:
            self.output_linear.weight = self.trg_embedding.weight
            self.x_logit_scale = (dim_model ** -0.5)
        else:
            self.x_logit_scale = 1.0

    def forward(self, seq_in_pad, encoder_padded_outputs, encoder_input_lengths):
        """
        args:
            padded_input: B x T
            encoder_padded_outputs: B x T x H
            encoder_input_lengths: B
        returns:
            pred: B x T x vocab
            gold: B x T
        """
        decoder_self_attn_list, decoder_encoder_attn_list = [], []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(seq_in_pad, pad_idx=uyghur_latin.pad_idx)
        self_attn_mask_subseq = get_subsequent_mask(seq_in_pad)
        self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=seq_in_pad, seq_q=seq_in_pad, pad_idx=uyghur_latin.pad_idx)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        output_length = seq_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs, encoder_input_lengths, output_length)

        decoder_output = self.dropout(self.trg_embedding(seq_in_pad) * self.x_logit_scale + self.positional_encoding(seq_in_pad))

        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_enc_attn = layer(decoder_output, encoder_padded_outputs, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask)

            decoder_self_attn_list += [decoder_self_attn]
            decoder_encoder_attn_list += [decoder_enc_attn]

        seq_logit = self.output_linear(decoder_output)

        return seq_logit

    def greedy_search(self, encoder_padded_outputs):
        """
        Greedy search, decode 1-best utterance
        args:
            encoder_padded_outputs: B x T x H
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        with torch.no_grad():
            device = encoder_padded_outputs.device
            max_seq_len = self.trg_max_length

            #ys = torch.ones(encoder_padded_outputs.size(0),1).fill_(uyghur_latin.sos_idx).long().to(device) # batch_size x 1
            max_seq_len = min(max_seq_len, encoder_padded_outputs.size(1))
            inps=[uyghur_latin.sos_idx]
            result = []
            for t in range(max_seq_len):
                ys = torch.LongTensor(inps).unsqueeze(0).to(device)
                non_pad_mask = torch.ones_like(ys).float().unsqueeze(-1) # batch_size x t x 1
                self_attn_mask = get_subsequent_mask(ys).gt(0) # batch_size x t x t

                decoder_output = self.dropout(self.trg_embedding(ys) * self.x_logit_scale + self.positional_encoding(ys))

                for layer in self.layers:
                    decoder_output, _, _ = layer(
                        decoder_output, encoder_padded_outputs,
                        non_pad_mask=non_pad_mask,
                        self_attn_mask=self_attn_mask,
                        dec_enc_attn_mask=None
                    )

                prob = self.output_linear(decoder_output) # batch_size x t x label_size
                _, next_word = torch.max(prob[:, -1], dim=1)
                next_word = next_word.item()
                result.append(next_word)
                if next_word == uyghur_latin.eos_idx: 
                    break

                inps.append(next_word)

        sent = uyghur_latin.decode(result)
        return sent

class DecoderLayer(nn.Module):
    """
    Decoder Transformer class
    """

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.encoder_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(
            dim_model, dim_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, non_pad_mask=None, self_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, decoder_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output, decoder_encoder_attn = self.encoder_attn(decoder_output, encoder_output, encoder_output, mask=dec_enc_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output = self.pos_ffn(decoder_output)
        decoder_output *= non_pad_mask

        return decoder_output, decoder_self_attn, decoder_encoder_attn        


""" 
Transformer common layers
"""

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """
    padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # B x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    """
    For masking out the padding part of key sequence.
    """
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1).byte()  # B x T_Q x T_K

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class PositionalEncoding(nn.Module):
    """
    Positional Encoding class
    """
    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term) # take the odd (jump by 2)
        pe[:, 1::2] = torch.cos(position * exp_term) # take the even (jump by 2)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        args:
            input: B x T x D
        output:
            tensor: B x T
        """
        return self.pe[:, :input.size(1)]



class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feedforward Layer class
    FFN(x) = max(0, xW1 + b1) W2+ b2
    """
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(dim_model, dim_ff)
        self.linear_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        """
        args:
            x: tensor
        output:
            y: tensor
        """
        residual = x
        output = self.dropout(self.linear_2(F.relu(self.linear_1(x))))
        output = self.layer_norm(output + residual)
        return output

class PositionwiseFeedForwardWithConv(nn.Module):
    """
    Position-wise Feedforward Layer Implementation with Convolution class
    """
    def __init__(self, dim_model, dim_hidden, dropout=0.1):
        super(PositionwiseFeedForwardWithConv, self).__init__()
        self.conv_1 = nn.Conv1d(dim_model, dim_hidden, 1)
        self.conv_2 = nn.Conv1d(dim_hidden, dim_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.conv_2(F.relu(self.conv_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dim_key, dim_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.value_linear = nn.Linear(dim_model, num_heads * dim_value)

        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.value_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_value)))

        self.attention = ScaledDotProductAttention(temperature=np.power(dim_key, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

        self.output_linear = nn.Linear(num_heads * dim_value, dim_model)
        nn.init.xavier_normal_(self.output_linear.weight)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: B x T_Q x H, key: B x T_K x H, value: B x T_V x H
        mask: B x T x T (attention mask)
        """
        batch_size, len_query, _ = query.size()
        batch_size, len_key, _ = key.size()
        batch_size, len_value, _ = value.size()

        residual = query

        query = self.query_linear(query).view(batch_size, len_query, self.num_heads, self.dim_key) # B x T_Q x num_heads x H_K
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key) # B x T_K x num_heads x H_K
        value = self.value_linear(value).view(batch_size, len_value, self.num_heads, self.dim_value) # B x T_V x num_heads x H_V

        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key) # (num_heads * B) x T_Q x H_K
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key) # (num_heads * B) x T_K x H_K
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, len_value, self.dim_value) # (num_heads * B) x T_V x H_V

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1) # (B * num_head) x T x T
        
        output, attn = self.attention(query, key, value, mask=mask)

        output = output.view(self.num_heads, batch_size, len_query, self.dim_value) # num_heads x B x T_Q x H_V
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_query, -1) # B x T_Q x (num_heads * H_V)

        output = self.dropout(self.output_linear(output)) # B x T_Q x H_O
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """

        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

if __name__ == "__main__":
    from data import melfuture, featurelen, uyghur_latin, SpeechDataset, _collate_fn
    device = 'cuda'
    model = UFormerCTC5(featurelen,uyghur_latin)
    model.to(device)
    #model.best_cer = 1.0
    model.save(0)

    txt = model.predict("test3.wav", device)
    print(txt)

    txt = model.predict("test4.wav", device)
    print(txt)

    train_dataset = SpeechDataset('uyghur_thuyg20_train_small.csv', augumentation=False)
    bbb = []
    bbb.append(train_dataset[0])
    bbb.append(train_dataset[3])
    bbb.append(train_dataset[4])
    inps, targs, in_lens,_,_ = _collate_fn(bbb)
    model.train()
    outs, trg = model(inps.to(device),in_lens, targs.to(device))
    print(outs.size())
    print(trg.size())

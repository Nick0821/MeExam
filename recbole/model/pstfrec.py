import logging
from collections import defaultdict
from copy import deepcopy
import logging
from collections import defaultdict
from copy import deepcopy
import random
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import KansformerEncoder, CLLayer,TransformerEncoder
from Params import args
from torch_scatter import scatter_sum, scatter_softmax

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform
### DST-GKANs
class PSTFRec(SequentialRecommender):

    def __init__(self, config, dataset, check_index):
        super(PSTFRec, self).__init__(config, dataset)

        self.config = config
        # load parameters info
        self.Sequential_model = args.Sequential_model
        self.device = config["device"]
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.n_user = args.n_user
        self.n_item_all = args.n_item_all
        self.n_age = args.n_age
        self.n_gender = args.n_gender
        self.n_relation = args.relation_num
        self.mask_ratio = config['mask_ratio']
        self.check_index = check_index
        self.check_index_dict = {int(value): index for index, value in enumerate(self.check_index)}

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)
        self.cross_attention = CrossAttention(dim=64)
        self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)
        self.LinearFusion = nn.Linear((self.hidden_size + 1) * (self.hidden_size + 1) * (self.hidden_size + 1), self.hidden_size)
        self.LinearFusionUI = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        # define layers and loss
        self.relation_embedding = nn.Embedding(self.n_relation + 1, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)# mask.log token add 1
        self.user_embedding = nn.Embedding(self.n_user + 1, self.hidden_size, padding_idx=0)
        self.age_embedding = nn.Embedding(self.n_age, self.hidden_size, padding_idx=0)
        self.gender_embedding = nn.Embedding(self.n_gender, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        if self.Sequential_model == 'Kansformer':
            self.trm_encoder = KansformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps
            )
        else:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps
            )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len
        return sequence

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask.log for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask.log
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_Embeds(self,item_seq):
        n_iEmbeds = torch.arange(0, self.n_items, device=item_seq.device)
        n_genderEmbeds = torch.arange(0, self.n_gender,device=item_seq.device)
        n_ageEmbeds = torch.arange(0, self.n_age ,device=item_seq.device)
        iEmbeds = self.item_embedding(n_iEmbeds)
        genderEmbeds = self.gender_embedding(n_genderEmbeds)
        ageEmbeds = self.age_embedding(n_ageEmbeds)
        eEmbeds = torch.cat([iEmbeds, genderEmbeds, ageEmbeds])

        uEmbeds = self.user_embedding(torch.arange(0, self.n_user + 1,device=item_seq.device))
        rEmbeds = self.relation_embedding(torch.arange(0, self.n_relation + 1,device=item_seq.device))
        return eEmbeds, uEmbeds, rEmbeds

    def get_last_item(self,item_seq,item_seq_len):
        last_nonzero_values = []
        n = 0
        for row in item_seq:
            last_nonzero_value = row[item_seq_len[n]-1].item()
            last_nonzero_values.append(last_nonzero_value)
            n += 1

        last_nonzero_values = torch.tensor(last_nonzero_values)
        return last_nonzero_values
    def graph_forward(self, item_seq, denoisedKG, user_id, age_id, gender_id, item_seq_len, mess_dropout=True):

        eEmbeds, uEmbeds, rEmbeds = self.get_Embeds(item_seq)
        user_res_emb, entity_res_emb = self.rgat.forward(uEmbeds, rEmbeds, eEmbeds, denoisedKG, mess_dropout)
        user_feature = user_res_emb[user_id]
        age_id = age_id + self.n_items + 1
        gender_id = gender_id + self.n_items - 1
        age_feature = entity_res_emb[age_id]
        gender_feature = entity_res_emb[gender_id]

        return user_feature, entity_res_emb, age_feature, gender_feature


    def sequnence_forward(self, item_seq, item_seq_len, item_res_emb):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = item_res_emb[item_seq]
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        seq_output_last = self.gather_indexes(output, item_seq_len - 1)

        return seq_output_last  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask.log 3 mask.log 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(
            0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def calculate_loss(self, interaction, denoisedKG):
        if self.config["graphcl_enable"]:
            return self.calculate_loss_graphcl(interaction)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        age_id = interaction.interaction['age_id']
        gender_id = interaction.interaction['gender_id']
        item_seq = interaction[self.ITEM_SEQ]
        last_item = interaction[self.ITEM_ID]
        user_id = interaction[self.USER_ID]
        user_feature, item_res_emb, age_feature, gender_feature = self.graph_forward(item_seq, denoisedKG, user_id, age_id, gender_id, item_seq_len)

        seq_output = self.sequnence_forward(item_seq, item_seq_len, item_res_emb)

        output = self.cross_attention.forward(seq_output.unsqueeze(1), user_feature.unsqueeze(1)).squeeze()

        last_item_encoder = []
        for i in last_item:
            last_item_encoder.append(self.check_index_dict[int(i)])

        last_item = torch.tensor(last_item_encoder, device=item_seq.device)

        test_item_emb = item_res_emb[self.check_index]

        logits = torch.mm(output, test_item_emb.transpose(0, 1))  # [mask_num, item_num]

        loss = self.loss_fct(logits, last_item)

        if torch.isnan(loss):
            print(loss)
            print(user_id)
            input()
        return loss

    def fast_predict(self, interaction, denoisedKG):
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        age_id = interaction.interaction['age_id']
        gender_id = interaction.interaction['gender_id']
        item_seq = interaction[self.ITEM_SEQ]
        user_id = interaction[self.USER_ID]
        test_item = interaction["item_id_with_negs"]
        user_feature, item_res_emb, age_feature, gender_feature = self.graph_forward(item_seq, denoisedKG, user_id,
                                                                                     age_id, gender_id, item_seq_len)
        seq_output = self.sequnence_forward(item_seq, item_seq_len, item_res_emb)
        output = self.cross_attention.forward(seq_output.unsqueeze(1), user_feature.unsqueeze(1)).squeeze()

        test_item_emb = item_res_emb[test_item]
        scores = torch.matmul(output.unsqueeze(1), test_item_emb.transpose(1, 2)).squeeze()
        return scores

class RGAT(nn.Module):
    def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
        super(RGAT, self).__init__()
        self.mess_dropout_rate = mess_dropout_rate
        self.W = nn.Parameter(init(torch.empty(size=(2 * latdim, latdim)), gain=nn.init.calculate_gain('relu')))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.n_hops = n_hops
        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def e_softmax(self, e, head):
        e_exp = e.exp()
        target_values = torch.zeros_like(e_exp)
        unique_indices = torch.unique(head)
        for idx in unique_indices:
            mask = (head == idx)
            target_values[mask] = torch.sum(e_exp.clone().detach()[mask])
        return e_exp / target_values

    def agg(self, entity_emb, user_embedding, relation_embedding, kg):
        edge_index, edge_type = kg
        head, tail = edge_index
        a_input = torch.cat([user_embedding[head], entity_emb[tail]], dim=-1)
        e_input = torch.multiply(torch.mm(a_input, self.W), relation_embedding[edge_type]).sum(-1)
        e = self.leakyrelu(e_input)

        e_user = self.e_softmax(e, head)
        agg_emb_user = entity_emb[tail] * e_user.view(-1, 1)
        agg_emb_user = scatter_sum(agg_emb_user, head, dim=0, dim_size=user_embedding.shape[0])
        agg_emb_user = agg_emb_user + user_embedding

        e_item = self.e_softmax(e, tail)
        agg_emb_item = user_embedding[head] * e_item.view(-1, 1)
        agg_emb_item = scatter_sum(agg_emb_item, tail, dim=0, dim_size=entity_emb.shape[0])
        agg_emb_item = agg_emb_item + entity_emb

        return agg_emb_user,agg_emb_item

    def forward(self, user_embedding, relation_embedding, entity_emb, kg, mess_dropout=True):
        entity_emb_all = entity_emb
        entity_user_all = user_embedding
        for _ in range(self.n_hops):
            entity_user, entity_emb = self.agg(entity_emb_all, entity_user_all, relation_embedding, kg)
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                entity_user = self.dropout(entity_user)
            entity_emb = F.normalize(entity_emb, eps = 1e-8)
            entity_user = F.normalize(entity_user, eps = 1e-8)
            entity_emb_all += entity_emb
            entity_user_all += entity_user

        return entity_user_all, entity_emb_all


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.qkv =nn.Linear(dim, dim *3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x, x1, seglen=1):
        B, N, C = x.shape
        B1,N1,C1 = x1.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv1 = self.qkv(x1).reshape(B1, N1, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v= qkv1[0], qkv[1], qkv[2]
        x = self.forward_cross_attention(q, k, v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_cross_attention(self, q, k, v):
        B, _, N, C= q.shape
        attn =(q @ k.transpose(-2, -1))* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x= attn @ v
        x=x.transpose(1,2).reshape(B,N,C*self.num_heads)
        return x
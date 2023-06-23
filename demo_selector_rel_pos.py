from abc import ABC

import ipdb
import torch
from torch import nn
import pytorch_lightning as pl

from pytorch_lightning.utilities.finite_checks import detect_nan_parameters


def check_nan(to_check_data, model):
    try:
        detect_nan_parameters(model)
    except:
        ipdb.set_trace()

    for data in to_check_data:
        if bool(torch.isnan(data).any()):
            ipdb.set_trace()


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class RelPosMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, max_relative_position):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.relative_position_k = RelativePosition(self.head_dim, max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, attn_mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        query = query.to(self.relative_position_k.embeddings_table.device)
        key = key.to(self.relative_position_k.embeddings_table.device)
        value = value.to(self.relative_position_k.embeddings_table.device)

        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)

        attn = (attn1 + attn2) / self.scale.to(attn1.device)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attn


class RelativePositionTransformerLayer(nn.Module):

    def __init__(self,
                 d_model: int,
                 nhead: int = 4,
                 dropout: float = 0.1,
                 dim_feedforward: int = 2048,
                 max_relative_position: int = 5,
                 ):
        super(RelativePositionTransformerLayer, self).__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = RelPosMultiHeadAttentionLayer(d_model, nhead, 0.01, max_relative_position)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        attn_output, attn_w = self.self_attention(src, src, src, attn_mask=src_mask)
        attn_output = self.dropout(attn_output)

        # Add residual connection and apply layer normalization
        norm_output1 = self.self_attn_layer_norm(src + attn_output)

        # Apply feedforward layer
        ff_output = self.linear2(self.dropout(torch.relu(self.linear1(norm_output1))))
        ff_output = self.dropout(ff_output)

        # Add residual connection and apply layer normalization
        norm_output2 = self.ff_layer_norm(norm_output1 + ff_output)

        return norm_output2


class DemoSelectorTransformerRelPosRegressor(pl.LightningModule, ABC):
    def __init__(self,
                 model_name: str,
                 max_len: int,
                 score_cls_num: int,
                 embedding_dim: int,
                 lr: float,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 dim_feedforward: int = 2048,
                 max_relative_position: int = 5,
                 is_use_abs_pos: bool = True,
                 is_use_score_embed: bool = True
                 ):
        # Two-Tower Model
        super(DemoSelectorTransformerRelPosRegressor, self).__init__()
        self.model_name = model_name
        self.is_use_abs_pos = is_use_abs_pos
        self.is_use_score_embed = is_use_score_embed

        if is_use_abs_pos:
            self.pos_embedding = torch.nn.Embedding(max_len, embedding_dim)
        if is_use_score_embed:
            self.score_embedding = torch.nn.Embedding(score_cls_num, embedding_dim)
        self.infer_text_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
        )
        self.prompt_head_text = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
        )
        self.prompt_head_demo = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
        )
        self.demo_encoder_layer = torch.nn.Sequential(RelativePositionTransformerLayer(d_model=embedding_dim,
                                                                                       nhead=num_heads,
                                                                                       dropout=dropout,
                                                                                       dim_feedforward=dim_feedforward,
                                                                                       max_relative_position=max_relative_position
                                                                                       ),
                                                      RelativePositionTransformerLayer(d_model=embedding_dim,
                                                                                       nhead=num_heads,
                                                                                       dropout=dropout,
                                                                                       dim_feedforward=dim_feedforward,
                                                                                       max_relative_position=max_relative_position
                                                                                       ),
                                                      )
        self.act = torch.nn.Sigmoid()
        self.lr = lr
        self.loss_func = nn.MSELoss()

    def infer_text_tower(self, prompt, infer_text):
        prompt = self.prompt_head_text(prompt)
        infer_text = infer_text + prompt
        infer_text_embed = self.infer_text_head(infer_text)
        return infer_text_embed

    def few_shot_demo_tower(self, demo_embed, demo_scores, prompt):

        if self.is_use_abs_pos:
            pos_demo_embed = torch.broadcast_to(torch.arange(demo_embed.shape[1]),
                                                (demo_embed.shape[0], demo_embed.shape[1])).to(self.device)
            pos_demo_embed = demo_embed.shape[1] - 1 - pos_demo_embed
            pos_demo_embed = self.pos_embedding(pos_demo_embed)
            assert demo_embed.shape == pos_demo_embed.shape
            demo_embed += pos_demo_embed
        if self.is_use_score_embed:
            score_embed = self.score_embedding(demo_scores)
            assert demo_embed.shape == score_embed.shape
            demo_embed += score_embed

        self.demo_encoder_layer = self.demo_encoder_layer.to(self.device)
        demo_embed = self.demo_encoder_layer(demo_embed)
        demo_embed_mean_pool = torch.mean(demo_embed, dim=1)

        prompt = self.prompt_head_demo(prompt)
        demo_embed_mean_pool = demo_embed_mean_pool + prompt
        return demo_embed_mean_pool

    def forward(self,
                demo_embed,
                demo_scores,
                prompt,
                infer_text
                ):
        """
        Args:
            demo_embed:   torch.Size([batch_size, max_demo_num, 768])
            demo_scores:  torch.Size([batch_size, max_demo_num])
            prompt: torch.Size([batch_size, 768])
            infer_text: torch.Size([batch_size, 768])

        Returns:
            logits: (batch_size, num_classes)
                tensor([[-0.7669,  0.0160,  0.0957],
                        [-0.8393, -0.0182,  0.1811]], grad_fn=<AddmmBackward0>)
        """
        # Infer Text Tower
        infer_text_embed = self.infer_text_tower(prompt, infer_text)

        # Demo Tower
        demo_embed_mean_pool = self.few_shot_demo_tower(demo_embed, demo_scores, prompt)

        output = torch.diag(torch.matmul(demo_embed_mean_pool, infer_text_embed.T))
        output = self.act(output)
        return output

    def compute_loss(self, batch_data):
        (demo_embed, demo_scores, prompt, infer_text), actual_Y = batch_data
        predicted_Y = self.forward(demo_embed, demo_scores, prompt, infer_text)
        loss = self.loss_func(predicted_Y, actual_Y)
        return loss

    def training_step(self, train_batch, batch_idx):
        X_tuple, Y = train_batch
        check_nan(X_tuple, self)
        check_nan(Y, self)
        mean_loss = self.compute_loss(train_batch)
        self.log('train_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)
        return mean_loss

    def validation_step(self, val_batch, batch_idx):
        X_tuple, Y = val_batch
        check_nan(X_tuple, self)
        check_nan(Y, self)
        mean_loss = self.compute_loss(val_batch)
        self.log('val_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        X_tuple, Y = test_batch
        check_nan(X_tuple, self)
        check_nan(Y, self)
        mean_loss = self.compute_loss(test_batch)
        self.log('test_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

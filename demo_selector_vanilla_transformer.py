from abc import ABC

import ipdb
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters


def check_nan(to_check_data, model, step):
    for data in to_check_data:
        if bool(torch.isnan(data).any()):
            ipdb.set_trace()

    try:
        detect_nan_parameters(model)
    except:
        ipdb.set_trace()


class DemoSelectorTransformerRegressor(pl.LightningModule, ABC):
    def __init__(self,
                 model_name: str,
                 max_len: int,
                 score_cls_num: int,
                 embedding_dim: int,
                 lr: float,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 dim_feedforward: int = 2048
                 ):
        # Two-Tower Model
        from torch.nn.modules.transformer import TransformerEncoderLayer

        super(DemoSelectorTransformerRegressor, self).__init__()
        self.model_name = model_name
        self.pos_embedding = torch.nn.Embedding(max_len, embedding_dim)
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
        self.demo_encoder_layer = torch.nn.Sequential(TransformerEncoderLayer(d_model=embedding_dim,
                                                                              nhead=num_heads,
                                                                              dropout=dropout,
                                                                              dim_feedforward=dim_feedforward),
                                                      TransformerEncoderLayer(d_model=embedding_dim,
                                                                              nhead=num_heads,
                                                                              dropout=dropout,
                                                                              dim_feedforward=dim_feedforward),
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
        pos_demo_embed = torch.broadcast_to(torch.arange(demo_embed.shape[1]),
                                            (demo_embed.shape[0], demo_embed.shape[1])).to(self.device)
        pos_demo_embed = demo_embed.shape[1] - 1 - pos_demo_embed  # 离infer text的距离，最近的pos是0
        pos_demo_embed = self.pos_embedding(pos_demo_embed)
        score_embed = self.score_embedding(demo_scores)
        assert demo_embed.shape == pos_demo_embed.shape == score_embed.shape
        demo_embed = demo_embed + pos_demo_embed + score_embed
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
        check_nan(X_tuple, self, 'before training')
        check_nan(Y, self, 'before training')
        mean_loss = self.compute_loss(train_batch)
        self.log('train_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)
        return mean_loss

    def validation_step(self, val_batch, batch_idx):
        X_tuple, Y = val_batch
        check_nan(X_tuple, self, 'before validation')
        check_nan(Y, self, 'before validation')
        mean_loss = self.compute_loss(val_batch)
        self.log('val_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, test_batch, batch_idx):
        X_tuple, Y = test_batch
        check_nan(X_tuple, self, 'before test')
        check_nan(Y, self, 'before test')
        mean_loss = self.compute_loss(test_batch)
        self.log('test_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

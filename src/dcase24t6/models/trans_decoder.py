#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ./logs/train-2025.02.25-23.44.55-baseline/

from typing import Any, Optional, TypedDict

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchoutil import (
    lengths_to_pad_mask,
    masked_mean,
    randperm_diff,
    tensor_to_pad_mask,
)
from torchoutil.nn import Transpose

from dcase24t6.augmentations.mixup import sample_lambda
from dcase24t6.datamodules.hdf import Stage
from dcase24t6.models.aac import AACModel, Batch, TestBatch, TrainBatch, ValBatch
from dcase24t6.nn.decoders.aac_tfmer import AACTransformerDecoder

# from dcase24t6.nn.decoders.rnn_decoder import RNNDecoder
from dcase24t6.nn.decoding.beam import generate
from dcase24t6.nn.decoding.common import get_forbid_rep_mask_content_words
from dcase24t6.nn.decoding.forcing import teacher_forcing
from dcase24t6.nn.decoding.greedy import greedy_search
from dcase24t6.optim.schedulers import CosDecayScheduler
from dcase24t6.optim.utils import create_params_groups
from dcase24t6.tokenization.aac_tokenizer import AACTokenizer

ModelOutput = dict[str, Tensor]

# import io

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

# import tensorflow as tf
import spacy

nlp = spacy.load("en_core_web_sm")


def add_time_stamps_to_tags(
    words: list = [], time_stamps: list = [], filter_tags: list = ["NOUN", "VERB"]
):

    processed = []

    for i, word in enumerate(words):
        doc = nlp(word)
        if doc[0].pos_ in filter_tags:  # pos = (doc[0].text, doc[0].pos_)
            processed.append(f"{doc[0].text} ({time_stamps[i]})")
        else:
            processed.append(doc[0].text)

    return " ".join(processed)


class AudioEncoding(TypedDict):
    frame_embs: Tensor
    frame_embs_pad_mask: Tensor


class TransDecoderModel(AACModel):
    def __init__(
        self,
        tokenizer: AACTokenizer,
        # decoder_type: str = "aac",
        # Model architecture args
        in_features: int = 768,
        d_model: int = 256,
        # Train args
        label_smoothing: float = 0.2,
        mixup_alpha: float = 0.4,
        # Inference args
        min_pred_size: int = 3,
        max_pred_size: int = 20,
        beam_size: int = 3,
        # Optimizer args
        custom_weight_decay: bool = True,
        lr: float = 5e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 2.0,
        sched_num_steps: int = 400,
        # Other args
        verbose: int = 0,
    ) -> None:
        super().__init__(tokenizer)
        self.projection: nn.Module = nn.Identity()
        self.decoder: AACTransformerDecoder = None  # type: ignore
        self.validation_iteration = 0
        # self.decoder_type = decoder_type
        # Add attention mechanism
        # self.attention = nn.MultiheadAttention(embed_dim=4371, num_heads=3)
        # Add linear layer to project logits to the correct dimension
        # self.linear_proj = nn.Linear(in_features, d_model)
        self.save_hyperparameters(ignore=["tokenizer"])

    def is_built(self) -> bool:
        return self.decoder is not None

    def setup(self, stage: Stage) -> None:
        if stage in ("fit", None) and "batch_size" in self.datamodule.hparams:
            source_batch_size = self.datamodule.hparams["batch_size"]
            target_batch_size = 1
            self.datamodule.hparams["batch_size"] = target_batch_size
            loader = self.datamodule.train_dataloader()
            self.datamodule.hparams["batch_size"] = source_batch_size
            batch = next(iter(loader))
            self.example_input_array = {"batch": batch}

    def configure_model(self) -> None:
        if self.is_built():
            return None

        projection = nn.Sequential(
            nn.Dropout(p=0.5),
            Transpose(1, 2),
            nn.Linear(self.hparams["in_features"], self.hparams["d_model"]),
            nn.ReLU(inplace=True),
            Transpose(1, 2),
            nn.Dropout(p=0.5),
        )

        # Original code
        decoder = AACTransformerDecoder(
            vocab_size=self.tokenizer.get_vocab_size(),
            pad_id=self.tokenizer.pad_token_id,
            d_model=self.hparams["d_model"],
        )
        # decoder = RNNDecoder(
        #     vocab_size=self.tokenizer.get_vocab_size(),
        #     d_model=self.hparams["d_model"],
        #     num_layers=6,
        # )
        # Alternative code
        # if self.decoder_type == "aac":
        #     decoder = AACTransformerDecoder(
        #         vocab_size=self.tokenizer.get_vocab_size(),
        #         pad_id=self.tokenizer.pad_token_id,
        #         d_model=self.hparams["d_model"],
        #     )
        # elif self.decoder_type == "rnn":
        #     decoder = RNNDecoder(
        #         vocab_size=self.tokenizer.get_vocab_size(),
        #         d_model=self.hparams["d_model"],
        #         num_layers=6,
        #     )
        # else:
        #     raise ValueError(f"Unknown decoder type: {self.decoder_type}")

        forbid_rep_mask = get_forbid_rep_mask_content_words(
            vocab_size=self.tokenizer.get_vocab_size(),
            token_to_id=self.tokenizer.get_token_to_id(),
            device=self.device,
            verbose=self.hparams["verbose"],
        )

        self.projection = projection
        self.decoder = decoder
        self.register_buffer("forbid_rep_mask", forbid_rep_mask)
        self.forbid_rep_mask: Optional[Tensor]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams["custom_weight_decay"]:
            params = create_params_groups(self, self.hparams["weight_decay"])
        else:
            params = self.parameters()

        optimizer_args = {
            name: self.hparams[name] for name in ("lr", "betas", "eps", "weight_decay")
        }
        optimizer = AdamW(params, **optimizer_args)

        num_steps = self.hparams["sched_num_steps"]
        scheduler = CosDecayScheduler(optimizer, num_steps)

        return [optimizer], [scheduler]

    def training_step(self, batch: TrainBatch) -> Tensor:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        captions = batch["captions"]

        bsize, _max_caption_length = captions.shape
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]
        del captions

        indexes = randperm_diff(bsize, device=self.device)
        audio, audio_shape, lbd = self.mix_audio(audio, audio_shape, indexes)
        captions_in_pad_mask = tensor_to_pad_mask(
            captions_in, pad_value=self.tokenizer.pad_token_id
        )
        captions_in = self.input_emb_layer(captions_in)
        captions_in = captions_in * lbd + captions_in[indexes] * (1.0 - lbd)

        # new
        # captions_in = captions_in.long()

        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(
            encoded,
            captions=captions_in,
            captions_pad_mask=captions_in_pad_mask,
            method="forcing",
        )
        logits = decoded["logits"]

        loss = self.train_criterion(logits, captions_out)
        self.log("train/loss", loss, batch_size=bsize, prog_bar=True)

        return loss

    def validation_step(self, batch: ValBatch) -> dict[str, Tensor]:
        self.validation_iteration += 1
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        mult_captions = batch["mult_captions"]

        bsize, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.tokenizer.pad_token_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                bsize,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(max_captions_per_audio):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.val_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("val/loss", loss, batch_size=bsize, prog_bar=True)

        decoded = self.decode_audio(encoded, method="generate")
        outputs = {
            "val/loss": losses,
        } | decoded

        # print(f"decoded: {decoded}")
        # print(f"decoded['candidate']: {decoded['candidate']}")

        # print("Executing validation_step")
        # print(f"fname: {batch['fname'][0]}")
        # for the attention maps only
        tensorboard = self.logger.experiment
        max_samples = 1
        for sample_id in range(max_samples):
            encoded_i = {}
            for k in encoded.keys():
                encoded_i[k] = encoded[k][sample_id : sample_id + 1, :]
            captions_in_i = mult_captions_in[sample_id : sample_id + 1, i]
            decoded_i = self.decode_audio(encoded_i, captions=captions_in_i)
            decoded_generate = self.decode_audio(encoded_i, method="generate")
            all_attn_weights = []
            for j, l in enumerate(self.decoder.layers):
                all_attn_weights.append(l.attn_weights.mean(dim=0))
                self.plot_attention(
                    attention=l.attn_weights.mean(dim=0),
                    title=f"layer {j}",
                    xtitle=decoded_generate["candidates"][0],
                    cands_list=decoded_generate["candidates_cands_list"][0][0],
                )
                tensorboard.add_figure(
                    f"validation_{batch['fname'][sample_id].replace(' ', '_')}/attn_weights_l{j}",
                    plt.gcf(),
                    global_step=self.trainer.current_epoch,
                )
            all_attn_weights_mean = torch.stack(all_attn_weights, dim=0).mean(dim=0)
            # if all_attn_weights_mean.shape[0] != len(
            #     decoded_generate["candidates_cands_list"][0]
            # ):
            #     stop = 1
            # if all_attn_weights_mean.shape[0] != len(
            #     decoded_generate["candidates"][0].split(" ")
            # ):
            #     stop = 1
            self.plot_attention(
                attention=all_attn_weights_mean,
                title="all layers mean",
                xtitle=decoded_generate["candidates"][0],
                cands_list=decoded_generate["candidates_cands_list"][0][0],
            )
            tensorboard.add_figure(
                f"validation_{batch['fname'][sample_id].replace(' ', '_')}/attn_weights_all_layers_mean",
                plt.gcf(),
                global_step=self.trainer.current_epoch,
            )
        # NEW
        # https://pytorch-lightning.readthedocs.io/en/0.10.0/logging.html#manual-logging
        # https://github.com/Lightning-AI/pytorch-lightning/issues/3697#issuecomment-703910904
        # Images-->https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
        # https://stackoverflow.com/a/66537286
        # https://stackoverflow.com/a/52467925
        # https://stackoverflow.com/questions/51496619/tensorboard-how-to-write-images-to-get-a-steps-slider

        # #### tensorboard = self.logger.experiment
        # #### for j, l in enumerate(self.decoder.layers):
        # print(l.attn_weights.mean(dim=0))
        # tensorboard.add_embedding(l.attn_weights.mean(dim=0), metadata = "validation/attn_weights")
        # #####self.plot_attention(attention = l.attn_weights.mean(dim=0), title = f"attn_weights layer {j}")
        # ---
        # image = io.BytesIO()
        # plt.savefig(image, format='png')
        # image = tf.summary.image(encoded_image_string=image.getvalue(),
        #                          height=10,
        #                          width=10)
        # summary = tf.summary(value=[tf.Summary.Value(tag=f"validation/attn_weights_l{i}", image=image)])
        # tensorboard.add_summary(summary, global_step=self.trainer.current_epoch)
        # ---
        # #####tensorboard.add_figure(f"validation/attn_weights_l{j}", plt.gcf(), global_step = self.trainer.current_epoch)
        # tensorboard.add_figure(f"validation/attn_weights", plt.gcf())
        # ---
        # self.log("validation/attn_weights", l.attn_weights.mean(dim=0))

        # print(l.attn_weights)
        # attention_maps.append(l.attn_weights)
        # print(l.attn_weights.shape)
        # attn_weights = self.decoder.get_attention_maps(x = encoded)
        # self.log("validation/attn_weights", attn_weights)

        return outputs

    # def plot_attention(self, attention, queries, keys, xtitle="Keys", ytitle="Queries"):
    def plot_attention(
        self, attention, title="", xtitle="Keys", ytitle="Queries", cands_list=[]
    ):
        """Plots the attention map

        Args:
            att (torch.FloatTensor): Attention map (T_q x T_k)
            queries (List[str]): Query Tensor
            keys (List[str]): Key Tensor
        """

        # sns.set(rc={'figure.figsize':(12, 8)})
        plt.figure(figsize=(16, 16))
        ax = sns.heatmap(attention.detach().cpu(), cmap="coolwarm")
        # linewidth=0.5,
        # xticklabels=keys,
        # yticklabels=queries,

        out, inds = torch.max(attention, dim=1)
        out_out, inds_inds = torch.max(out, dim=0)
        # title += f"max at {out_out:.4f} at frame {inds[inds_inds]} * 320/1000 = {inds[inds_inds]*320/1000} secs"
        title += f" (token {inds_inds} in frame {inds[inds_inds]}  = {inds[inds_inds]*320/1000:.4f} secs)"

        firstpart, secondpart = xtitle[: len(xtitle) // 2], xtitle[len(xtitle) // 2 :]
        arr = np.array(inds.tolist()) * 320 / 1000
        subtitle = (
            str(len(arr)) + " time stamps : " + ", ".join([str(i) for i in list(arr)])
        )
        subtitle2 = add_time_stamps_to_tags(words=cands_list, time_stamps=list(arr))
        sub_firstpart, sub_secondpart = (
            subtitle2[: len(subtitle2) // 2],
            subtitle2[len(subtitle2) // 2 :],
        )
        xtitle = (
            f"{firstpart}\n{secondpart}\n{subtitle}\n{sub_firstpart}\n{sub_secondpart}"
        )

        ax.set_title(title)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)

    def test_step(self, batch: TestBatch) -> dict[str, Any]:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        mult_captions = batch["mult_captions"]

        bsize, max_captions_per_audio, _max_caption_length = mult_captions.shape
        mult_captions_in = mult_captions[:, :, :-1]
        mult_captions_out = mult_captions[:, :, 1:]
        is_valid_caption = (mult_captions != self.tokenizer.pad_token_id).any(dim=2)
        del mult_captions

        encoded = self.encode_audio(audio, audio_shape)
        losses = torch.empty(
            (
                bsize,
                max_captions_per_audio,
            ),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(max_captions_per_audio):
            captions_in_i = mult_captions_in[:, i]
            captions_out_i = mult_captions_out[:, i]

            decoded_i = self.decode_audio(encoded, captions=captions_in_i)
            logits_i = decoded_i["logits"]
            losses_i = self.test_criterion(logits_i, captions_out_i)
            losses[:, i] = losses_i

        loss = masked_mean(losses, is_valid_caption)
        self.log("test/loss", loss, batch_size=bsize, prog_bar=True)

        decoded = self.decode_audio(encoded, method="generate")
        outputs = {
            "test/loss": losses,
        } | decoded
        return outputs

    def forward(
        self,
        batch: Batch,
        **method_kwargs,
    ) -> ModelOutput:
        audio = batch["frame_embs"]
        audio_shape = batch["frame_embs_shape"]
        captions = batch.get("captions", None)
        encoded = self.encode_audio(audio, audio_shape)
        decoded = self.decode_audio(encoded, captions, **method_kwargs)
        return decoded

    def train_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(
            logits,
            target,
            ignore_index=self.tokenizer.pad_token_id,
            label_smoothing=self.hparams["label_smoothing"],
        )
        return loss

    def val_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        losses = F.cross_entropy(
            logits,
            target,
            ignore_index=self.tokenizer.pad_token_id,
            reduction="none",
        )
        # We apply mean only on second dim to get a tensor of shape (bsize,)
        losses = masked_mean(losses, target != self.tokenizer.pad_token_id, dim=1)
        return losses

    def test_criterion(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.val_criterion(logits, target)

    def input_emb_layer(self, ids: Tensor) -> Tensor:
        # New: Ensure ids is of type torch.LongTensor
        # ids = ids.long()
        return self.decoder.emb_layer(ids)

    def mix_audio(
        self, audio: Tensor, audio_shape: Tensor, indexes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        lbd = sample_lambda(
            self.hparams["mixup_alpha"],
            asymmetric=True,
            size=(),
        )
        mixed_audio = audio * lbd + audio[indexes] * (1.0 - lbd)
        mixed_audio_shape = torch.max(audio_shape, audio_shape[indexes])
        return mixed_audio, mixed_audio_shape, lbd

    def encode_audio(
        self,
        frame_embs: Tensor,
        frame_embs_shape: Tensor,
    ) -> AudioEncoding:
        # frame_embs: (bsize, 1, in_features, max_seq_size)
        # frame_embs_shape: (bsize, 3)

        time_dim = -1

        frame_embs = frame_embs.squeeze(dim=1)  # remove channel dim
        frame_embs_lens = frame_embs_shape[:, time_dim]
        # frame_embs_lens: (bsize,)

        frame_embs = self.projection(frame_embs)
        # frame_embs: (bsize, d_model, max_seq_size)

        frame_embs_max_len = max(
            int(frame_embs_lens.max().item()), frame_embs.shape[time_dim]
        )
        frame_embs_pad_mask = lengths_to_pad_mask(frame_embs_lens, frame_embs_max_len)

        return {
            "frame_embs": frame_embs,
            "frame_embs_pad_mask": frame_embs_pad_mask,
        }

    def decode_audio(
        self,
        audio_encoding: AudioEncoding,
        captions: Optional[Tensor] = None,
        captions_pad_mask: Optional[Tensor] = None,
        method: str = "auto",
        **method_overrides,
    ) -> dict[str, Tensor]:
        if method == "auto":
            if captions is None:
                method = "generate"
            else:
                method = "forcing"

        common_args: dict[str, Any] = {
            "decoder": self.decoder,
            "pad_id": self.tokenizer.pad_token_id,
        } | audio_encoding

        match method:
            case "forcing":
                forcing_args = {
                    "caps_in": captions,
                    "caps_in_pad_mask": captions_pad_mask,
                }
                kwargs = common_args | forcing_args | method_overrides
                logits = teacher_forcing(**kwargs)
                # print(f"logits shape 1: {logits.shape}")
                # Project logits to the correct dimension
                # logits = self.linear_proj(logits)
                # print(f"logits shape 2: {logits.shape}")
                # Apply attention mechanism
                # logits, _ = self.attention(logits, logits, logits)
                # --------------
                # logits_permuted = logits.permute(
                #     2, 0, 1
                # )  # [batch_size, embed_dim, seq_len] -> [seq_len, batch_size, embed_dim]
                # attended_output, _ = self.attention(
                #     logits_permuted, logits_permuted, logits_permuted
                # )
                # logits_attended = attended_output.permute(
                #     1, 2, 0
                # )  # [seq_len, batch_size, embed_dim] -> [batch_size, embed_dim, seq_len]
                # logits = logits + logits_attended
                # --------------
                outs = {"logits": logits}

            case "greedy":
                greedy_args = {
                    "bos_id": self.tokenizer.bos_token_id,
                    "eos_id": self.tokenizer.eos_token_id,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                    "min_pred_size": self.hparams["min_pred_size"],
                    "max_pred_size": self.hparams["max_pred_size"],
                    "forbid_rep_mask": self.forbid_rep_mask,
                }
                kwargs = common_args | greedy_args | method_overrides
                logits = greedy_search(**kwargs)
                # Project logits to the correct dimension
                # logits = self.linear_proj(logits)
                # Apply attention mechanism
                # logits, _ = self.attention(logits, logits, logits)
                # --------------
                # logits_permuted = logits.permute(2, 0, 1)
                # attended_output, _ = self.attention(
                #     logits_permuted, logits_permuted, logits_permuted
                # )
                # logits_attended = attended_output.permute(1, 2, 0)
                # logits = logits + logits_attended
                # --------------
                outs = {"logits": logits}

            case "generate":
                generate_args = {
                    "bos_id": self.tokenizer.bos_token_id,
                    "eos_id": self.tokenizer.eos_token_id,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                    "min_pred_size": self.hparams["min_pred_size"],
                    "max_pred_size": self.hparams["max_pred_size"],
                    "forbid_rep_mask": self.forbid_rep_mask,
                    "beam_size": self.hparams["beam_size"],
                }
                kwargs = common_args | generate_args | method_overrides
                outs = generate(**kwargs)
                outs = outs._asdict()

                # Decode predictions ids to sentences
                keys = list(outs.keys())
                for key in keys:
                    if "prediction" not in key:
                        continue
                    preds: Tensor = outs[key]
                    if preds.ndim == 2:
                        cands = self.tokenizer.decode_batch(preds.tolist())
                        cands_list = [
                            [
                                self.tokenizer.decode(
                                    [token], skip_special_tokens=False
                                )
                                for token in seq
                            ]
                            for seq in preds.tolist()
                        ]
                    elif preds.ndim == 3:
                        cands = [
                            self.tokenizer.decode_batch(value_i)
                            for value_i in preds.tolist()
                        ]
                        cands_list = [
                            [
                                [
                                    self.tokenizer.decode(
                                        [token], skip_special_tokens=False
                                    )
                                    for token in seq
                                ]
                                for seq in seq_list
                            ]
                            for seq_list in preds.tolist()
                        ]
                        # cands_list = [
                        #     self.tokenizer.decode_batch(
                        #         value_i, skip_special_tokens=False
                        #     )
                        #     for value_i in preds.tolist()
                        # ]
                    # if preds.ndim == 2:
                    #     cands = self.tokenizer.decode_batch(preds.tolist())
                    #     cands_list = [[self.tokenizer.decode([token], skip_special_tokens=False) for token in seq] for seq in preds.tolist()]
                    # elif preds.ndim == 3:
                    #     cands = [
                    #         self.tokenizer.decode_batch(value_i)
                    #         for value_i in preds.tolist()
                    #     ]
                    #     cands_list = [
                    #             self.tokenizer.decode_batch(value_i, skip_special_tokens=False)
                    #             for value_i in preds.tolist()
                    #     ]
                    else:
                        raise ValueError(
                            f"Invalid shape {preds.ndim=}. (expected one of {(2, 3)})"
                        )
                    new_key = key.replace("prediction", "candidate")
                    outs[new_key] = cands
                    outs[f"{new_key}_cands_list"] = cands_list

            case method:
                DECODE_METHODS = ("forcing", "greedy", "generate", "auto")
                raise ValueError(
                    f"Unknown argument {method=}. (expected one of {DECODE_METHODS})"
                )

        return outs

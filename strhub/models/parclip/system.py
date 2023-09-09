# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from itertools import permutations
from typing import Sequence, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pytorch_lightning.utilities.types import STEP_OUTPUT
from timm.models.helpers import named_apply

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .modules import DecoderLayer, Decoder, Encoder, TokenEmbedding

from .CLIP import clip,model,simple_tokenizer
import torch

import matplotlib.pyplot as plt
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

class PARCLIP(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int,
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int,
                 dec_num_heads: int, dec_mlp_ratio: int, dec_depth: int,
                 perm_num: int, perm_forward: bool, perm_mirrored: bool,
                 decode_ar: bool, refine_iters: int, dropout: float, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()

        self.max_label_length = max_label_length
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters

        decoder_layer = DecoderLayer(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dropout)
        self.decoder = Decoder(decoder_layer, num_layers=dec_depth, norm=nn.LayerNorm(embed_dim))

        # Perm/attn mask stuff
        self.rng = np.random.default_rng()
        #self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        #self.perm_forward = perm_forward
        #self.perm_mirrored = perm_mirrored
        self.embed_dim = embed_dim
        # We don't predict <bos> nor <pad>
        self.head = nn.Linear(embed_dim, len(self.tokenizer) - 2)
        #print( len(self.tokenizer) - 2) #37
        #self.text_embed = TokenEmbedding(len(self.tokenizer), embed_dim)

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, 77, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        # Encoder has its own init.
        named_apply(partial(init_weights, exclude=['encoder']), self)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

        self.CLIPmodel, self.CLIPpreprocess = clip.load('ViT-B/16')
        # 모델 파라미터 고정하기
        for param in self.CLIPmodel.parameters():
            param.requires_grad = False

        self.charset_train = charset_train

        # self.encoder = Encoder(img_size, patch_size, embed_dim=embed_dim, depth=enc_depth, num_heads=enc_num_heads,
        #                        mlp_ratio=enc_mlp_ratio)

        dic = simple_tokenizer.SimpleTokenizer(max_label_length= self.max_label_length, charset = self.charset_train)
        self.label_origin = dic.getLabelVocab()
        

        self.new = True
        #Leehakho
        self.padding = False
        self.load_features = False
        self.use_gt = False
        self.seperate = False

        self.label = self.label_origin
        if self.load_features:
            self.new = False
            # 파일에서 텐서를 불러오기
            features = torch.load('text_features_new_30000.pth').to(self._device)
            number = ["60000", "87837"]
            for num in number:
                temp = torch.load('text_features_new_' + num + '.pth').to(self._device)
                features = torch.cat((features, temp), axis=0)
            #print(features.shape)
            self.text_features = features
            self.tm = self.text_features
        else:
            if self.seperate:
                self.text_token = []
                for l in self.label:
                    a = []
                    a.append(l)
                    tt =torch.cat([clip.tokenize(f"word {c}") for c in a]).to(self._device)
                    self.text_token.append(tt)
            else:
                self.label = random.sample(self.label_origin, 3000)
                #self.label = self.label_origin
                #print(self.label)

                if self.padding:
                    self.label = self.label_origin[:1]
                
                self.text_token = torch.cat([clip.tokenize(f"word {c}") for c in self.label])
                #print(self.text_token.shape)
    
        #self.my_linear1 = nn.Linear(512,self.embed_dim)
        #self.my_linear2 = nn.Linear(512,self.embed_dim)
    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        #enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names#.union(enc_param_names)

    def clip_encode(self, img: torch.Tensor):
        #print(img.shape)
        #img = F.interpolate(img, size=224)
        #print(img.shape)
        #img = F.interpolate(img.unsqueeze(0), size=(3,224,224), mode='bilinear', align_corners=False)
        #print(img.shape) #torch.Size([768, 3, 32, 128])
        with torch.no_grad():
            #emb = self.CLIPmodel.visual(img)
            #print(img[0][0].shape)
            emb = self.CLIPmodel.encode_image(img)
            #print(emb[:][0].shape, emb[:][1].shape) #512,768
        return emb
        #return self.encoder(img)

    def encode(self, img: torch.Tensor):
        return self.encoder(img)

    def txtencode(self, text: torch.Tensor):

        #print(text.shape)
        with torch.no_grad():
            emb = self.CLIPmodel.encode_text(text.to(self._device))
        return emb
    
    def decode(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None, max_len: Optional[int] = None, GT: Optional[Tensor] = None):


        if GT is not None:
            tgt_list = GT
        else:
            if self.load_features:
                if self.new:
                    self.text_features /= self.text_features.norm(dim=-1, keepdim=True).to(self._device)
                    self.new = False

            elif self.seperate:
                if self.new:
                    #self.text_features = torch.cat([self.txtencode(c) for c in self.text_token]).to(self._device)
                    self.tm = []
                    self.text_features =[]
                    for c in self.text_token:
                        temp = self.txtencode(c)
                        self.tm.append(temp[1])
                        self.text_features.append(temp[0])
                        #print(self.tm[0].shape, self.text_features[0].shape)
                    self.tm = torch.stack(self.tm)
                    self.text_features = torch.stack(self.text_features)
                    self.tm = self.tm.squeeze()
                    self.text_features = self.text_features.squeeze()
                    #print(self.tm.shape, self.text_features.shape)
                    # self.text_features = torch.cat([self.txtencode(c)) for c in self.text_token])
                    # self.tm = self.text_features[:][1]
                    # self.text_features = self.text_features[:][0]
                    self.text_features /= self.text_features.norm(dim=-1, keepdim=True).to(self._device)
                    self.new = False

            else:
                if self.new:
                    #self.text_token = self.text_token.to(self._device)
                    
                    self.text_features = self.txtencode(self.text_token)
                    self.tm = self.text_features[:][1]
                    self.text_features = self.text_features[:][0]
                    
                    self.text_features /= self.text_features.norm(dim=-1, keepdim=True).to(self._device)
                    self.new = False

            tgt_list = []
            tf = []
            for image_features in x:
                image_features /= image_features.norm(dim=-1, keepdim=True)
                #print(image_features.shape)
                similarity = (100.0 * image_features.to(self._device) @ self.text_features.T.to(self._device)).softmax(dim=-1).to(self._device)
                #print(similarity[0])
                values, indices = similarity.topk(1)

                #for value, index in zip(values, indices):
                #print(indices)
                #print("index:", indices, " clip pred:", tgt)
                #tgt = self.tokenizer.encode(tgt, self._device)

                tf.append(self.tm[indices])
                #tgt_emb.append(txt_emb[indices])

        tgt = torch.cat(tf, dim=0).to(self._device)
        #print(tgt.shape) #64,11
        #tgt = torch.stack(tgt_list)
        #tgt = tgt.squeeze(0)
        #B, N, L = tgt.shape
        B, N, L = tgt.shape
        #print(L)
        # <bos> stands for the null context. We only supply position information for characters after <bos>.

        tgt_emb = self.pos_queries[:, :L - 1] + tgt

        tgt_emb = self.dropout(tgt_emb)
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
            #print(tgt_query.shape, L, N) #torch.Size([64, 10, 384]) 10 64
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        #print(images.shape, max_length)
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.

        num_steps = max_length + 1
        x, memory = self.clip_encode(images)
        #print(x.shape, memory.shape) #torch.Size([768, 512]) torch.Size([768, 197, 768])

        #memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        #tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        tgt_out = self.decode(x, memory,tgt_query=pos_queries, max_len = max_length)
        #print(tgt_out.shape)  #torch.Size([32, 77, 512])
        #print(tgt_out.shape)
        # if max_length:
        #     tgt_out = tgt_out[:, :max_length +1, :]
        logits = self.head(tgt_out)
        #print(logits.shape, "!!!!") #1, 6, 37 same
        #loss = F.cross_entropy(logits, logits, ignore_index=self.pad_id)

        return logits

    def write_unique_strings_to_file(self, strings):
        existing_strings = set()
        file_path = "/home/ohh/PycharmProject/PARCLIP/labels.txt"
        try:
            # 기존 파일에서 이미 있는 문자열 읽어오기
            with open(file_path, 'r') as file:
                existing_strings = set(file.read().splitlines())
        except FileNotFoundError:
            pass

        unique_strings = set(strings) - existing_strings  # 중복 제거
        with open(file_path, 'a') as file:
            for string in unique_strings:
                file.write(string + '\n')

    def training_step(self, batch, batch_idx, max_length: Optional[int] = None) -> STEP_OUTPUT:
        images, labels = batch

        #GT 개수 뽑아보기
        #self.write_unique_strings_to_file(labels)

        tgt = self.tokenizer.encode(labels, self._device)
        gt = tgt[:, 1:]
        #tgt_out = tgt[:, 1:]

        loss = 0
        loss_numel = 0
        n = (gt != self.pad_id).sum().item()
        max_len = tgt.shape[1] -2 # exclude <eos> from count





        #out = self.forward(images, max_len)
        #forward 부분
        max_length = max_len
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        bs = images.shape[0]
        # +1 for <eos> at end of sequence.
        num_steps = max_length + 1
        x, memory = self.clip_encode(images)
        #memory = self.my_linear2(memory)
        
        #memory = self.encode(images)

        # Query positions up to `num_steps`
        pos_queries = self.pos_queries[:, :num_steps].expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        #tgt_mask = query_mask = torch.triu(torch.full((num_steps, num_steps), float('-inf'), device=self._device), 1)

        if self.use_gt:
            tgt_out = self.decode(x, memory, tgt_query=pos_queries, GT=labels)
        else:
            tgt_out = self.decode(x, memory, tgt_query=pos_queries)
        #print(tgt_out.shape) #torch.Size([64, 26, 384])
        # if max_length:
        #     tgt_out = tgt_out[:, :max_length +1, :]
        logits = self.head(tgt_out)
        #print(logits.shape, "!!!!") #1, 6, 37 same
        #loss = F.cross_entropy(logits, logits, ignore_index=self.pad_id)
        out = logits





        #print(out.shape, gt.shape)
        logits = out.flatten(end_dim=1)
        #print(logits.shape, gt.flatten().shape) #torch.Size([1664(고정), 37]) torch.Size([1024])
        loss += n * F.cross_entropy(logits, gt.flatten(), ignore_index=self.pad_id)
        loss_numel += n

        #tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
        #n = (tgt_out != self.pad_id).sum().item()

        #loss = F.cross_entropy(logits, gt.flatten(), ignore_index=self.pad_id)
        loss /= loss_numel


        self.log('loss', loss)
        #print(loss) #tensor(4.1242, device='cuda:6', grad_fn=<DivBackward0>)
        return loss

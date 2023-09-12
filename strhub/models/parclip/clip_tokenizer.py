from .CLIP import clip,model,simple_tokenizer
import torch
import random

class clip_tokinzer():
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int) -> None:
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
                    self.text_token.append(torch.cat([clip.tokenize(f"word {c}") for c in a]).to(self._device))
            else:
                self.label = random.sample(self.label_origin, 3000)
                #print(self.label)

                if self.padding:
                    self.label = self.label_origin[:1]
                
                self.text_token = torch.cat([clip.tokenize(f"word {c}") for c in self.label])
    
    def txtencode(self, text: torch.Tensor):

        #print(text.shape)
        with torch.no_grad():
            emb = self.CLIPmodel.encode_text(text.to(self._device))
        return emb


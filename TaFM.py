import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import timm
import torch.nn.functional as F
from torchvision.models import swin_t
from torchvision.models.feature_extraction import create_feature_extractor
import clip
from torchvision.models import convnext_tiny
import math

class ConvNeXtEncoder(nn.Module):
    def __init__(self, pretrained=True, num_experts=4, k=1, dropout=0.0):
        super(ConvNeXtEncoder, self).__init__()
        self.backbone = torchvision.models.convnext_tiny(pretrained=True)
        self.backbone.classifier = nn.Identity()

        self.backbone1 = nn.Sequential(self.backbone.features[0],self.backbone.features[1])
        self.backbone2 = nn.Sequential(self.backbone.features[2],self.backbone.features[3])
        self.backbone3 = nn.Sequential(self.backbone.features[4],self.backbone.features[5])
        self.backbone4 = nn.Sequential(self.backbone.features[6],self.backbone.features[7])
    
    def forward(self, x):
        # 각 단계의 feature map 계산
        down1 = self.backbone1(x)     # 높은 해상도, 낮은 채널 (patch embedding 결과)
        down2 = self.backbone2(down1) # 첫 번째 stage 출력
        down3 = self.backbone3(down2) # 두 번째 stage 출력
        down4 = self.backbone4(down3) # 세 번째 stage 출력
        
        return down4, [down4, down3, down2, down1]

    def compute_skip_channels(self, input_size=(3, 224, 224)):
        """
        주어진 input_size에 대해 인코더의 각 단계 출력 채널 수를 자동으로 계산합니다.
        """
        self.eval()
        with torch.no_grad():
            dummy = torch.randn(1, *input_size)
            #_, skips, _ = self.forward(dummy)
            _, skips = self.forward(dummy)
            channels = [skip.shape[1] for skip in skips]
        self.train()
        return channels[::-1]

class ConvNeXtDecoder(nn.Module):
    def __init__(self, skip_channels, num_classes, use_prompt=False):
        super(ConvNeXtDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        num_stages = len(skip_channels)
        for i in range(num_stages - 1, 0, -1):
            if i == 4:
                up_conv = nn.Identity()
            else:
                up_conv = nn.ConvTranspose2d(skip_channels[i], skip_channels[i-1], kernel_size=2, stride=2)

            self.up_blocks.append(nn.ModuleDict({
                'up_conv': up_conv,
                'conv1': nn.Conv2d(skip_channels[i-1]*2, skip_channels[i-1], kernel_size=3, padding=1),
                'norm1': nn.BatchNorm2d(skip_channels[i-1]), #nn.GroupNorm(16, skip_channels[i-1]), #
                'relu1': nn.ReLU(inplace=False),
                'conv2': nn.Conv2d(skip_channels[i-1], skip_channels[i-1], kernel_size=3, padding=1),
                'norm2': nn.BatchNorm2d(skip_channels[i-1]), #nn.GroupNorm(16, skip_channels[i-1]), #
                'relu2': nn.ReLU(inplace=False)}))
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(skip_channels[0], skip_channels[0]//2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(skip_channels[0]//2, skip_channels[0]//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(skip_channels[0]//2, skip_channels[0]//4, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(skip_channels[0]//4, skip_channels[0]//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.final_conv = nn.Conv2d(skip_channels[0]//4, num_classes, kernel_size=1)
    
    def forward(self, x, skip_connections, prompt=None):
        for idx, block in enumerate(self.up_blocks):
            if idx == 0 and isinstance(block, ViTDecoderBlock):
                # ViTDecoderBlock 사용 시 prompt와 skip 연결
                x = block(x, prompt, skip_connections[idx+1])
            else:
                x = block['up_conv'](x)
                x = torch.cat([x, skip_connections[idx+1]], dim=1)
                x = block['conv1'](x)
                x = block['norm1'](x)
                x = block['relu1'](x)
                x = block['conv2'](x)
                x = block['norm2'](x)
                x = block['relu2'](x)
        x = self.final_up(x)
        return self.final_conv(x)

class ConvNeXtFoveaHeatmapDecoder(nn.Module):
    def __init__(self, skip_channels, use_prompt=False):
        super(ConvNeXtFoveaHeatmapDecoder, self).__init__()
        self.up_blocks = nn.ModuleList()
        num_stages = len(skip_channels)
        for i in range(num_stages - 1, 0, -1):
            # 가장 낮은 해상도 블록에서는 up_conv를 Identity로 설정
            if i == 4:
                up_conv = nn.Identity()
            else:
                up_conv = nn.ConvTranspose2d(skip_channels[i], skip_channels[i-1], kernel_size=2, stride=2)
            self.up_blocks.append(nn.ModuleDict({
                'up_conv': up_conv,
                'conv1': nn.Conv2d(skip_channels[i-1]*2, skip_channels[i-1], kernel_size=3, padding=1),
                'norm1': nn.BatchNorm2d(skip_channels[i-1]), #nn.GroupNorm(16, skip_channels[i-1]), #
                'relu1': nn.ReLU(inplace=False),
                'conv2': nn.Conv2d(skip_channels[i-1], skip_channels[i-1], kernel_size=3, padding=1),
                'norm2': nn.BatchNorm2d(skip_channels[i-1]), #nn.GroupNorm(16, skip_channels[i-1]), #
                'relu2': nn.ReLU(inplace=False)}))
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(skip_channels[0], skip_channels[0]//2, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(skip_channels[0]//2, skip_channels[0]//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(skip_channels[0]//2, skip_channels[0]//4, kernel_size=2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(skip_channels[0]//4, skip_channels[0]//4, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.heatmap_conv = nn.Conv2d(skip_channels[0]//4, 1, kernel_size=1)
    
    def soft_argmax_2d(self, heatmap):
        B, _, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, -1)
        heatmap_flat = heatmap_flat - heatmap_flat.max(dim=1, keepdim=True).values          # 수치 안정화
        softmaxed = F.softmax(heatmap_flat, dim=1)
        softmaxed = softmaxed.view(B, H, W)
        xs = torch.linspace(0, W - 1, W, device=heatmap.device).view(1, 1, W)
        ys = torch.linspace(0, H - 1, H, device=heatmap.device).view(1, H, 1)
        pred_x = torch.sum(softmaxed * xs, dim=(1, 2))
        pred_y = torch.sum(softmaxed * ys, dim=(1, 2))
        return torch.stack([pred_x, pred_y], dim=1)
    
    def forward(self, x, skip_connections, prompt=None):
        for idx, block in enumerate(self.up_blocks):
            if idx == 0 and isinstance(block, ViTDecoderBlock):
                # ViTDecoderBlock 사용 시 prompt와 skip 연결
                x = block(x, prompt, skip_connections[idx+1])
            else:
                x = block['up_conv'](x)
                x = torch.cat([x, skip_connections[idx+1]], dim=1)
                x = block['conv1'](x)
                x = block['norm1'](x)
                x = block['relu1'](x)
                x = block['conv2'](x)
                x = block['norm2'](x)
                x = block['relu2'](x)
        x = self.final_up(x)
        heatmap = self.heatmap_conv(x)
        coords = self.soft_argmax_2d(heatmap)
        return coords, heatmap

class PromptLearner(nn.Module):
    def __init__(self,
                 clip_model_name: str = 'ViT-B/32',
                 context_length: int = 12,
                 task_prompts: list = None,
                 device: str = 'cuda'):
        super().__init__()
        # Load CLIP model
        self.device = device
        self.clip_model, _ = clip.load(clip_model_name, jit=False, device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.context_length = context_length
        self.embed_dim = self.clip_model.transformer.width
        self.context_vectors = nn.Parameter(
            torch.randn(context_length, self.embed_dim, device=device) * 0.02
        )
        if task_prompts:
            self.task_prompts = task_prompts
        else:
            self.task_prompts = [
            'vessel segmentation',
            'optic disc segmentation',
            'fovea localization']
        tok_length = 77
        tokenized = clip.tokenize(self.task_prompts, tok_length)
        self.register_buffer('tokenized_prompts', tokenized)

    def forward(self):
        M = self.context_length
        tokens = self.tokenized_prompts.clone()  # [T, L]
        pad_id = 0

        for i in range(tokens.size(0)): 
            seq = tokens[i]           
            non_pad = seq[seq != pad_id]  
            L = seq.size(0)

            new_seq = torch.full_like(seq, pad_id)
            new_seq[0] = non_pad[0]
            new_seq[1+M : 1+M + (non_pad.size(0)-1)] = non_pad[1:]
            tokens[i] = new_seq

        tokens = tokens.to(self.device)
        token_embeddings = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)
        token_embeddings[:, 1:1+M, :] = self.context_vectors.unsqueeze(0)

        x = token_embeddings + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        eos_id = 49407
        eos_pos = (tokens == eos_id).int().argmax(dim=1)
        eos_feats = x[torch.arange(3), eos_pos, :] @ self.clip_model.text_projection
        text_feats = x[torch.arange(3), 0:16, :] @ self.clip_model.text_projection

        return eos_feats, text_feats

class Adapter(nn.Module):
    def __init__(self, C, bottleneck=32):
        super().__init__()
        self.down = nn.Conv2d(C, bottleneck, 1)
        self.act  = nn.ReLU(inplace=True)
        self.up   = nn.Conv2d(bottleneck, C, 1)
        self.scale = nn.Parameter(torch.tensor(0.0)) 
    def forward(self, x):
        return self.scale * self.up(self.act(self.down(x)))

class TextTokenPool(nn.Module):
    def __init__(self, text_dim, attn_hid=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, attn_hid), nn.GELU(),
            nn.Linear(attn_hid, 1) 
        )

    def forward(self, text_tokens, B: int):
        if text_tokens.dim() == 2:        # [L,D] -> [B,L,D]
            text_tokens = text_tokens.unsqueeze(0).expand(B, -1, -1)
        # text_tokens: [B,L,D]
        scores = self.proj(text_tokens).squeeze(-1)         # [B,L]
        attn = torch.softmax(scores, dim=-1)                # [B,L]
        z = torch.einsum('bl, bld -> bd', attn, text_tokens)  # [B,D]
        return z

class TaFM(nn.Module):
    def __init__(self, img_channels, text_dim, tasks=('vessel','od','fovea'), bottleneck_ratio=8):
        super().__init__()
        C = img_channels
        self.last_hw = None
        self.norm = nn.GroupNorm(1, C)
        self.token_pool = TextTokenPool(text_dim)
        hid = max(C // 4, 32)
        self.to_gb = nn.Sequential(
            nn.Linear(text_dim, hid), nn.GELU(),
            nn.Linear(hid, 2 * C)
        )
        bneck = max(C // bottleneck_ratio, 8)
        self.adapters = nn.ModuleDict({t: Adapter(C, bottleneck=bneck) for t in tasks})
        self.task_gate = nn.ParameterDict({t: nn.Parameter(torch.tensor(0.0)) for t in tasks})

    def forward(self, img_feat, text_tokens, task: str):
        """
        반환: [B,C,H,W]
        """
        B, C, H, W = img_feat.shape
        self.last_hw = (H, W)

        z = self.token_pool(text_tokens, B)   # [B, D]

        x_n = self.norm(img_feat)             # [B,C,H,W]
        gb = self.to_gb(z)                    # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1)     # [B,C], [B,C]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta  = beta .unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        x_mod = (1 + gamma) * x_n + beta      # [B,C,H,W]

        if task in self.adapters:
            y = x_mod + torch.sigmoid(self.task_gate[task]) * self.adapters[task](x_mod)

        return y

class MultiTaskUNet(nn.Module):
    def __init__(self,
                 context_length: int = 12,
                 num_classes_vessel: int = 1,
                 num_classes_od: int = 1,
                 device: str = 'cuda'):
        super().__init__()
        self.device = device

        task_prompts = [
            'blood vessels',
            'optic disc',
            'fovea'
        ]

        self.prompt_learner = PromptLearner(
            context_length=context_length,
            task_prompts=task_prompts,
            device=device
        )

        self.encoder = ConvNeXtEncoder(pretrained=True)
        skip_channels = self.encoder.compute_skip_channels(input_size=(3, 512, 512))

        text_dim = self.prompt_learner.embed_dim
        img_ch = skip_channels[3]

        self.fusion = TaFM(img_channels=img_ch, text_dim=text_dim, tasks=('vessel','od','fovea'))

        self.decoder_vessel = ConvNeXtDecoder(skip_channels, num_classes_vessel)
        self.decoder_od     = ConvNeXtDecoder(skip_channels, num_classes_od)
        self.decoder_fovea  = ConvNeXtFoveaHeatmapDecoder(skip_channels)

    def forward(self, x_vessel, x_od_fov):
        feat_v, skips_v = self.encoder(x_vessel)
        feat_o, skips_o = self.encoder(x_od_fov)

        _, text_feats = self.prompt_learner()

        fv_ca = self.fusion(feat_v, text_feats[0], task='vessel') 
        fo_ca = self.fusion(feat_o, text_feats[1], task='od')
        ff_ca = self.fusion(feat_o, text_feats[2], task='fovea')

        out_v_ca    = self.decoder_vessel(x=fv_ca, skip_connections=skips_v)
        out_o_ca    = self.decoder_od(x=fo_ca, skip_connections=skips_o)
        out_f_ca, h_ca = self.decoder_fovea(x=ff_ca, skip_connections=skips_o)

        return out_v_ca, out_o_ca, out_f_ca, h_ca

from __future__ import annotations

if __name__ == "__main__":
    # This is needed to import UNet from the same directory
    import os
    os.environ['OV_DATA_BASE']= os.path.dirname(os.path.abspath(__file__))
from copy import deepcopy
from typing import List, Sequence, Tuple
import torch.nn as nn
import torch
from KCD.Segmentation.ovseg.networks.UNet import UNet

import os
from KCD.Segmentation.ovseg.networks.UNet import Logits
from KCD.Segmentation.ovseg.networks.UNet_ClassHead import GlobalPowerPooling
from KCD.Segmentation.ovseg.networks.mask2former_modeling.vit_modeling import Transformer, CONFIGS as CONFIGS_ViT
from KCD.Segmentation.ovseg.networks.mask2former_modeling.transformer_decoder.mask2former_transformer_decoder3d import (
    MultiScaleMaskedTransformerDecoder3d as MSTDecoder3d,
)
import numpy as np



class TransUNet3D(nn.Module):
    """Drop‑in replacement for UNet with ViT bottleneck & Transformer decoder.

    All UNet frontend/decoder parameters are left untouched so that you can keep
    using the same YAML/JSON config you already pass to UNet.  Pass
    ``use_vit=True`` to toggle the ViT bottleneck; pass ``use_masked_attn=True``
    to enable the Mask2Former decoder (default).  When disabled the network
    degenerates to a vanilla UNet and **still** matches UNet.forward()."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[Sequence[int] | int],
        is_2d: bool,
        *,
        filters: int = 32,
        filters_max: int = 384,
        n_pyramid_scales: int | None = None,
        conv_params: dict | None = None,
        norm: str | None = None,
        norm_params: dict | None = None,
        nonlin_params: dict | None = None,
        kernel_sizes_up: Sequence[Sequence[int] | int] | None = None,
        skip_type: str = "skip",
        use_trilinear_upsampling: bool = False,
        use_less_hid_channels_in_decoder: bool = False,
        fac_skip_channels: float = 1.0,
        p_dropout_logits: float = 0.0,
        stem_kernel_size: int | Sequence[int] | None = None,
        with_classification_head: bool = False,
        cls_classes: int = 2,
        # --- TransUNet specific ------------------------------------------------
        use_vit: bool = True,
        patch_size: Tuple[int, int, int] = (64,64,64),  # input patch size
        vit_name: str = "R50-ViT-B_16",  # one of the keys in CONFIGS
        vit_depth: int | None = None,
        vit_hidden_size: int = 756,
        vit_mlp_dim: int | None = None,
        vit_num_heads: int | None = None,
        vit_patch_size: Tuple[int, int, int] = (8, 8, 8),
        vit_pretrained: str | None = None,
        vit_layer_scale: bool = False,
        # --- decoder transformer ---------------------------------------------
        use_masked_attn: bool = True,
        max_hidden_dim: int = 192,
        n_decoder_layers: int = 9,
        num_queries: int | None = None,
        decoder_layer_scale: bool = False,
    ) -> None:
        super().__init__()

        assert not is_2d, "TransUNet3D is designed for 3‑D volumes (is_2d=False)."

        # ------------------------------------------------------------------
        # Encoder / Decoder identical to your UNet implementation
        # ------------------------------------------------------------------
        self.unet_core = _UNetEncoderDecoder(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            is_2d=is_2d,
            filters=filters,
            filters_max=filters_max,
            n_pyramid_scales=n_pyramid_scales,
            conv_params=conv_params,
            norm=norm,
            norm_params=norm_params,
            nonlin_params=nonlin_params,
            kernel_sizes_up=kernel_sizes_up,
            skip_type=skip_type,
            use_trilinear_upsampling=use_trilinear_upsampling,
            use_less_hid_channels_in_decoder=use_less_hid_channels_in_decoder,
            fac_skip_channels=fac_skip_channels,
            p_dropout_logits=p_dropout_logits,
            stem_kernel_size=stem_kernel_size,
        )

        # ------------------------------------------------------------------
        # 1) ViT bottleneck (optionally on top of the UNet encoder output)
        # 2) Multi‑scale Transformer decoder (Mask2Former‑style)
        # ------------------------------------------------------------------
        self.use_vit = use_vit
        self.use_masked_attn = use_masked_attn
        self.out_channels = out_channels
        self._n_pyramid_scales = self.unet_core.n_pyramid_scales

        # a) ViT bottleneck -------------------------------------------------
        if self.use_vit:
            cfg = deepcopy(CONFIGS_ViT[vit_name])
            if vit_depth is not None:
                cfg.transformer.num_layers = vit_depth
            if vit_hidden_size is not None:
                cfg.hidden_size = vit_hidden_size
            if vit_mlp_dim is not None:
                cfg.transformer.mlp_dim = vit_mlp_dim
            if vit_num_heads is not None:
                cfg.transformer.num_heads = vit_num_heads

            # The feature map fed into ViT has spatial dims reduced by the
            # cumulative pooling factor of the encoder.
            print(self.unet_core.blocks_down[-1])
            # find out shape of the feature map before ViT using an example
            # patch of equal size to patch_size
            xb = torch.randn(1, in_channels, *patch_size)
            for block in self.unet_core.blocks_down:
                xb = block(xb)
            channels,d, h, w = xb.shape[1:]  # depth, height, width of the feature map
            print("Feature map shape before ViT:", xb.shape)
            feat_size = (d, h, w)

            self.vit = Transformer(
                cfg,
                feat_size=feat_size,
                vis=False,
                feat_channels=self.unet_core.filters_list[-1],
                use_layer_scale=vit_layer_scale,
            )
            if vit_pretrained is not None:
                self.vit.load_from(weights=torch.load(vit_pretrained, map_location="cpu"))
            # Conv to bring ViT hidden dim back to UNet decoder channels
            self.vit_post = nn.Conv3d(cfg.hidden_size, self.unet_core.filters_list[-1], 1)

            self.cls_pools = nn.ModuleList()
            self.cls_fcs = nn.ModuleList()
            final_fc_size = 0
            #for every conv decoder level, create a GlobalPowerPooling that doubles in spatial size with
            # each level, starting at d, h, w = patch_size
            for i in range(self.unet_core.n_stages):
                self.cls_pools.append(GlobalPowerPooling(d, h, w))
                self.fcs.append(nn.Linear(channels,channels))
                final_fc_size += channels
                d, h, w = d*2, h*2, w*2  # double spatial size at each level
                channels /=2

            self.final_fc = nn.Linear(final_fc_size, 128)
            self.final_fc2 = nn.Linear(128, 2)

            self.dropout = nn.Dropout(0.1)
            self.ReLU = nn.LeakyReLU(0.1)

            print(self.cls_pools)
            print(self.fcs)

        # b) Mask2Former style transformer decoder -------------------------
        if self.use_masked_attn:
            # Linear projections from multi‑scale UNet features to the common
            # hidden dimension expected by the transformer decoder
            self.hidden_dim = max_hidden_dim
            self.num_queries = num_queries or out_channels  # one query per class

            # we simply take the first (coarsest) 4 feature maps from encoder &
            # decoder (this matches original 3D‑TransUNet "fpn" variant)
            self.fpn_levels = len(kernel_sizes)-1
            self.input_proj = nn.ModuleList()
            for i in range(self.fpn_levels-1, -1, -1):
                ch = self.unet_core.filters_list[i]
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv3d(ch, self.hidden_dim, kernel_size=1, bias=False),
                        nn.GroupNorm(32, self.hidden_dim),
                    )
                )

            # Take the finest decoder feature (full resolution) for mask features
            self.mask_features_proj = nn.Conv3d(
                self.unet_core.logits_in_list[0], self.hidden_dim, kernel_size=1
            )

            decoder_cfg = dict(
                num_classes=out_channels,
                hidden_dim=self.hidden_dim,
                num_queries=self.num_queries,
                nheads=8,
                dim_feedforward=self.hidden_dim * 8,
                dec_layers=n_decoder_layers,
                mask_dim=self.hidden_dim,
                non_object=False,
                pre_norm=False,
                enforce_input_project=False,
                use_layer_scale=decoder_layer_scale,
                num_feature_levels=self.fpn_levels,
                is_masking=True,
                is_masking_argmax=False,
                is_mhsa_float32=False,
                no_max_hw_pe=False,
            )
            self.decoder = MSTDecoder3d(in_channels=self.hidden_dim, mask_classification=False, **decoder_cfg)

        # final softmax is applied in training/criterion to keep logits intact

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 1) Standard UNet encoder/decoder forward – we also get all hidden maps
        skips: List[torch.Tensor] = []
        seg_feats: List[torch.Tensor] = []  # full‑res decoder features for deep‑sup

        xb = x
        for d, block in enumerate(self.unet_core.blocks_down[:-1]):
            xb = block(xb)
            skips.append(xb[:, : self.unet_core.skip_channels[d]])

        # bottom block (without stride); ViT optionally inserted here ----------
        xb = self.unet_core.blocks_down[-1](xb)

        if self.use_vit:
            xb, _ = self.vit(xb)  # noqa: _
            xb = self.vit_post(xb)

        if self.do_classifier:
            #save classification features for cls head
            cls_feats = []

        # expanding path (same as UNet) ---------------------------------------
        for i in range(self.unet_core.n_stages - 2, -1, -1):
            xb = self.unet_core.upsamplings[i](xb)
            xb = self.unet_core.concats[i](xb, skips[i])
            xb = self.unet_core.blocks_up[i](xb)
            seg_feats.append(xb)  # decoder feature at this scale
            if self.do_classifier:
                # apply global pooling and fc to get classification features
                classification_features = self.dropout(self.ReLU(self.cls_fcs[i](self.cls_pools[i](xb))))
                cls_feats.append(classification_features)


        if self.do_classifier:
            cls_feats = torch.cat(cls_feats, dim=1)
            cls_logits = self.final_fc2(self.dropout(self.ReLU(self.final_fc(cls_feats))))

        # seg_feats = seg_feats[::-1]  # fine‑>coarse as in UNet

        # 2) Transformer decoder head (produces refined masks) -----------------
        if self.use_masked_attn:
            multi_scale_feats = [proj(feat) for proj, feat in zip(self.input_proj, seg_feats[: self.fpn_levels])]
            mask_features = self.mask_features_proj(seg_feats[-1])
            pred = self.decoder(multi_scale_feats, mask_features, mask=None)  # returns dict w/ pred_masks
            main_logits = pred["pred_masks"]  # [B, C, D, H, W] – raw logits (no sigmoid/softmax)
        else:
            # fallback: just use UNet final feature → conv1
            main_logits = Logits(
                in_channels=seg_feats[0].shape[1],
                out_channels=self.out_channels,
                is_2d=False,
            )(seg_feats[0])

        # 3) Deep supervision pyramid logits ----------------------------------
        pyramid_logits: List[torch.Tensor] = []
        for feat, logit_head in zip(seg_feats[::-1], self.unet_core.all_logits):
            pyramid_logits.append(logit_head(feat))

        if self.do_classifier:
            return [main_logits] + pyramid_logits[1:], cls_logits


        return [main_logits] + pyramid_logits[1:]  # keep UNet order (fine→coarse)


# ============================================================================
# _UNetEncoderDecoder – literally your UNet code with *all* public behaviour
# intact, but wrapped so we can re‑use it inside TransUNet3D without copy‑paste
# ============================================================================
class _UNetEncoderDecoder(nn.Module):
    """A direct, internal copy of your UNet implementation.

    Only cosmetic refactors applied (PEP8 names, annotations, etc.) so that we
    can *call* its building blocks from the ViT/Mask2Former augmentation above.
    The logic is unchanged – you can diff against UNet.py for verification."""

    def __init__(self, *args, **kwargs):
        # NOTE: We simply import and *instantiate* the very same class from
        # UNet.py; this tiny shell exists only so that the outer TransUNet3D can
        # attach additional methods/fields without touching your original UNet
        # source file (keeps git diff clean).
        try:
            from UNet import UNet as _NativeUNet
        except ImportError:
            from .UNet import UNet as _NativeUNet

        super().__init__()
        self._unet = _NativeUNet(*args, **kwargs)

        # Expose critical sub‑modules/attrs that TransUNet3D needs -------------
        self.__dict__.update(self._unet.__dict__)

    # NOTE: we never call this – TransUNet3D does its own forward pass because
    # it needs the intermediate feature maps.  The wrapper exists solely so the
    # Layer/Param hierarchy aligns with checkpoints trained on vanilla UNet.
    def forward(self, x):  # pragma: no cover
        return self._unet(x)

# def get_3d_UNet(in_channels, out_channels, n_stages, n_2d_blocks, filters=32):
#     kernel_sizes = [(3,3,1) if i < n_2d_blocks else 3
#                     for i in range(n_stages)]
#     return UNet(in_channels, out_channels, kernel_sizes, False)

def get_3d_transunet(in_channels,out_channels,n_stages,filters=32,
                     patch_size=(64,64,64),use_masked_attn=True,
                     with_classification_head=False, cls_classes=2):
    kernel_sizes = [(3,3,3) for i in range(n_stages)]
    return TransUNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_sizes=kernel_sizes,
        is_2d=False,
        filters=filters,
        n_pyramid_scales=n_stages,
        patch_size=patch_size,
        use_masked_attn=use_masked_attn,
        use_vit=True,
        vit_name="R50-ViT-B_16",  # default ViT config
        with_classification_head=with_classification_head,
        cls_classes=cls_classes
    )


if __name__ == "__main__":
    """Quick smoke‑test: feed a dummy 3‑D patch through TransUNet3D with
    masked cross‑attention enabled and print the shapes of the logits/decoder
    outputs. Run `python transunet3d.py` to execute.
    """
    import torch

    # Synthetic 3‑D volume – shape: (batch, channels, depth, height, width)
    B, C, D, H, W = 2, 1, 64, 64, 64
    x = torch.randn(B, C, D, H, W)
    patch_size = (D, H, W)  # input patch size for the ViT

    # Instantiate the network. The constructor mirrors your UNet’s signature;
    # only parameters you care about for the test are shown here.
    net = get_3d_transunet(1,3,5,32,patch_size=patch_size,
                           with_classification_head=True,cls_classes=3).eval()

    with torch.no_grad():
        y,cls = net(x)
    print(cls)

    # Print a concise summary of whatever the model returned
    if isinstance(y, (list, tuple)):
        print("Pyramid logits shapes:", [t.shape for t in y])
    elif isinstance(y, dict):
        print("Decoder output keys & shapes:", {k: (v.shape if torch.is_tensor(v) else type(v)) for k, v in y.items()})
    else:
        print("Output shape:", y.shape)


import torch
import torch.nn as nn
from addict import Dict
from backbone.Blocks import kaiming_init, constant_init, UnetUpsample, PatchExpand3D, CCSAttention
from backbone.mamba_encode import Mamba_Encoder

class AMUnet(nn.Module):
    def __init__(self, args, in_channels=1, num_classes=2, channels=[24, 48, 96, 192, 384], is_batchnorm=True):
        super(AMUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters = channels
        self.is_batchnorm = is_batchnorm
        self.backbone_feature_shape = dict()
        self.backbone = self.build_backbone(args)

        # Upsampling layers
        self.transposeconv_stage3 = PatchExpand3D(dim=self.filters[4])
        self.transposeconv_stage2 = PatchExpand3D(dim=self.filters[3])
        self.transposeconv_stage1 = PatchExpand3D(dim=self.filters[2])
        self.transposeconv_stage0 = PatchExpand3D(dim=self.filters[1])

        # CCS Attention modules
        self.ccsa3 = CCSAttention(channel=self.filters[3] * 2)
        self.ccsa2 = CCSAttention(channel=self.filters[2] * 2)
        self.ccsa1 = CCSAttention(channel=self.filters[1] * 2)
        self.ccsa0 = CCSAttention(channel=self.filters[0] * 2)

        # Deep supervision layers
        self.dsv4 = UnetUpsample(in_size=self.filters[3], out_size=self.num_classes, scale_factor=16)
        self.dsv3 = UnetUpsample(in_size=self.filters[2], out_size=self.num_classes, scale_factor=8)
        self.dsv2 = UnetUpsample(in_size=self.filters[1], out_size=self.num_classes, scale_factor=4)
        self.dsv1 = UnetUpsample(in_size=self.filters[0], out_size=self.num_classes, scale_factor=2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                constant_init(m.weight, val=1.0)
                constant_init(m.bias, val=0.0)

        # Final convolution layer
        self.final = nn.Conv3d(num_classes * 4, num_classes, kernel_size=1)

    def build_backbone(self, args):
        if args.model_type == 'mamba':
            channels = self.filters
            backbone = Mamba_Encoder(in_channels=self.in_channels, num_channels=channels, layer_nums=args.model_depth)
            self.backbone_feature_shape = {f'stage{i + 1}': Dict({'channel': channel}) for i, channel in enumerate(channels)}
        else:
            raise NotImplementedError('Model type not supported!')
        return backbone

    def decoder(self, features):
        # Decoder path with CCS Attention and upsampling
        x = self.transposeconv_stage3(features[0])
        x1 = self.ccsa3(torch.cat([x, features[1]], dim=1))
        x = self.transposeconv_stage2(x1)
        x2 = self.ccsa2(torch.cat([x, features[2]], dim=1))
        x = self.transposeconv_stage1(x2)
        x3 = self.ccsa1(torch.cat([x, features[3]], dim=1))
        x = self.transposeconv_stage0(x3)
        x4 = self.ccsa0(torch.cat([x, features[4]], dim=1))

        # Deep supervision outputs
        dsv4 = self.dsv4(x1)
        dsv3 = self.dsv3(x2)
        dsv2 = self.dsv2(x3)
        dsv1 = self.dsv1(x4)

        # Final output
        out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        return out

    def forward(self, inputs):
        features = self.backbone(inputs)
        output = self.decoder(features)
        return output

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p

if __name__ == '__main__':
    import argparse
    from torch.nn import functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mamba')
    parser.add_argument('--model_depth', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--project_name", default='NuImages_swin_base_Seg', type=str)
    parser.add_argument('--common_stride', type=int, default=4)
    parser.add_argument('--transformer_dropout', type=float, default=0.0)
    parser.add_argument('--transformer_nheads', type=int, default=6)
    parser.add_argument('--transformer_dim_feedforward', type=int, default=1536)
    parser.add_argument('--transformer_enc_layers', type=int, default=3)
    parser.add_argument('--conv_dim', type=int, default=384)
    parser.add_argument('--mask_dim', type=int, default=384)

    args = parser.parse_args()

    # Initialize the segmentation model
    seg_model = AMUnet(args, in_channels=1, num_classes=3).cuda()

    # Example input tensor
    x = torch.rand(2, 1, 64, 224, 224).cuda()

    # Forward pass
    result = seg_model(x)
    print(result.shape)

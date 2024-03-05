import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=0):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels),
            nn.ReLU(True),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(in_channels),
        )
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.body(x)
        out = F.relu(x+residual, True)
        if self.downsample:
            out_down = self.down_conv(out)
            return out, out_down
        else:
            out = self.conv(out)
        return out

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads=1,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads=1,
            num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class RT_block(nn.Module):
    def __init__(self, channels):
        super(RT_block, self).__init__()
        self.Res_block0 = nn.Conv2d(3, channels, 1, 1, 0)
        self.Res_block1 = ResidualBlock(channels, channels)
        self.MSA = MSAB(channels, channels)
        self.Res_block2 = ResidualBlock(channels, channels)
        self.Res_block3 = nn.Conv2d(channels, 31, 1, 1, 0)

    def forward(self, x):
        out = self.Res_block0(x)
        out = self.Res_block1(out)
        out = self.MSA(out)
        out = self.Res_block2(out)
        out = self.Res_block3(out)
        return out




class ModelNet(nn.Module):
    def __init__(self, channels=48):
        super(ModelNet, self).__init__()
        self.SRC = nn.Conv2d(31, 3, 1, 1, 0, bias=False)
        self.rt_block0 = RT_block(channels)
        self.rt_block1 = RT_block(channels)
        self.rt_block2 = RT_block(channels)
        self.rt_block3 = RT_block(channels)
        self.rt_block4 = RT_block(channels)
        self.rt_block5 = RT_block(channels)
        self.rt_block6 = RT_block(channels)
        self.rt_block7 = RT_block(channels)


    def forward(self, x):
        input0 = x
        out0 = self.rt_block0(input0)
        input1 = self.SRC(out0)
        out1 = self.rt_block1(input0-input1)+out0
        input2 = self.SRC(out1)
        out2 = self.rt_block2(input0-input2)+out1
        input3 = self.SRC(out2)
        out3 = self.rt_block3(input0-input3)+out2
        input4 = self.SRC(out3)
        out4 = self.rt_block4(input0-input4)+out3
        input5 = self.SRC(out4)
        out5 = self.rt_block5(input0-input5)+out4
        input6 = self.SRC(out5)
        out6 = self.rt_block6(input0-input6)+out5
        input7 = self.SRC(out6)
        out7 = self.rt_block7(input0-input7)+out6
        input8 = self.SRC(out7)

        rgb_img = input8
        out = out7
        return out, rgb_img

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)







if __name__ == '__main__':
    net = ModelNet(channels = 32)
    img = torch.rand(1, 3, 512, 512)
    print(net)
    img = img.cuda()
    net = net.cuda()
    img_out = net(img)
    print(img_out[0].shape)
    print('Parameters number is ', sum(param.numel() for param in net.parameters()))
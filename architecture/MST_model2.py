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
        self.Res_block0 = nn.Conv2d(31, channels, 1, 1, 0)
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


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ResidualDenseBlock_5C, self).__init__()
        # dense convolutions
        self.activation = nn.LeakyReLU(0.2, inplace = True)
        self.Res_block0 = nn.Conv2d(31, in_channels, 1, 1, 0)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.conv4 = nn.Conv2d(in_channels * 2, in_channels, kernel_size, stride, padding)
        self.conv5 = nn.Conv2d(in_channels * 2, in_channels, kernel_size, stride, padding)
        self.conv6 = nn.Conv2d(in_channels * 2, in_channels, kernel_size, stride, padding)
        self.MSA = MSAB(in_channels, in_channels)
        self.Res_block3 = nn.Conv2d(in_channels, 31, 1, 1, 0)


    def forward(self, x):
        x0 = self.Res_block0(x)
        x1 = self.activation(self.conv1(x0))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        msa = self.MSA(x3)
        x4 = self.activation(self.conv4(torch.cat((x3, msa), 1)))
        x5 = self.activation(self.conv5(torch.cat((x2, x4), 1)))
        x6 = self.activation(self.conv6(torch.cat((x1, x5), 1)))
        x6 = self.Res_block3(x6)
        return x6



class Degration_block(nn.Module):
    def __init__(self, channels):
        super(Degration_block, self).__init__()
        self.rt_block = ResidualDenseBlock_5C(channels)
        self.fusion_conv = nn.Conv2d(31*2, 31, 3, 1, 1)

    def forward(self, x, y, res, sigma):
        b,c,h,w = x.shape
        x_down = torch.bmm(res, x.view(b,c,h*w))
        Ex = y - x_down
        res = torch.transpose(res, 1, 2)
        Dx = torch.bmm(res, Ex)
        Dx = Dx.view(b,c,h,w)
        # out = self.rt_block(x - sigma*Dx)
        out = self.rt_block(self.fusion_conv(torch.concat([x, Dx], 1)))
        return out



class ModelNet(nn.Module):
    def __init__(self, channels=48, iter=8):
        super(ModelNet, self).__init__()
        self.sigma = 1
        self.iter = iter
        self.sh_conv = nn.Conv2d(3, 31, 3, 1, 1)
        self.backbone = self.make_layer(Degration_block, channels, self.iter)


    def forward(self, x, res):
        b,c,h,w = x.shape
        rgb = x
        rgb = rgb.view(b,c,h*w)
        x = self.sh_conv(x)
        out_list = []
        for i in range(self.iter):
            x = self.backbone[i](x, rgb, res, self.sigma)
        return x

    def make_layer(self, block, channels, block_num):
        layer = []
        layer.append(block(channels))
        for _ in range(block_num-1):
            layer.append(block(channels))
        return nn.Sequential(*layer)







if __name__ == '__main__':
    net = ModelNet(channels = 48)
    img = torch.rand(1, 3, 512, 512)
    res = torch.rand(1, 3, 31)
    # print(net)
    img = img.cuda()
    net = net.cuda()
    res = res.cuda()
    img_out = net(img, res)
    print(img_out.shape)
    print('Parameters number is ', sum(param.numel() for param in net.parameters()))

import torch 
from torch import Tensor, nn
from einops import rearrange
from dataclasses import dataclass


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int = 3072, double: bool = True):
        super().__init__()
        self.multiplier = 6
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        print(f"{out[0].shape=}")
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:])
        )


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


class DoubleStreamBlock(nn.Module):
    def __init__(
        self, 
        hidden_size: int = 3072, 
        num_heads: int  = 24, 
        mlp_ratio: float = 4.0, 
        qkv_bias: bool = True,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)  # three [b, 1, hidden_dim] tensors per mod
        txt_mod1, txt_mod2 = self.txt_mod(vec)  # three [b, 1, hidden_dim] tensors per mod

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

"""
img: torch.Size([1, 4080, 3072])
txt: torch.Size([1, 512, 3072])
vec: torch.Size([1, 3072])
pe: torch.Size([1, 1, 4592, 64, 2, 2])
"""
b, s, dim = 1, 4080, 3072
img_in_dim, txt_in_dim = 4080, 512
img = torch.randn(b, img_in_dim, dim, dtype=torch.float, device='cuda')
txt = torch.randn(b, txt_in_dim, dim, dtype=torch.float, device='cuda')
vec = torch.randn(b, dim, dtype=torch.float, device='cuda')
pe = torch.randn(b, 1, (img_in_dim + txt_in_dim), 64, 2, 2, dtype=torch.float, device='cuda')

layer = DoubleStreamBlock().cuda()
# layer(img=img, txt=txt, vec=vec, pe=pe)


def simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe):
    multiplier = 6
    dim = 3072
    num_heads = 24
    head_dim = dim // num_heads

    # Modules and Weights
    lin1 = nn.Linear(dim, 6 * dim, bias=True).cuda()
    lin2 = nn.Linear(dim, 6 * dim, bias=True).cuda()
    norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda() 
    norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    img_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda()
    txt_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda()
    img_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda()
    txt_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda()
    img_proj = nn.Linear(dim, dim).cuda()
    txt_proj = nn.Linear(dim, dim).cuda()
    q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    k_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    q_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    k_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()


    # three [b, 1, hidden_dim] tensors per mod tensor
    out_img = lin1(nn.functional.silu(vec))[:, None, :].chunk(multiplier, dim=-1)
    img_mod1, img_mod2 = out_img[:3], out_img[3:]  
    img_modulated = norm1(img)
    img_modulated = (1 + img_mod1[1]) * img_modulated + img_mod1[0]

    img_qkv = img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)

    # RMS norm
    rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
    img_q = (img_q * rrms) * q_img_rms_norm_scale
    rrms = torch.rsqrt(torch.mean(img_k**2, dim=-1, keepdim=True) + 1e-6)
    img_k = (img_k * rrms) * k_img_rms_norm_scale


    print(f"{img_q.shape=}, {img_k.shape=}")

    # prepare txt for attention
    out_txt = lin2(nn.functional.silu(vec))[:, None, :].chunk(multiplier, dim=-1)
    txt_mod1, txt_mod2 = out_txt[:3], out_txt[3:]  
    txt_modulated = txt_norm1(txt)
    txt_modulated = (1 + txt_mod1[1]) * txt_modulated + txt_mod1[0]
    txt_qkv = txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)
    rrms = torch.rsqrt(torch.mean(txt_q**2, dim=-1, keepdim=True) + 1e-6)
    txt_q = (txt_q * rrms) * q_txt_rms_norm_scale
    rrms = torch.rsqrt(torch.mean(txt_k**2, dim=-1, keepdim=True) + 1e-6)
    txt_k = (txt_k * rrms) * k_txt_rms_norm_scale

    # Run actual attention
    q = torch.cat((txt_q, img_q), dim=2) # torch.Size([1, 24, 4592, 128])
    k = torch.cat((txt_k, img_k), dim=2) # torch.Size([1, 24, 4592, 128])
    v = torch.cat((txt_v, img_v), dim=2)

    # Attention
    # attn = attention(q, k, v, pe=pe)
    attn = torch.randn(b, (txt_in_dim + img_in_dim), dim, dtype=q.dtype, device=q.device) 
    txt_attn, img_attn = attn[:, : txt_in_dim], attn[:, txt_in_dim :]

    # calculate the img bloks
    img = img + img_mod1[2] * img_proj(img_attn)
    img = img + img_mod2[2] * img_mlp((1 + img_mod2[1]) * norm2(img) + img_mod2[0])

    # calculate the txt bloks
    txt = txt + txt_mod1[2] * txt_proj(txt_attn)
    txt = txt + txt_mod2[2] * txt_mlp((1 + txt_mod2[1]) * txt_norm2(txt) + txt_mod2[0])

    return img, txt

simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe)


def pre_attn_img_portion(img=img, txt=txt, vec=vec, pe=pe):
    multiplier = 6
    dim = 3072
    num_heads = 24
    head_dim = dim // num_heads

    # Modules and Weights
    lin1 = nn.Linear(dim, 6 * dim, bias=True).cuda()
    lin2 = nn.Linear(dim, 6 * dim, bias=True).cuda()
    norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda() 
    norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    img_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda()
    txt_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda()
    img_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda()
    txt_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda()
    img_proj = nn.Linear(dim, dim).cuda()
    txt_proj = nn.Linear(dim, dim).cuda()
    q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    k_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    q_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
    k_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()        

    # GEMMs
    out_img = lin1(nn.functional.silu(vec))[:, None, :].chunk(multiplier, dim=-1)
    img_mod1, img_mod2 = out_img[:3], out_img[3:] 
    out_txt = lin2(nn.functional.silu(vec))[:, None, :].chunk(multiplier, dim=-1)
    txt_mod1, txt_mod2 = out_txt[:3], out_txt[3:]  
    
    # Kernel 1
    img_modulated = (1 + img_mod1[1]) * norm1(img) + img_mod1[0]
    txt_modulated = (1 + txt_mod1[1]) * txt_norm1(txt) + txt_mod1[0]

    img_qkv = img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)
    txt_qkv = txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)

    # kernel 2
    rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
    img_q = (img_q * rrms) * q_img_rms_norm_scale
    rrms = torch.rsqrt(torch.mean(img_k**2, dim=-1, keepdim=True) + 1e-6)
    img_k = (img_k * rrms) * k_img_rms_norm_scale
    rrms = torch.rsqrt(torch.mean(txt_q**2, dim=-1, keepdim=True) + 1e-6)
    txt_q = (txt_q * rrms) * q_txt_rms_norm_scale
    rrms = torch.rsqrt(torch.mean(txt_k**2, dim=-1, keepdim=True) + 1e-6)
    txt_k = (txt_k * rrms) * k_txt_rms_norm_scale

    # Run actual attention
    q = torch.cat((txt_q, img_q), dim=2) # torch.Size([1, 24, 4592, 128])
    k = torch.cat((txt_k, img_k), dim=2) # torch.Size([1, 24, 4592, 128])
    v = torch.cat((txt_v, img_v), dim=2)

    # Attention
    # attn = attention(q, k, v, pe=pe)
    attn = torch.randn(b, (txt_in_dim + img_in_dim), dim, dtype=q.dtype, device=q.device) 
    txt_attn, img_attn = attn[:, : txt_in_dim], attn[:, txt_in_dim :]

    # calculate the img bloks
    img_proj_out = img_proj(img_attn)
    txt_proj_out = txt_proj(txt_attn)

    # kernel 3
    img = img + img_mod1[2] * img_proj_out
    img_mlp_in = (1 + img_mod2[1]) * norm2(img) + img_mod2[0]
    img = img + img_mod2[2] * img_mlp(img_mlp_in)


    txt = txt + txt_mod1[2] * txt_proj_out
    txt_mlp_in = (1 + txt_mod2[1]) * txt_norm2(txt) + txt_mod2[0]
    txt = txt + txt_mod2[2] * txt_mlp(txt_mlp_in)

    return img, txt


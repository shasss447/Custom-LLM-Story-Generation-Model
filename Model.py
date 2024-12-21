import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tk_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.dp_emb=nn.Dropout(cfg["drop_rate"])

        self.trnsf_bck=nn.Sequential(*[TransformerBlock(cfg)for _ in range(cfg["n_layers"])])
        self.fnl_norm=LayerNorm(cfg["emb_dim"])
        self.out_hd=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        _,seq_len=in_idx.shape
        tkn_emb=self.tk_emb(in_idx)
        pos_emb=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tkn_emb+pos_emb
        x=self.dp_emb(x)
        x=self.trnsf_bck(x)
        x=self.fnl_norm(x)
        lgts=self.out_hd(x)
        return lgts
    

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att=MultiheadAttention(d_in=cfg["emb_dim"],d_out=cfg["emb_dim"],ctxt_len=cfg["context_length"],num_hd=cfg["n_heads"],dp=cfg["drop_rate"],qkv_bias=cfg["qkv_bias"])
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.dp_scut=nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        stct=x
        x=self.norm1(x)
        x=self.att(x)
        x=self.dp_scut(x)
        x=x+stct

        stct=x
        x=self.norm2(x)
        x=self.ff(x)
        x=self.dp_scut(x)
        x=x+stct

        return x

class LayerNorm(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.eps=1e-5
        self.sc=nn.Parameter(torch.ones(emb_dim))
        self.st=nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self,x):
        mn=x.mean(dim=-1,keepdim=True)
        var=x.var(dim=-1,keepdim=True,unbiased=False)
        norm_x=(x-mn)/torch.sqrt(var+self.eps)

        return self.sc*norm_x+self.st


class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"]),
        )

    def forward(self,x):
        return self.layers(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class MultiheadAttention(nn.Module):
    def __init__(self,d_in,d_out,ctxt_len,dp,num_hd,qkv_bias=False):
        super().__init__()
        assert d_in%num_hd==0
        self.d_out=d_out
        self.num_hd=num_hd
        self.hd_dim=d_out//num_hd
        self.W_q=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v=nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj=nn.Linear(d_out,d_out)
        self.drpout=nn.Dropout(dp)
        self.register_buffer('mask',torch.triu(torch.ones(ctxt_len,ctxt_len),diagonal=1))

    def forward(self,x):
        b,num_tk,_=x.shape
        k=self.W_k(x)
        q=self.W_q(x)
        v=self.W_v(x)

        k=k.view(b,num_tk,self.num_hd,self.hd_dim)
        v=v.view(b,num_tk,self.num_hd,self.hd_dim)
        q=q.view(b,num_tk,self.num_hd,self.hd_dim)

        k=k.transpose(1,2)
        q=q.transpose(1,2)
        v=v.transpose(1,2)
        
        attn_sc=q@k.transpose(2,3)
        msk_bool=self.mask.bool()[:num_tk,:num_tk]
        attn_sc.masked_fill_(msk_bool,-torch.inf)
        attn_wt=torch.softmax(attn_sc/k.shape[-1]**0.5,dim=-1)
        attn_wt=self.drpout(attn_wt)
        cntxt_vec=(attn_wt@v).transpose(1,2)
        cntxt_vec=cntxt_vec.contiguous().view(b,num_tk,self.d_out)
        cntxt_vec=self.out_proj(cntxt_vec)
        
        return cntxt_vec
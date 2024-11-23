import re
import tiktoken
import torch
import torch.nn as nn


def preprocess_lyrics(text):

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\b(\w+)(,?\s+\1)+\b", r"\1", text)
    text = re.sub(r"(\b\w+-\w+\b)(,?\s+\1)+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s*\.\s*", ". ", text)
    text = text.strip()
    
    lines = text.split("\n")
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line not in unique_lines:
            unique_lines.append(line)
    cleaned_text = " ".join(unique_lines)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = cleaned_text.lower()
    
    return cleaned_text


# with open("lyrics.txt", "r",encoding="utf-8") as file:
#     raw_lyrics = file.read()

# cleaned_lyrics = preprocess_lyrics(raw_lyrics)

# with open("cleaned_lyrics.txt", "w") as file:
#     file.write(cleaned_lyrics)


from torch.utils.data import Dataset,DataLoader

class CMDataset(Dataset):
    def __init__(self,txt,tokenizer,mx_len,stride):
        self.in_id=[]
        self.tgt_id=[]

        tkn_id=tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        for i in range(0,len(tkn_id)-mx_len,stride):
            in_ch=tkn_id[i:i+mx_len]
            tgt_ch=tkn_id[i+1:i+mx_len+1]
            self.in_id.append(torch.tensor(in_ch))
            self.tgt_id.append(torch.tensor(tgt_ch))
        
    def __len__(self):
        return len(self.in_id)

    def __getitem__(self, idx):
        return self.in_id[idx],self.tgt_id[idx]

def create_dataloader_v1(txt,bt_sz=4,mx_len=256,stride=128,shuffle=True,dp_lst=True,num_wk=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dtst=CMDataset(txt,tokenizer,mx_len,stride)

    datlod=DataLoader(dtst,batch_size=bt_sz,shuffle=shuffle,drop_last=dp_lst,num_workers=num_wk)
    
    return datlod


with open("cleaned_lyrics.txt", "r", encoding="utf-8") as file:
    data = file.read()
vocb_sz=50257
ot_dim=256
ctxt_len=4
tkn_emb_lay=torch.nn.Embedding(vocb_sz,ot_dim)
pos_emb_lay=torch.nn.Embedding(ctxt_len,ot_dim)

dtld=create_dataloader_v1(data,bt_sz=8,mx_len=4,stride=4,shuffle=False)

for bt in dtld:
    x,y=bt
    tkn_emb=tkn_emb_lay(x)
    pos_emb=pos_emb_lay(torch.arange(ctxt_len))
    it_emb=tkn_emb+pos_emb
    break



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
        b,num_tk,d_in=x.shape
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

# torch.manual_seed(123)
# mh=MultiheadAttention(ot_dim,ot_dim,ctxt_len,0.0,2)
# ctvc=mh(it_emb)
# print(ctvc.shape)

cofig_124M={
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class CactusModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tk_emb=nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb=nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.dp_emb=nn.Dropout(cfg["drop_rate"])

        self.trnsf_bck=nn.Sequential(*[TransformerBlock(cfg)for _ in range(cfg["n_layers"])])
        self.fnl_norm=LayerNorm(cfg["emb_dim"])
        self.out_hd=nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        bt_sz,seq_len=in_idx.shape
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

def gen_txt(model,idx,mx_n_tk,ctxt_sz):
    for _ in range(mx_n_tk):
        idx_c=idx[:,-ctxt_sz:]

        with torch.no_grad():
          lgt=model(idx_c)
        lgt=lgt[:,-1,:]
        pro=torch.softmax(lgt,dim=-1)
        id_n=torch.argmax(pro,dim=-1,keepdim=True)
        idx=torch.cat((idx,id_n),dim=1)
    return idx



torch.manual_seed(123)
model=CactusModel(cofig_124M)

st_ct="hello, i am"
tokenizer = tiktoken.get_encoding("gpt2")
enc=tokenizer.encode(st_ct)
enct=torch.tensor(enc).unsqueeze(0)

model.eval()

ot=gen_txt(model=model,idx=enct,mx_n_tk=6,ctxt_sz=cofig_124M["context_length"])
dct=tokenizer.decode(ot.squeeze(0).tolist())
print(dct)
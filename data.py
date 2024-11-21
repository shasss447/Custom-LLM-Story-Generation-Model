import re
import tiktoken
import torch
from torch import nn as nn
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
dt_itr=iter(dtld)
it,ot=next(dt_itr)

tkn_emb=tkn_emb_lay(it)
pos_emb=pos_emb_lay(torch.arange(ctxt_len))

it_emb=tkn_emb+pos_emb



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
        self.register_buffer('mask',torch.triu(torch.ones(ctxt_len,ctxt_len,diagonal=1)))

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
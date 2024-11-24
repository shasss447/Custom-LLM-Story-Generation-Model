import re
import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

tokenizer = tiktoken.get_encoding("gpt2")

def preprocess_lyrics(text):

    text=re.sub(r"\[.*?\]", "", text)
    text=re.sub(r"[^\x00-\x7F]+", "", text)
    text=re.sub(r"\(.*?\)", "", text)
    text=re.sub(r"\b(\w+)(,?\s+\1)+\b", r"\1", text)
    text=re.sub(r"(\b\w+-\w+\b)(,?\s+\1)+", r"\1", text)
    text=re.sub(r"\s+", " ", text)
    text=re.sub(r"\s+,", ",", text)
    text=re.sub(r",\s*,", ",", text)
    text=re.sub(r"\s*\.\s*", ". ", text)
    text=text.strip()
    text=re.sub(r'\b(\w+)( \1\b){' + str(2) + r',}', r'\1 ' * 2, text)
    
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

class CMDataset(Dataset):
    def __init__(self,text,mx_len,stride):
        self.in_id=[]
        self.tgt_id=[]

        tkn_id=tokenizer.encode(text,allowed_special={"<|endoftext|>"})

        for i in range(0,len(tkn_id)-mx_len,stride):
            in_ch=tkn_id[i:i+mx_len]
            tgt_ch=tkn_id[i+1:i+mx_len+1]
            self.in_id.append(torch.tensor(in_ch))
            self.tgt_id.append(torch.tensor(tgt_ch))
        
    def __len__(self):
        return len(self.in_id)

    def __getitem__(self, idx):
        return self.in_id[idx],self.tgt_id[idx]

def create_dataloader(txt,bt_sz=4,mx_len=256,stride=128,shuffle=True,dp_lst=True,num_wk=0):
    
    dtst=CMDataset(txt,mx_len,stride)

    datlod=DataLoader(dtst,batch_size=bt_sz,shuffle=shuffle,drop_last=dp_lst,num_workers=num_wk)
    
    return datlod

def text_to_token_ids(text):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def plot_loss(ep_sn,tk_sn,trn_ls,valn_ls):
    fig,ax1=plt.subplots(figsize=(5,3))
    ax1.plot(ep_sn,trn_ls,label="Training loss")
    ax1.plot(ep_sn,valn_ls,linestyle="-.",label="Validation loss")
    ax1.set_xlabel=("Epochs")
    ax1.set_ylabel=("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2=ax1.twiny()
    ax2.plot(tk_sn,trn_ls,alpha=0)
    ax2.set_xlabel("Token seen")

    fig.tight_layout()
    plt.savefig("loss-plot.png")
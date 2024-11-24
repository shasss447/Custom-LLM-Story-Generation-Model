import torch
from data import text_to_token_ids,token_ids_to_text

def generate_text(model,idx,mx_n_tk,ctxt_sz,temp=0.0,t_k=None,eos_id=None):
    for _ in range(mx_n_tk):
        idx_c=idx[:,-ctxt_sz:]

        with torch.no_grad():
          lgt=model(idx_c)
        lgt=lgt[:,-1,:]
        if t_k is not None:
            tp_lgt,_=torch.topk(lgt,t_k)
            mn_val=tp_lgt[:,-1]
            lgt=torch.where(lgt<mn_val,torch.tensor(float("-inf")).to(lgt.device),lgt)
        
        if temp>0.0:
            lgt=lgt/temp
            pro=torch.softmax(lgt,dim=-1)
            id_n=torch.multinomial(pro,num_samples=1)
        else:
            id_n=torch.argmax(lgt,dim=-1,keepdim=True)
        
        if id_n==eos_id:
            break

        idx=torch.cat((idx,id_n),dim=1)
    return idx

def generate_print(model,device,st_ctx,top_k,temp):
    model.eval()
    ct_sz=model.pos_emb.weight.shape[0]
    enc=text_to_token_ids(st_ctx).to(device)
    with torch.no_grad():
        tk_id=generate_text(model=model,idx=enc,mx_n_tk=10,ctxt_sz=ct_sz,temp=temp,t_k=top_k)
    dc_tk=token_ids_to_text(tk_id)
    print(dc_tk.replace("\n"," "))
    model.train()
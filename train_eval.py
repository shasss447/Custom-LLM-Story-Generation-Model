import torch
from generation import generate_print

def train_model(model, tr_loader, val_loader, opti, device, num_ep, ev_fq, ev_itr, st_ctx, top_k, temp):
    trn_loss, valn_loss, trck_tkn_seen = [], [], []
    tokn_seen, glob_step = 0, -1

    for epoch in range(num_ep):
        model.train()
        for it_bt, tg_bt in tr_loader:
            opti.zero_grad()
            loss = cal_loss_bt(it_bt, tg_bt, model, device)
            loss.backward()
            opti.step()
            tokn_seen += it_bt.numel()
            glob_step += 1

            if glob_step % ev_fq == 0:
                tr_loss, val_loss = eval_model(model, tr_loader, val_loader, device, ev_itr)
                trn_loss.append(tr_loss)
                valn_loss.append(val_loss)
                trck_tkn_seen.append(tokn_seen)
                print(f"Ep {epoch+1} (Step {glob_step:06d}): "
                      f"Train loss {tr_loss:.3f}, Val loss {val_loss:.3f}")
                generate_print(model,device,st_ctx,top_k,temp)
                
    return trn_loss, valn_loss, trck_tkn_seen


def eval_model(model,tr_load,val_load,device,ev_itr):
    model.eval()
    with torch.no_grad():
        tr_loss=cal_loss_ld(tr_load,model,device,nm_bt=ev_itr)
        val_loss=cal_loss_ld(val_load,model,device,nm_bt=ev_itr)
    model.train()
    return tr_loss,val_loss

def cal_loss_bt(it_bt,tg_bt,model,device):
    it_bt,tg_bt=it_bt.to(device),tg_bt.to(device)
    lgt=model(it_bt)
    loss=torch.nn.functional.cross_entropy(lgt.flatten(0,1),tg_bt.flatten())
    return loss

def cal_loss_ld(dt_ld,model,device,nm_bt=None):
    tt_loss=0.
    if len(dt_ld)==0:
        return float("nan")
    elif nm_bt is None:
        nm_bt=len(dt_ld)
    else:
        nm_bt=min(nm_bt,len(dt_ld))
    for i,(it_bt,tg_bt) in enumerate(dt_ld):
        if i<nm_bt:
            loss=cal_loss_bt(it_bt,tg_bt,model,device)
            tt_loss+=loss.item()
        else:
            break
    return tt_loss/nm_bt
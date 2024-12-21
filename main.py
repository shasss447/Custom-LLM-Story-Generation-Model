import torch
from Model import Model
from config import Model_Config,Settings
from data import preprocess_data,create_dataloader,plot_loss
from train_eval import train_model

def main(Model_Config,Settings):
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # with open("raw_story.txt", "r",encoding="utf-8") as file:
    #  data = file.read()
    # clean_data = preprocess_data(data)
    # with open("story.txt", "w") as file:
    #  file.write(clean_data)

    with open("story.txt", "r", encoding="utf-8") as file:
      data = file.read()
    model=Model(Model_Config)
    model.to(device)
    opti=torch.optim.AdamW(model.parameters(),lr=Settings["learning_rate"],weight_decay=Settings["weight_decay"])

    tr_rat=0.9
    spt_idx=int(tr_rat*len(data))
    
    tr_loader = create_dataloader(
    txt=data[:spt_idx],
    bt_sz=Settings["batch_size"],
    mx_len=Model_Config["context_length"],
    stride=Model_Config["context_length"],
    dp_lst=True,
    shuffle=True,
    num_wk=0)

    val_loader = create_dataloader(
    txt=data[spt_idx:],
    bt_sz=Settings["batch_size"],
    mx_len=Model_Config["context_length"],
    stride=Model_Config["context_length"],
    dp_lst=False,
    shuffle=False,
    num_wk=0)


    tr_loss,val_loss,tk_sn=train_model(model,tr_loader,val_loader,opti,device,num_ep=Settings["num_epochs"],ev_fq=5,ev_itr=1,st_ctx="every efforts moves you",top_k=25,temp=1)

    return tr_loss,val_loss,tk_sn,model

if __name__=="__main__":
    tr_loss,val_loss,tk_sn,model=main(Model_Config,Settings)
    epc_tn=torch.linspace(0,Settings["num_epochs"],len(tr_loss))
    plot_loss(epc_tn,tk_sn,tr_loss,val_loss)

    torch.save(model.state_dict(),"model.pth")

    # model=Model(Model_Config)
    # model.load_state_dict(torch.load("model.pth"),weigths_only=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = tiktoken.get_encoding("gpt2")
    # tokens = text_to_token_ids("the incident is going to", tokenizer).to(device)
    # generated_tokens = generate_text(model, tokens, mx_n_tk=50, ctxt_sz=Model_Config["context_length"], temp=1, t_k=25)
    # output_text = token_ids_to_text(generated_tokens)
    # print(output_text.replace("\n"," "))
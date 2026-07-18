import numpy as np, torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
def evaluate(model,loader,criterion,device,fake_index):
    model.eval(); loss=0; ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); z=model(x); loss+=criterion(z,y).item()*x.size(0); q=torch.softmax(z,1); p=z.argmax(1); ys+=y.cpu().tolist(); ps+=p.cpu().tolist(); probs+=q[:,fake_index].cpu().tolist()
    y=np.array(ys); p=np.array(ps); f=(y==fake_index).astype(int)
    return {'loss':loss/len(loader.dataset),'accuracy':accuracy_score(y,p),'precision_fake':precision_score(y,p,pos_label=fake_index,zero_division=0),'recall_fake':recall_score(y,p,pos_label=fake_index,zero_division=0),'f1_fake':f1_score(y,p,pos_label=fake_index,zero_division=0),'roc_auc_fake':roc_auc_score(f,np.array(probs)),'confusion_matrix':confusion_matrix(y,p).tolist()}

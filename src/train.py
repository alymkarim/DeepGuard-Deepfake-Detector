import json,tempfile,torch
from pathlib import Path
from torch import nn
from .config import settings
from .gcs import download_prefix,upload_file,upload_text
from .dataset import make_loaders
from .model import create_model,freeze_backbone,unfreeze_final_blocks
from .evaluate import evaluate

def epoch(m,l,c,o,d):
    m.train(); loss=correct=total=0
    for x,y in l:
        x,y=x.to(d),y.to(d); o.zero_grad(set_to_none=True); z=m(x); q=c(z,y); q.backward(); o.step(); loss+=q.item()*x.size(0); correct+=(z.argmax(1)==y).sum().item(); total+=y.size(0)
    return {'loss':loss/total,'accuracy':correct/total}
def main():
    settings.validate(); d=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device',d)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); data=root/'frames'
        for s in ('train','validation','test'): download_prefix(settings.bucket,f"{settings.frame_prefix}/{s}",data/s)
        tr,va,te,map_=make_loaders(data,settings.image_size,settings.batch_size,settings.num_workers); fake=map_['fake']; m=create_model(True).to(d); freeze_backbone(m); c=nn.CrossEntropyLoss(); best=1e9; ck=root/'best_model.pth'; hist=[]
        def run(n,lr,stage):
            nonlocal best,hist
            o=torch.optim.AdamW(filter(lambda p:p.requires_grad,m.parameters()),lr=lr,weight_decay=1e-4)
            for i in range(n):
                a=epoch(m,tr,c,o,d); v=evaluate(m,va,c,d,fake); hist.append({'stage':stage,'epoch':i+1,'train':a,'validation':v}); print(hist[-1])
                if v['loss']<best: best=v['loss']; torch.save({'architecture':'efficientnet_b0','model_state_dict':m.state_dict(),'class_to_idx':map_,'image_size':settings.image_size},ck)
        run(settings.epochs,settings.learning_rate,'classifier'); unfreeze_final_blocks(m); run(settings.fine_tune_epochs,settings.fine_tune_learning_rate,'fine_tune')
        cp=torch.load(ck,map_location=d); m.load_state_dict(cp['model_state_dict']); test=evaluate(m,te,c,d,fake); metrics=root/'metrics.json'; metrics.write_text(json.dumps({'history':hist,'test':test,'class_to_idx':map_},indent=2)); upload_file(settings.bucket,ck,f"{settings.output_prefix}/best_model.pth"); upload_file(settings.bucket,metrics,f"{settings.output_prefix}/metrics.json"); upload_text(settings.bucket,json.dumps(map_),f"{settings.output_prefix}/class_to_idx.json"); print(test)
if __name__=='__main__': main()

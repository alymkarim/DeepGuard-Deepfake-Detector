from torchvision import datasets,transforms
from torch.utils.data import DataLoader
def make_loaders(root,size,batch,workers):
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    tr=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(5),transforms.ToTensor(),norm])
    ev=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),norm])
    d1=datasets.ImageFolder(root/'train',transform=tr); d2=datasets.ImageFolder(root/'validation',transform=ev); d3=datasets.ImageFolder(root/'test',transform=ev)
    kw=dict(batch_size=batch,num_workers=workers,pin_memory=True)
    return DataLoader(d1,shuffle=True,**kw),DataLoader(d2,shuffle=False,**kw),DataLoader(d3,shuffle=False,**kw),d1.class_to_idx

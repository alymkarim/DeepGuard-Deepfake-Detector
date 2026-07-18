import random, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import settings
from .gcs import list_video_blobs, upload_text

def split_paths(paths,seed,train_ratio,validation_ratio,test_ratio):
    train,rem=train_test_split(paths,train_size=train_ratio,random_state=seed,shuffle=True)
    val_frac=validation_ratio/(validation_ratio+test_ratio)
    val,test=train_test_split(rem,train_size=val_frac,random_state=seed,shuffle=True)
    return sorted(train),sorted(val),sorted(test)

def rows(paths,label,index): return [{"gcs_blob":p,"label":label,"class_index":index,"video_id":Path(p).stem} for p in paths]
def main():
    settings.validate(); random.seed(settings.seed)
    real=list_video_blobs(settings.bucket,settings.raw_real_prefix); fake=list_video_blobs(settings.bucket,settings.raw_fake_prefix)
    n=min(len(real),len(fake),settings.max_videos_per_class)
    if n<2: raise RuntimeError(f"Not enough videos: real={len(real)} fake={len(fake)}")
    real=random.sample(real,n); fake=random.sample(fake,n)
    rs=split_paths(real,settings.seed,settings.train_ratio,settings.validation_ratio,settings.test_ratio)
    fs=split_paths(fake,settings.seed,settings.train_ratio,settings.validation_ratio,settings.test_ratio)
    summary={}
    for name,i in zip(("train","validation","test"),range(3)):
        data=rows(rs[i],"real",1)+rows(fs[i],"fake",0); random.Random(settings.seed).shuffle(data); df=pd.DataFrame(data)
        upload_text(settings.bucket,df.to_csv(index=False),f"{settings.manifest_prefix}/{name}.csv")
        summary[name]={"videos":len(df),"real":int((df.label=='real').sum()),"fake":int((df.label=='fake').sum())}
        print(name,summary[name])
    upload_text(settings.bucket,json.dumps(summary,indent=2),f"{settings.manifest_prefix}/split_summary.json")
if __name__=='__main__': main()

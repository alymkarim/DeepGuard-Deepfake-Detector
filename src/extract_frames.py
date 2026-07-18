import tempfile
from pathlib import Path
from io import StringIO
import cv2, numpy as np, pandas as pd
from google.cloud import storage
from tqdm import tqdm
from .config import settings

def extract(video,out,n):
    out.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(str(video)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); saved=[]
    if not cap.isOpened() or total<=0: return saved
    for i,idx in enumerate(np.linspace(int(total*.1),max(int(total*.9)-1,0),n,dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx)); ok,frame=cap.read()
        if ok:
            p=out/f"frame_{i:03d}.jpg"
            if cv2.imwrite(str(p),frame,[cv2.IMWRITE_JPEG_QUALITY,95]): saved.append(p)
    cap.release(); return saved

def process(split):
    c=storage.Client(); b=c.bucket(settings.bucket); text=b.blob(f"{settings.manifest_prefix}/{split}.csv").download_as_text(); df=pd.read_csv(StringIO(text)); n=settings.train_frames_per_video if split=='train' else settings.eval_frames_per_video
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        for r in tqdm(df.itertuples(index=False),total=len(df),desc=split):
            v=td/'video'/Path(r.gcs_blob).name; v.parent.mkdir(parents=True,exist_ok=True); b.blob(r.gcs_blob).download_to_filename(str(v))
            for p in extract(v,td/'frames'/r.label/r.video_id,n): b.blob(f"{settings.frame_prefix}/{split}/{r.label}/{r.video_id}/{p.name}").upload_from_filename(str(p))
def main():
    settings.validate()
    for s in ('train','validation','test'): process(s)
if __name__=='__main__': main()

from pathlib import Path
from google.cloud import storage
VIDEO_EXTENSIONS={".mp4",".avi",".mov",".mkv"}
def list_video_blobs(bucket,prefix):
    return sorted(b.name for b in storage.Client().list_blobs(bucket,prefix=prefix) if Path(b.name).suffix.lower() in VIDEO_EXTENSIONS)
def upload_file(bucket,source,dest): storage.Client().bucket(bucket).blob(dest).upload_from_filename(str(source))
def upload_text(bucket,text,dest): storage.Client().bucket(bucket).blob(dest).upload_from_string(text)
def download_prefix(bucket,prefix,local_root):
    out=[]
    for b in storage.Client().list_blobs(bucket,prefix=prefix):
        if b.name.endswith('/'): continue
        rel=Path(b.name).relative_to(prefix); dst=local_root/rel; dst.parent.mkdir(parents=True,exist_ok=True); b.download_to_filename(str(dst)); out.append(dst)
    return out

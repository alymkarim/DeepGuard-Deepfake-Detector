import os
from google.cloud import aiplatform
p=os.environ['GCP_PROJECT_ID']; r=os.getenv('GCP_REGION','europe-west1'); b=os.environ['GCS_BUCKET']; image=os.environ['TRAIN_IMAGE_URI']
aiplatform.init(project=p,location=r,staging_bucket=f'gs://{b}/staging')
job=aiplatform.CustomContainerTrainingJob(display_name='deepfake-efficientnet-training',container_uri=image)
job.run(replica_count=1,machine_type=os.getenv('VERTEX_MACHINE_TYPE','n1-standard-8'),accelerator_type=os.getenv('VERTEX_ACCELERATOR_TYPE','NVIDIA_TESLA_T4'),accelerator_count=1,environment_variables={'GCP_PROJECT_ID':p,'GCP_REGION':r,'GCS_BUCKET':b,'SEED':'42','BATCH_SIZE':'32','EPOCHS':'5','FINE_TUNE_EPOCHS':'3'},base_output_dir=f'gs://{b}/vertex-output',sync=True)

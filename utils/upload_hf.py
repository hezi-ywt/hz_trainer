import os
import tarfile
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
from hfutils.operate import upload_file_to_file, upload_directory_as_archive, upload_directory_as_directory
from dotenv import load_dotenv


load_dotenv()

repo_id = 'heziiiii/lu2_lightning_test'

local_file = '/mnt/hz_trainer/checkpoints7/checkpoint-epoch_0_step_500.safetensors'
file_in_repo = 'checkpoints7/checkpoint-epoch_0_step_500.safetensors'
upload_file_to_file(
    local_file=local_file,
    repo_id=repo_id,
    file_in_repo=file_in_repo,
    repo_type="model",
    hf_token=os.environ.get("HF_TOKEN")
)
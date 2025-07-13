from huggingface_hub import HfApi
import os

def upload_folder():
    api = HfApi()
    repo_id = "Pingsz/fine-tuned"
    checkpoint_path = "./checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)

    api.upload_folder(
        folder_path=checkpoint_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload checkpoint",
        token=os.environ["HF_TOKEN"]
    )

import torch
import subprocess
import os
from pathlib import Path

def run(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result

def setup_flash_attention():
    if torch.cuda.get_device_capability()[0] >= 8:
        try:
            subprocess.run(["pip", "uninstall", "-y", "flash-attn"], check=False)

            repo_url = "https://github.com/Dao-AILab/flash-attention.git"
            clone_dir = Path("flash-attention")
            if not clone_dir.exists():
                run(f"git clone {repo_url}")

            run("pip install packaging ninja", cwd=clone_dir)
            run("pip install .", cwd=clone_dir)

            attn_implementation = "flash_attention_2"
            torch_dtype = torch.bfloat16

        except Exception as e:
            print(f"⚠️ Failed to install flash-attn from source: {e}")
            attn_implementation = "eager"
            torch_dtype = torch.float16

    else:
        print("⚠️ GPU compute capability < 8.0 — falling back to eager attention.")
        attn_implementation = "eager"
        torch_dtype = torch.float16

    return attn_implementation, torch_dtype

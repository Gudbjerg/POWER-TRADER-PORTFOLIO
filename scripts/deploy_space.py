"""
Upload the hf-deploy tree to the Hugging Face Space, without binary files.

The Space rejects binary files that are not stored through Xet, and hf-deploy carries
the README screenshots; the Space needs none of them. Run from a checkout of hf-deploy
with HF_TOKEN set. Usage: python -m scripts.deploy_space "<commit message>"
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

SPACE = "TobGud/power-trader-portfolio"


def main(message: str) -> int:
    from huggingface_hub import HfApi

    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    numstat = subprocess.run(["git", "diff", "--numstat", "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "HEAD"],
                             capture_output=True, text=True, check=True).stdout
    binary = {line.split("\t", 2)[2] for line in numstat.splitlines() if line.startswith("-\t-\t")}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(f"git archive HEAD | tar -x -C {tmp}", shell=True, check=True)
        for f in binary:
            path = os.path.join(tmp, f)
            if os.path.exists(path):
                os.remove(path)
        api = HfApi(token=os.environ["HF_TOKEN"])
        info = api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=tmp, commit_message=message)
    print(f"deployed {info.oid} ({len([f for f in files if f]) - len(binary)} files, {len(binary)} binary left out)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "Deploy from hf-deploy"))

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    HF_TOKEN     = os.getenv("HF_TOKEN")
    HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "YOUR_USERNAME/propml-models")

    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        print("  export HF_TOKEN=hf_your_token_here")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{HF_MODEL_REPO}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    models_dir = Path("models")
    files_to_upload = [
        "models_xgb_ensemble.pkl",
        "models_lgb_ensemble.pkl",
        "models_cat_ensemble.pkl",
        "ensemble_weights.pkl",
        "feature_list.pkl",
        "version.txt",
        "combined_engineered.parquet",
        "combined_cleaned.parquet",
    ]

    for filename in files_to_upload:
        if filename.endswith(".parquet"):
            if "engineered" in filename:
                local_path = Path("data/features") / filename
            else:
                local_path = Path("data/cleaned") / filename
        else:
            local_path = models_dir / filename
        if not local_path.exists():
            print(f"  Skipping (not found): {filename}")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
        except Exception as e:
            print(f"  Failed {filename}: {e}")

    # Also upload target_encoding_map.json — needed by API at inference
    enc_path = Path("data/features/target_encoding_map.json")
    if enc_path.exists():
        try:
            api.upload_file(
                path_or_fileobj=str(enc_path),
                path_in_repo="target_encoding_map.json",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            print("  Uploaded: target_encoding_map.json")
        except Exception as e:
            print(f"  Failed target_encoding_map.json: {e}")

    print(f"\nAll uploads complete.")
    print(f"View: https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
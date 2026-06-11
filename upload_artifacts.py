from huggingface_hub import HfApi
import os

api = HfApi()

artifacts_dir = "reports/artifacts"

for filename in os.listdir(artifacts_dir):
    filepath = os.path.join(artifacts_dir, filename)
    if os.path.isfile(filepath):
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=f"reports/artifacts/{filename}",
            repo_id="nandininautiyal/customer-intent-engine",
            repo_type="space",
        )
        print(f"Done: {filename}")

print("All artifacts uploaded successfully.")
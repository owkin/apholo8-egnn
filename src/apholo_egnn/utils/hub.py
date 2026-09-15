"""HuggingFace Hub download utilities."""

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

logger = logging.getLogger(__name__)

HF_REPO_ID = "owkin/apholo-egnn"
HF_FILENAME = "model.safetensors"


def download_from_pretrained(
    repo_id: str = HF_REPO_ID,
    filename: str = HF_FILENAME,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download (or retrieve from cache) a checkpoint from the Hub.

    Returns the local path to the cached file.
    """
    logger.debug("Fetching '%s' from Hub repo '%s'", filename, repo_id)
    rev = f"@{revision}" if revision else ""
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as e:
        raise FileNotFoundError(
            f"'{filename}' not found in repo '{repo_id}{rev}'"
            " - not in local cache and network access is disabled."
        ) from e
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            raise PermissionError(
                f"Authentication failed for repo '{repo_id}'. "
                "Set a valid token via the HF_TOKEN environment variable "
                "or run `hf auth login`."
            ) from e
        if e.response is not None and e.response.status_code == 404:
            raise FileNotFoundError(
                f"'{filename}' not found in repo '{repo_id}{rev}' on the Hugging Face Hub."
            ) from e
        raise

    logger.debug("Checkpoint ready at %s", cached_path)
    return Path(cached_path)

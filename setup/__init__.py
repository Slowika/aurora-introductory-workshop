"""Init env vars for session without third-party dotenv."""

import os
from pathlib import Path

env_path = Path(".env")
if not env_path.exists():
    msg = "No .env file found in current directory."
    raise FileNotFoundError(msg)
with env_path.open() as f:
    for line in f.readlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split("=", 1)
        os.environ[name] = value.strip()

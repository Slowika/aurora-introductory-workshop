"""Init temporary (session) environment variables from .env without third-party dotenv.

This file is used to inject environment variables set in the .env file in the project
root (i.e. aurora-introductory-workshop/.env) into the session's os.environ dictionary,
serving as temporary environment variables with no clean-up required.

This will be run when any code below /setup/ is imported.
"""

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

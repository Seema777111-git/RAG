"""DeepEval-based evaluation."""

import os

# Runs before any submodule imports `deepeval`, so telemetry stays off unless the user opts in.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")  # DeepEval would otherwise copy .env (including empty lines) into os.environ

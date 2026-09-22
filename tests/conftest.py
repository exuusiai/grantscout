import os

# Keep tests hermetic regardless of the developer's .env: the CI workflow pins
# the same defaults. setdefault preserves deliberately exported overrides.
os.environ.setdefault("GRANTSCOUT_USE_MODEL_REASONING", "false")
os.environ.setdefault("GRANTSCOUT_RETRIEVAL_MODE", "lexical")
os.environ.setdefault("GRANTSCOUT_USE_RERANKER", "false")

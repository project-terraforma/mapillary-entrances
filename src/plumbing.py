#!/usr/bin/env python3
# Thin wrapper to keep backward-compat entrypoint.
try:
    from .cli_plumbing import main  # package mode: python -m src.plumbing
except Exception:  # script mode: python src/plumbing.py
    from cli_plumbing import main

if __name__ == "__main__":
    main()

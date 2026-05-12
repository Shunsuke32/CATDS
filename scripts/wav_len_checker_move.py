#!/usr/bin/env python3
"""
Deprecated compatibility wrapper.
Use scripts/wav_len_checker.py with --mode move instead.
"""
import subprocess
import sys


def main():
    cmd = [sys.executable, "scripts/wav_len_checker.py", "--mode", "move"] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail
ROOT=/weka/gfaria/primebeaker/examples/frontier-science-live-search
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$ROOT/deps:/weka/gfaria/jtc/.venv/lib/python3.12/site-packages"
mkdir -p /tmp/result
/weka/gfaria/experiments/reverse-text-online-20261013/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12 "/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/grade.py" > "/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/grade.log" 2>&1

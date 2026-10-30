"""Apply the bundled MC32 patches only to their recorded source contents."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("prime-rl", "renderers", "verifiers", "vllm"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parent
    manifest = json.loads((recipe / "manifest.json").read_text())
    component = manifest["components"][args.component]
    patch = recipe / f"{args.component}.patch"
    if sha256(patch) != component["patch_sha256"]:
        raise ValueError(f"MC32 patch checksum mismatch: {patch}")
    source = args.source.resolve()
    for relative, hashes in component["files"].items():
        path = source / relative
        expected = hashes["before_sha256"]
        if expected is None:
            if path.exists():
                raise ValueError(f"MC32 new file already exists: {path}")
        elif not path.is_file() or sha256(path) != expected:
            raise ValueError(f"MC32 source differs from pinned {args.component}: {path}")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=source, check=True)
    for relative, hashes in component["files"].items():
        if sha256(source / relative) != hashes["after_sha256"]:
            raise ValueError(f"MC32 patched source checksum mismatch: {relative}")
    print(f"Applied MC32 {args.component} patch ({component['revision']})", flush=True)


if __name__ == "__main__":
    main()

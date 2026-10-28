"""Generate missing localhost credentials without displaying or replacing existing keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from secrets import token_urlsafe


def prepare_local_env(path: Path) -> tuple[str, ...]:
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = content.splitlines(keepends=True)
    values = {}
    for index, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            values[name.strip()] = (index, value.strip())
    generators = {
        "AGENTOPS_POSTGRES_PASSWORD": lambda: token_urlsafe(36),
        "AGENTOPS_API_KEYS": lambda: f"{token_urlsafe(36)}=local-admin:admin",
        "AGENTOPS_SESSION_SECRET": lambda: token_urlsafe(48),
    }
    changed = []
    for name, generate in generators.items():
        existing = values.get(name)
        if (
            existing
            and existing[1]
            and not (
                name == "AGENTOPS_SESSION_SECRET"
                and existing[1].strip("\"'") == "agentops-local-session-change-me-2026"
            )
        ):
            continue
        replacement = f"{name}={generate()}\n"
        if existing:
            lines[existing[0]] = replacement
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(replacement)
        changed.append(name)
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return tuple(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(".env"))
    result = prepare_local_env(parser.parse_args().path)
    print("Prepared configuration names: " + (", ".join(result) or "already configured"))


if __name__ == "__main__":
    main()

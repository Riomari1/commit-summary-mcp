"""Commit Summary MCP server implementation."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

DEFAULT_RECENT_LIMIT = 5
MAX_RECENT_LIMIT = 20
MAX_DIFF_LINES = 900
MAX_DIFF_CHARS = 20_000
DIFF_TRUNCATION_NOTICE = "[diff truncated to keep the summary focused]"
DEFAULT_MODEL = os.getenv("COMMIT_SUMMARY_MODEL", "gpt-oss:20b")
OLLAMA_HOST = (
    os.getenv("COMMIT_SUMMARY_OLLAMA_HOST")
    or os.getenv("OLLAMA_HOST")
    or "http://127.0.0.1:11434"
)

_REPO_ROOT: Path | None = None
_OLLAMA_CLIENT: httpx.Client | None = None


def _get_repo_root() -> Path:
    """Locate the target git repository once and reuse it."""
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT

    override = os.getenv("COMMIT_SUMMARY_REPO")
    start = Path(override).expanduser() if override else Path.cwd()
    candidate = start.resolve()
    while True:
        if (candidate / ".git").exists():
            _REPO_ROOT = candidate
            return candidate
        if candidate.parent == candidate:
            raise RuntimeError(
                "Could not find a git repository. "
                "Start the server inside a repo or set COMMIT_SUMMARY_REPO=/path/to/repo."
            )
        candidate = candidate.parent


def _get_ollama_client() -> httpx.Client:
    """Initialize the Ollama HTTP client on demand."""
    global _OLLAMA_CLIENT
    if _OLLAMA_CLIENT is None:
        _OLLAMA_CLIENT = httpx.Client(
            base_url=OLLAMA_HOST.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
    return _OLLAMA_CLIENT


async def _run_git(args: list[str]) -> tuple[int, str, str]:
    """Execute a git command inside the selected repository."""

    def _execute() -> tuple[int, str, str]:
        result = subprocess.run(
            ["git", *args],
            cwd=_get_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr

    return await asyncio.to_thread(_execute)


def _clean_diff(raw: str) -> str:
    """Remove binary blobs/noise and trim the diff to a safe size."""
    cleaned: list[str] = []
    seen_binary_notice = False
    for line in raw.splitlines():
        if "\x00" in line:
            continue
        stripped = line.rstrip("\r")
        if stripped.startswith(("Binary files", "GIT binary patch")):
            if not seen_binary_notice:
                cleaned.append("[binary changes omitted]")
                seen_binary_notice = True
            continue
        if len(stripped) > 1200:
            stripped = stripped[:1200] + " …"
        cleaned.append(stripped)
        if len(cleaned) >= MAX_DIFF_LINES:
            cleaned.append(DIFF_TRUNCATION_NOTICE)
            break

    text = "\n".join(cleaned).strip()
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS] + f"\n{DIFF_TRUNCATION_NOTICE}"
    return text


def _hash_error(message: str, hint: str | None = None) -> dict[str, Any]:
    """Standard JSON error payload for summarize_diff."""
    return {
        "error": {
            "message": message,
            "hint": hint
            or "Use get_recent_commits() to grab a valid hash or double-check the one you provided.",
        }
    }


async def _summarize_with_model(diff_text: str, commit_hash: str) -> str:
    """Send the cleaned diff to the local Ollama model."""

    def _call() -> str:
        client = _get_ollama_client()
        prompt = (
            "You are a senior engineer who writes crisp, plain language summaries of git diffs. "
            "Explain what changed, why it likely changed, and note risks or follow-up work when inferable."
            f"\n\nSummarize commit {commit_hash}. Stay under three short paragraphs and do not repeat raw diff hunks."
            f"\n\nDiff:\n{diff_text}"
        )
        response = client.post(
            "/api/generate",
            json={
                "model": DEFAULT_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            },
        )
        response.raise_for_status()
        payload = response.json()
        summary = payload.get("response", "")
        return summary.strip()

    return await asyncio.to_thread(_call)


server = FastMCP(
    name="Commit Summary MCP",
    instructions=(
        "Tools that inspect the current git repository to list recent commits and summarize their diffs."
        " Requires access to git CLI and a local Ollama model for summaries."
    ),
)


@server.tool(
    description="Return the most recent commits in the local repository. Limit defaults to 5 and caps at 20."
)
async def get_recent_commits(limit: int = DEFAULT_RECENT_LIMIT) -> list[dict[str, str]]:
    """Fetch recent commits with author, ISO date, and the first line of the message."""
    if limit <= 0:
        limit = DEFAULT_RECENT_LIMIT
    limit = min(limit, MAX_RECENT_LIMIT)

    format_spec = "%h%x1f%an%x1f%ad%x1f%s"
    args = [
        "--no-pager",
        "log",
        f"-n{limit}",
        f"--pretty=format:{format_spec}",
        "--date=iso-strict",
    ]
    code, stdout, stderr = await _run_git(args)
    if code != 0:
        raise RuntimeError(f"git log failed: {stderr.strip() or 'unknown error'}")

    commits: list[dict[str, str]] = []
    for line in stdout.strip().splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        commit_hash, author, date, subject = parts
        commits.append(
            {
                "hash": commit_hash.strip(),
                "author": author.strip(),
                "date": date.strip(),
                "message": subject.strip(),
            }
        )
    return commits


@server.tool(
    description="Summarize the intent and impact of a commit by analyzing its diff.",
)
async def summarize_diff(commit_hash: str) -> dict[str, Any]:
    """Generate a short natural-language summary for the provided commit hash."""
    candidate = commit_hash.strip()
    if not candidate:
        return _hash_error("Commit hash is required.")

    code, _, stderr = await _run_git(["rev-parse", "--verify", f"{candidate}^{{commit}}"])
    if code != 0:
        return _hash_error(
            message=stderr.strip() or f"Git could not find commit '{candidate}'.",
            hint="Pass a hash returned by get_recent_commits().",
        )

    _, short_hash, _ = await _run_git(["rev-parse", "--short", candidate])
    short_hash = short_hash.strip() or candidate

    show_args = [
        "--no-pager",
        "show",
        "--patch",
        "--stat",
        "--unified=5",
        "--no-ext-diff",
        "--format=medium",
        candidate,
    ]
    code, raw_diff, stderr = await _run_git(show_args)
    if code != 0:
        return _hash_error(stderr.strip() or "Unable to read the diff for that commit.")

    cleaned = _clean_diff(raw_diff)
    if not cleaned:
        return {
            "commit": short_hash,
            "summary": "Git did not produce a textual diff for this commit.",
        }

    try:
        summary = await _summarize_with_model(cleaned, short_hash)
    except httpx.HTTPError as exc:
        return _hash_error(
            message=f"Ollama request failed: {exc}",
            hint="Confirm ollama serve is running and COMMIT_SUMMARY_MODEL is pulled.",
        )
    except Exception as exc:  # pragma: no cover - defensive path
        return _hash_error(f"Unexpected summarizer failure: {exc}")

    if not summary:
        summary = "The model returned an empty response."

    return {"commit": short_hash, "summary": summary}


async def _run_server() -> None:
    """Entry point used by both CLI and MCP client launchers."""
    await server.run_stdio_async()


def main() -> None:
    """Sync-friendly entry point for the console script."""
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()

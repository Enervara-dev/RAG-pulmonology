"""
Interactive entry point for the GraphRAG medical assistant.

Drives the full query loop:
    memory load → gatekeeper/analyzer → routing → vector retrieval + rerank →
    entity extraction → graph traversal → answer generation → memory update.

Usage:
    python run_graphrag.py                          # interactive chat (REPL)
    python run_graphrag.py --query "fever and cough for 3 days"   # one-shot
    python run_graphrag.py --session-id alice       # name the session (memory key)
    python run_graphrag.py --user-id alice          # enable episodic memory (if configured)
    python run_graphrag.py --quiet                  # hide stage logs, show only the answer
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Run from the project root so `import graphrag.*` / `Memory_Layer.*` resolve.
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Windows TLS bootstrap — route Python's ssl through the OS trust store so
# Pinecone/Neo4j/Gemini calls verify correctly behind a corporate proxy/AV/VPN
# (fixes "unable to get local issuer certificate"). Must run BEFORE any client
# import. Falls back to certifi if truststore isn't installed.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass

# Log lines + answers contain non-ASCII (e.g. "→", emoji); force UTF-8 so the
# cp1252 Windows console doesn't raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

EXIT_WORDS = {"exit", "quit", "q", ":q", "bye"}


def _configure_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "run_graphrag.log", encoding="utf-8"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG medical assistant")
    parser.add_argument("--query", default=None,
                        help="Run a single query and exit (non-interactive).")
    parser.add_argument("--session-id", default="cli-session",
                        help="Session id — the memory key for this conversation (default: cli-session).")
    parser.add_argument("--user-id", default=None,
                        help="User id — enables the episodic (long-term) memory layer if configured.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress pipeline stage logs; print only answers.")
    args = parser.parse_args()

    _configure_logging(args.quiet)
    logger = logging.getLogger("run_graphrag")

    # Fail fast on missing config before constructing clients or making paid calls.
    from graphrag.config.settings import ConfigError, settings
    try:
        settings.validate_required("cli")
    except ConfigError as e:
        logger.error(str(e))
        sys.exit(2)

    from graphrag.pipeline.graphrag_pipeline import GraphRAGPipeline

    try:
        pipeline = GraphRAGPipeline()
    except Exception as e:
        logger.error("Failed to start the pipeline: %s", e)
        sys.exit(1)

    try:
        if args.query:
            _run_once(pipeline, args.query, args.session_id, args.user_id, logger)
            return

        _print_banner(args.session_id, args.user_id)
        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not query:
                continue
            if query.lower() in EXIT_WORDS:
                print("Goodbye.")
                break

            _run_once(pipeline, query, args.session_id, args.user_id, logger)
    finally:
        pipeline.close()


def _run_once(pipeline, query: str, session_id: str, user_id: str | None, logger) -> None:
    """Run one query. The answer streams back as NDJSON blocks — print each line
    as it arrives (one JSON block object per line)."""
    import json

    try:
        for block in pipeline.run(query_text=query, session_id=session_id, user_id=user_id):
            print(json.dumps(block, ensure_ascii=False), flush=True)
    except Exception:
        logger.exception("Query failed — the session is preserved; you can try again.")


def _print_banner(session_id: str, user_id: str | None) -> None:
    print("=" * 72)
    print("  Enervera GraphRAG — interactive medical assistant")
    print(f"  session: {session_id}" + (f"   user: {user_id}" if user_id else ""))
    print("  Type your question. 'exit' / 'quit' / Ctrl-C to leave.")
    print("=" * 72)


if __name__ == "__main__":
    main()

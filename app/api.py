"""
api.py — Flask application with async routes and advanced RAG pipeline integration.

Routes:
  GET  /                    — Serve the web UI
  GET  /api/status          — Check if a URL is indexed
  POST /api/initialize      — Index a new page
  POST /api/query           — Run the RAG pipeline and return an answer
  GET  /api/metrics         — Return runtime metrics
  DELETE /api/reset         — Clear the index and cache
"""

import asyncio
import logging
import os
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from app.services.storage import store_manager
from app.services.cache import query_cache
from app.utils.chunks import process_elements
from app.RAG.graph import rag_app
from app.RAG.states import GraphState
from app.evaluation.metrics import metrics_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

_ENABLE_CACHE = os.getenv("ENABLE_SEMANTIC_CACHE", "true").lower() == "true"


# ── Helper: run async in current or new event loop ────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    url = request.args.get("url", "")
    is_init = store_manager.is_initialized() and (
        not url or store_manager.is_url_indexed(url)
    )
    return jsonify({
        "initialized": is_init,
        "stats": store_manager.stats(),
    })


@app.route("/api/initialize", methods=["POST"])
def initialize():
    """
    Index a web page.
    Accepts: { url, title, elements: [{type, content, ...}] }
    """
    t0 = time.perf_counter()
    try:
        data     = request.json or {}
        url      = data.get("url", "")
        title    = data.get("title", "Untitled Page")
        elements = data.get("elements", [])

        logger.info("Initialize: url=%s | elements=%d", url, len(elements))

        # Check if already indexed
        if store_manager.is_url_indexed(url):
            logger.info("URL already indexed: %s", url)
            return jsonify({"status": "already_indexed", "url": url})

        # Process elements into chunks (async)
        chunks = _run_async(process_elements(url, title, elements))
        logger.info("Chunks produced: %d", len(chunks) if chunks else 0)

        # Store in vector stores (async)
        _run_async(store_manager.add_documents(chunks, url=url))

        # Invalidate cache since index changed
        query_cache.invalidate()

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("Initialization complete in %dms", elapsed_ms)

        return jsonify({
            "status": "success",
            "chunks_indexed": len(chunks) if chunks else 0,
            "elapsed_ms": elapsed_ms,
        })

    except Exception as e:
        import traceback
        logger.error("Initialization error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/query", methods=["POST"])
def query():
    """
    Run the RAG pipeline for a user query.
    Accepts: { prompt: str }
    Returns: { answer, retrieval_strategy, faithfulness_score, retrieval_trace, cached, ... }
    """
    if not store_manager.is_initialized():
        return jsonify({"error": "No page indexed. Please index a page first."}), 400

    t0 = time.perf_counter()
    data = request.json or {}
    user_prompt = (data.get("prompt") or "").strip()

    if not user_prompt:
        return jsonify({"error": "Empty query."}), 400

    logger.info("Query: %s", user_prompt[:120])

    # ── Semantic cache lookup ─────────────────────────────────────────────────
    if _ENABLE_CACHE:
        cached = _run_async(
            query_cache.get(user_prompt, store_manager.index_version)
        )
        if cached:
            cached["cached"] = True
            cached["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            metrics_tracker.record_query(cached, cached=True)
            return jsonify(cached)

    # ── Run the RAG graph ─────────────────────────────────────────────────────
    initial_state = GraphState(
        question=user_prompt,
        retry_count=0,
        blocked=False,
        retrieval_trace=[],
        guardrail_flags=[],
        node_timings={},
        search_queries=[user_prompt],
    )

    try:
        final_state = _run_async(rag_app.ainvoke(initial_state))
    except Exception as e:
        import traceback
        logger.error("RAG pipeline error: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    strategy   = final_state.get("retrieval_strategy", "hybrid")
    analysis   = final_state.get("query_analysis", {})

    result = {
        "answer":              final_state.get("answer", ""),
        "retrieval_strategy":  strategy,
        "query_intent":        analysis.get("intent", "factual"),
        "search_queries":      final_state.get("search_queries", [user_prompt]),
        "faithfulness_score":  final_state.get("faithfulness_score", 1.0),
        "retrieval_trace":     final_state.get("retrieval_trace", []),
        "guardrail_flags":     final_state.get("guardrail_flags", []),
        "node_timings":        final_state.get("node_timings", {}),
        "latency_ms":          elapsed_ms,
        "cached":              False,
    }

    # ── Cache the result ──────────────────────────────────────────────────────
    if _ENABLE_CACHE and not final_state.get("blocked"):
        _run_async(query_cache.set(user_prompt, result, store_manager.index_version))

    metrics_tracker.record_query(result, cached=False)
    logger.info(
        "Query complete: strategy=%s | faith=%.2f | %dms",
        strategy,
        result["faithfulness_score"],
        elapsed_ms,
    )
    return jsonify(result)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return runtime performance metrics."""
    return jsonify({
        "query_metrics": metrics_tracker.summary(),
        "cache_stats":   query_cache.stats(),
        "index_stats":   store_manager.stats(),
    })


@app.route("/api/reset", methods=["DELETE"])
def reset():
    """Clear the index and cache (useful for testing)."""
    from app.services import storage as _s
    _s.store_manager.__init__()  # Re-initialise in place
    query_cache.invalidate()
    metrics_tracker.reset()
    logger.info("Index and cache reset.")
    return jsonify({"status": "reset"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

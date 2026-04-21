#!/usr/bin/env python3
# ruff: noqa: I001
"""Generic tool for probing an OpenAI-compatible embedding endpoint."""

import argparse
import json
import math
import os
import urllib.error
import urllib.request


DEFAULT_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://192.168.3.91:8081")
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0")
DEFAULT_TIMEOUT = 10.0
TEST_VECTORS: list[tuple[str, str]] = [
    (
        "firmware_a",
        "GPIO interrupt controller handles edge-triggered events "
        "and updates mask registers.",
    ),
    (
        "firmware_b",
        "GPIO interrupt service routine clears pending flags "
        "and prioritizes edge events.",
    ),
    (
        "dsp_fft",
        "FFT pipeline computes spectrum magnitude, applies windowing, "
        "and estimates dominant frequency bins.",
    ),
    (
        "finance",
        "Quarterly revenue guidance was revised after margin compression "
        "in the enterprise subscription business.",
    ),
]


def _build_embeddings_url(base_url: str, endpoint: str | None) -> str:
    if endpoint:
        return endpoint
    return f"{base_url.rstrip('/')}/v1/embeddings"


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=False))
    return dot_product / (left_norm * right_norm)


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=False)))


def _print_vector_summary(label: str, vector: list[float]) -> None:
    first_values = [round(value, 6) for value in vector[:5]]
    print(
        f"  {label}: dim={len(vector)}, norm={_vector_norm(vector):.4f}, "
        f"first5={first_values}"
    )


def _print_pairwise_analysis(labels: list[str], embeddings: list[list[float]]) -> bool:
    print("\nPairwise comparison:")
    similarities: list[tuple[str, str, float, float]] = []
    for left_index, left_label in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right_label = labels[right_index]
            cosine = _cosine_similarity(
                embeddings[left_index],
                embeddings[right_index],
            )
            distance = _euclidean_distance(
                embeddings[left_index],
                embeddings[right_index],
            )
            similarities.append((left_label, right_label, cosine, distance))
            print(
                f"  {left_label} vs {right_label} -> "
                f"cosine={cosine:.4f}, l2={distance:.4f}"
            )

    if not similarities:
        return False

    most_similar = max(similarities, key=lambda item: item[2])
    least_similar = min(similarities, key=lambda item: item[2])
    cosine_spread = most_similar[2] - least_similar[2]

    print("\nContrast summary:")
    print(
        f"  Most similar: {most_similar[0]} vs {most_similar[1]} "
        f"(cosine={most_similar[2]:.4f})"
    )
    print(
        f"  Least similar: {least_similar[0]} vs {least_similar[1]} "
        f"(cosine={least_similar[2]:.4f})"
    )
    print(f"  Cosine spread: {cosine_spread:.4f}")

    return cosine_spread > 0.05


def probe_embedding_service(
    *,
    base_url: str = DEFAULT_BASE_URL,
    endpoint: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> bool:
    """Send one batch embedding request and print contrast analysis."""
    url = _build_embeddings_url(base_url, endpoint)
    labels = [label for label, _ in TEST_VECTORS]
    payload = {
        "model": model,
        "input": [text for _, text in TEST_VECTORS],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        print("Probing embedding service...")
        print(f"  URL: {url}")
        print(f"  Model: {model}")
        print(f"  Test vectors: {', '.join(labels)}")

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        response_data = sorted(data["data"], key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in response_data]

        print("\nConnection successful")
        for label, embedding in zip(labels, embeddings, strict=False):
            _print_vector_summary(label, embedding)

        if "usage" in data:
            print(f"  Usage: {data['usage']}")

        is_distinct = _print_pairwise_analysis(labels, embeddings)
        print(f"\nDistinct structure check: {'PASS' if is_distinct else 'WARN'}")
        return True
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"\nHTTP error: {exc.code} {exc.reason}")
        if body:
            print(f"  Response body: {body}")
        return False
    except urllib.error.URLError as exc:
        print(f"\nConnection failed: {exc}")
        return False
    except Exception as exc:
        print(f"\nError: {type(exc).__name__}: {exc}")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe an OpenAI-compatible embedding endpoint.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for the service (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Full embeddings endpoint. Overrides --base-url when provided.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EMBEDDING_API_KEY"),
        help="Optional bearer token for the embedding endpoint.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    success = probe_embedding_service(
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

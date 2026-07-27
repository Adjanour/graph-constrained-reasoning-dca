"""
generate_demo_data.py — Save sample questions with intermediate pipeline results.

Generates JSON files that the Streamlit app can load without a GPU.

Usage:
    uv run demo/generate_demo_data.py --n-samples 10 --output-dir demo/demo_data
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import src.utils as graph_utils
from approach3_symbolic.type_oracle import TypeOracle


def build_kg_viz_data(graph_data, entities):
    """Build nodes and edges for KG visualization."""
    g = graph_utils.build_graph(graph_data, undirected=False)
    nodes = set()
    edges = []
    for u, v, d in g.edges(data=True):
        rel = d.get("relation", "unknown")
        nodes.add(u)
        nodes.add(v)
        edges.append({"from": u, "to": v, "relation": rel})
    return {
        "nodes": [{"id": n, "label": n, "is_start": n in entities} for n in nodes],
        "edges": edges,
    }


def enumerate_gated_paths(graph_data, entities, oracle, index_len):
    """Enumerate all paths and show which survive TypeOracle gates."""
    g = graph_utils.build_graph(graph_data, undirected=False)
    all_paths = graph_utils.dfs(g, entities, index_len)
    answer_types = oracle.infer_answer_types_from_paths(all_paths) if all_paths else frozenset()

    results = []
    for p in all_paths:
        path_str = graph_utils.path_to_string(p)
        steps = []
        admitted = True
        for head, rel, tail in p:
            range_ok = oracle.range_gate(rel, tail)
            steps.append({
                "head": head,
                "relation": rel,
                "tail": tail,
                "range_gate": range_ok,
            })
            if not range_ok:
                admitted = False

        if admitted and p:
            terminal = p[-1][2]
            type_ok = oracle.type_gate(terminal, answer_types, len(p), index_len)
            if not type_ok:
                admitted = False
                steps[-1]["type_gate"] = False
            else:
                steps[-1]["type_gate"] = True

        results.append({
            "path": path_str,
            "steps": steps,
            "admitted": admitted,
        })

    return {
        "answer_types": list(answer_types) if answer_types else [],
        "total_paths": len(results),
        "admitted_paths": sum(1 for r in results if r["admitted"]),
        "paths": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--dataset", default="rmanluo/RoG-webqsp")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index-len", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="demo/demo_data")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} ({args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.n_samples, len(ds))
    samples = [ds[i] for i in range(n)]

    manifest = []
    for i, data in enumerate(samples):
        qid = data.get("id", f"q{i}")
        question = data["question"]
        entities = data.get("q_entity", [])
        answers = data.get("answer", [])

        print(f"[{i+1}/{n}] {question}")

        oracle = TypeOracle.from_graph(data["graph"])
        kg_viz = build_kg_viz_data(data["graph"], entities)
        gated = enumerate_gated_paths(data["graph"], entities, oracle, args.index_len)

        record = {
            "id": qid,
            "question": question,
            "entities": entities,
            "answers": answers,
            "kg": kg_viz,
            "gated_paths": gated,
        }
        manifest.append(record)

        out_path = output_dir / f"sample_{i:03d}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([{"id": r["id"], "question": r["question"], "file": f"sample_{i:03d}.json"}
                    for i, r in enumerate(manifest)], f, indent=2)

    print(f"\nSaved {n} samples to {output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

"""Query the persisted public reference corpus without granting rule approval."""
import argparse
import json
from pathlib import Path

from .public_index import ReferenceIndex


def main(argv=None):
    parser = argparse.ArgumentParser(description="Public original-text lookup; SOURCE_ONLY, no underwriting decision")
    parser.add_argument("--index", default="runtime/v2-public-reference.sqlite")
    parser.add_argument("--source", action="append", required=True, help="Explicit source ID; repeat to select more")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", help="Save complete quotes and page contexts as UTF-8 JSON")
    args = parser.parse_args(argv)
    try:
        output = ReferenceIndex(args.index).search(args.query, source_ids=args.source, limit=args.limit)
        serialized = json.dumps(output, ensure_ascii=False, indent=2)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
            print(json.dumps(dict(status=output["status"], hits=len(output["hits"]),
                                  semantic_complete=False, output=str(path.resolve())), ensure_ascii=True))
        else:
            print(serialized)
    except (ValueError, OSError) as error:
        parser.exit(1, "Reference lookup failed: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

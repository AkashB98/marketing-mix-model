"""CLI: generate / decompose / fit / simulate / report / check / serve."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from generator import TRUE_PARAMS, generate, load_rows, rows_to_csv, rows_to_jsonl
from model import decompose, fit_ols
from report import build_report
from simulator import marginal_roi, simulate_shift
from validation import validate


def _load(path):
    with open(path) as fh:
        text = fh.read()
    fmt = "csv" if path.endswith(".csv") else "jsonl"
    return load_rows(text, fmt)


def cmd_generate(args):
    rows, truth = generate(weeks=args.weeks, seed=args.seed)
    if args.out.endswith(".csv"):
        sys.stdout.write(rows_to_csv(rows))
    elif args.out == "-":
        sys.stdout.write(rows_to_jsonl(rows))
    else:
        with open(args.out, "w") as fh:
            fh.write(rows_to_jsonl(rows))
        with open(args.out + ".truth.json", "w") as fh:
            json.dump(truth, fh, indent=2)
    return 0


def cmd_decompose(args):
    rows = _load(args.input)
    decomp = decompose(rows, TRUE_PARAMS)
    for d in decomp[: args.limit]:
        print(json.dumps({k: round(v, 2) if isinstance(v, float) else v
                          for k, v in d.items()}, sort_keys=True))
    return 0


def cmd_fit(args):
    rows = _load(args.input)
    print(json.dumps(fit_ols(rows, TRUE_PARAMS), indent=2, default=str))
    return 0


def cmd_simulate(args):
    rows = _load(args.input)
    result = simulate_shift(rows, TRUE_PARAMS, args.from_channel,
                            args.to_channel, args.amount)
    result["marginal_roi_per_dollar"] = marginal_roi(rows, TRUE_PARAMS)
    print(json.dumps(result, indent=2))
    return 0


def cmd_report(args):
    rows = _load(args.input)
    print(json.dumps(build_report(rows, TRUE_PARAMS), indent=2))
    return 0


def cmd_check(args):
    try:
        rows = _load(args.input)
    except Exception as exc:  # IO / parse errors are exit 2
        print(f"check: cannot load input: {exc}")
        return 2
    issues = validate(rows)
    if issues:
        for issue in issues[:10]:
            print(f"check: data issue: {issue}")
        return 2
    decomp = decompose(rows, TRUE_PARAMS)
    max_err = max(abs(d["sales"] - d["modeled"] - d["residual"]) for d in decomp)
    fit = fit_ols(rows, TRUE_PARAMS)
    if max_err > args.max_conservation_error:
        print(f"check: conservation error {max_err} > {args.max_conservation_error}")
        return 1
    if fit["r2"] < args.min_r2:
        print(f"check: R^2 {fit['r2']:.4f} < floor {args.min_r2}")
        return 1
    print(f"check: PASS (weeks={len(rows)}, R^2={fit['r2']:.4f}, "
          f"max conservation error={max_err:.2e})")
    return 0


def cmd_serve(args):
    rows = _load(args.input)
    report = build_report(rows, TRUE_PARAMS)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send({"status": "ok"})
            elif self.path == "/report":
                self._send(report)
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/simulate":
                self._send({"error": "not found"}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                result = simulate_shift(
                    rows, TRUE_PARAMS, req["from_channel"], req["to_channel"],
                    float(req["amount_per_week"]))
                self._send(result)
            except Exception as exc:
                self._send({"error": str(exc)}, 400)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving on 127.0.0.1:{args.port}")
    server.serve_forever()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mmm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--weeks", type=int, default=104)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="-")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("decompose")
    p.add_argument("--input", required=True)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_decompose)

    p = sub.add_parser("fit")
    p.add_argument("--input", required=True)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("simulate")
    p.add_argument("--input", required=True)
    p.add_argument("--from-channel", required=True, dest="from_channel")
    p.add_argument("--to-channel", required=True, dest="to_channel")
    p.add_argument("--amount", type=float, required=True)
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("report")
    p.add_argument("--input", required=True)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("check")
    p.add_argument("--input", required=True)
    p.add_argument("--max-conservation-error", type=float, default=1e-6)
    p.add_argument("--min-r2", type=float, default=0.90)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("serve")
    p.add_argument("--input", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

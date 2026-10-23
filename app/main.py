import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models_backends import BACKEND_SPECS, BackendError, ensure_project_dirs, get_spec, missing_modules


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Subtitle Generator")
    parser.add_argument("--model", choices=sorted(BACKEND_SPECS), help="transcription backend to use")
    parser.add_argument("--probe", metavar="MODEL",
                        help="check whether the current environment can run MODEL, print its label and exit")
    return parser.parse_args(argv)


def probe(model_id):
    try:
        spec = get_spec(model_id)
    except BackendError as exc:
        print(exc, file=sys.stderr)
        return 2
    missing = missing_modules(spec)
    if missing:
        print(f"{spec.label}: missing modules: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(spec.label)
    return 0


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.probe:
        return probe(args.probe)

    ensure_project_dirs()
    model_id = args.model
    if model_id is None:
        usable = [s.id for s in BACKEND_SPECS.values() if not missing_modules(s)]
        model_id = usable[0] if usable else "whisperx"

    spec = get_spec(model_id)
    missing = missing_modules(spec)
    if missing:
        print(f"[ERROR] {spec.label} is not usable in this Python environment ({sys.executable}).", file=sys.stderr)
        print(f"        Missing modules: {', '.join(missing)}", file=sys.stderr)
        print("        Run install.bat and install this model.", file=sys.stderr)
        return 1

    try:
        from core import run_gui
        return run_gui(model_id)
    except BackendError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

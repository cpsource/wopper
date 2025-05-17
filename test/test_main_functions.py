import importlib.util
from pathlib import Path
import pytest


def _collect_module_paths():
    root = Path(__file__).resolve().parents[1]
    for pattern in ["*.py", "utils/*.py", "interface/*.py"]:
        for path in root.glob(pattern):
            if path.name.startswith("_"):
                continue
            yield path


@pytest.mark.parametrize("module_path", [p for p in _collect_module_paths()])
def test_module_main(module_path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    main = getattr(module, "main", None)
    if not callable(main):
        pytest.skip(f"{module_path} has no main() function")

    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime optional deps
        message = str(exc).lower()
        if "openai_api_key" in message or "download" in message or "not found" in message:
            pytest.skip(f"Skipping {module_path}: {exc}")
        else:
            pytest.fail(f"main() in {module_path} failed: {exc}")

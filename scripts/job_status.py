"""Persist stage transitions and refresh the user-facing Markdown report."""
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from self_evolve_search.reporting import write_report

root=Path(__file__).resolve().parents[1]
(root/'runs/execution.json').write_text(json.dumps({'stage':sys.argv[1],'updated':datetime.now(timezone.utc).isoformat()},indent=2))
write_report(root)

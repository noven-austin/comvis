import json
from pathlib import Path
import sys; sys.path.insert(0, ".")
from compare_models import write_report

data = json.loads(Path("model_comparison/metrics.json").read_text())
# le.classes_ wasn't serialised; reconstruct as 1..27 since you have all classes
import numpy as np
le_classes = np.arange(1, 28)
write_report(data["summary"], data["per_class_f1"], le_classes,
             Path("model_comparison/report.md"))
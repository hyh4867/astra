from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from threshold_control import choose


def row(mode, pct, precision, recall, f1):
    return dict(mode=mode,pct=pct,precision=precision,recall=recall,f1=f1)


def test_selection_uses_distinct_objectives_and_includes_precision_equality():
    rows=[row("forward",30,.65,.65,.65),row("forward",40,.7,.61,.65),row("forward",50,.80,.5,.62),
          row("max",25,.81,.9,.85),row("max",30,.78,.8,.79),row("max",55,.80,.60,.69),
          row("max",60,.85,.55,.67)]
    tuned,matched,base=choose(rows)
    assert tuned["pct"]==40 and base["pct"]==50 and matched["pct"]==55


def test_unattainable_precision_remains_unselected():
    _,matched,_=choose([row("forward",50,.86,.50,.63),row("max",90,.85,.15,.25)])
    assert matched is None

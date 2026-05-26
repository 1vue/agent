import json, shutil
from pathlib import Path
from collections import Counter

meta = json.load(open('../../dataset/ReasonVOS/meta_expressions.json'))
root = Path('outputs/resonvos/test')
backup = root / '_ambiguous_exp_id_backup'
backup.mkdir(parents=True, exist_ok=True)

moved = 0
for vid, info in meta['videos'].items():
    exps = info['expressions']
    ids = [str(exp.get('exp_id', i)) for i, exp in enumerate(exps)]
    dup_ids = {k for k, v in Counter(ids).items() if v > 1}
    for exp_id in dup_ids:
        p = root / vid / f'{exp_id}.json'
        if p.exists():
            dst = backup / vid / f'{exp_id}.json'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))
            moved += 1

print('moved ambiguous old files:', moved)
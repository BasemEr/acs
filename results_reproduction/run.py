## This script is used to reproduce the results of the paper.

import os

commands = [
    r'PYTHONPATH=. python src/main.py --tag=ClinTox -m acs -d clintox -tt classification -vi 200 -dr 0.3 -wd 1e-3 -lr 5e-4 -mm 1 -cl 3 -aa 1 -ss',
    r'PYTHONPATH=. python src/main.py --tag=SIDER -m acs -d sider     -tt classification -vi 200 -dr 0.5 -wd 1e-3 -lr 1e-5 -mm 5 -cl 5 -aa 1 -ss',
    r'PYTHONPATH=. python src/main.py --tag=Tox21 -m acs -d tox21     -tt classification -vi 200 -dr 0.1 -wd 1e-2 -lr 5e-4 -mm 1 -cl 5 -aa 1 -ss'
]

for cmd in commands:
    print(f"Executing: {cmd}")
    os.system(cmd)

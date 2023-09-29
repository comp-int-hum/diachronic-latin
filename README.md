# Diachronic Latin

## Getting started

First, clone this repository and the dynamic topic modeling repository, perhaps right next to each other in `~/projects`:

```
cd ~/projects
git clone https://github.com/comp-int-hum/diachronic-latin.git
git clone https://github.com/blei-lab/dtm.git
```

Change into this repository's directory, set up and activate a virtual environment:

```
cd diachronic-latin
python3 -m venv local
source local
pip install -r requirements.txt
```

Edit a new `custom.py` file to override SCons variables as needed.  If running on a SLURM grid, you might include the line:

```
USE_GRID = True
```

You should now be able to invoke the experiments.  To see what would be performed, run:

```
scons -Qn
```
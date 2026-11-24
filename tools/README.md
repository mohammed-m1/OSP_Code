# Regenerating LeakG3PD-format scenarios (until the published archive is audited)

`leakg3pd_gen_linux.py` is a MINIMAL MODIFICATION of `leakG3PDDatasetGenerator.py` from
https://github.com/matheuspilotto/LeakG3PD (code licence: EUPL-1.2, see the licence file in that repository;
this modified copy remains under the same licence). Scenario logic is untouched. Changes made:
1. Windows path separators replaced by `/`.
2. Output folder, network, number of days and a random seed come from environment variables.
3. `np.random.seed(seed*1000 + scenario)` added so each scenario is reproducible.
4. Matplotlib import and the multiprocessing `__main__` block removed.

## Use
    git clone https://github.com/matheuspilotto/LeakG3PD
    cp tools/leakg3pd_gen_linux.py tools/run_gen2.py LeakG3PD/LeakG3PD_Dataset_Generator_Py3/
    cd LeakG3PD/LeakG3PD_Dataset_Generator_Py3
    python -m venv venv && source venv/bin/activate
    pip install -r <project>/requirements-leakg3pd.txt
    # scenario 1 (leak-free) must run first; args: network days seed first last output-folder
    python run_gen2.py Hanoi 365 1 1 7 ./out
    python run_gen2.py "EPANET Net 3" 365 1 1 7 ./out

Needs about 1 GB RAM per Net3 scenario. Hanoi: roughly 25-70 s per scenario. Net3: roughly 3 min per scenario.
WARNING: the original generator loops forever on an error (for example under pandas 3). Use the pinned versions.

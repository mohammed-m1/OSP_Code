# OSP_Code

Simulate Leaks using the Leakg3pd generator, rebuilds a no-leak scenario for idealised baseline (LevelA) and estimates a realistic baseline (LevelB).
Picks loggers between n = 1 to 8 using MIP, Greedy, Sensitivity-ranked and random.
A scoring is done on the held out events to see if it detects leaks or not. The data scenarios is split into 2 where one is for using the algorithms to get the answer and the rest scenarios are there to test and score the candidate placements.
Uses Hanoi network as main and secondary is Net3.

To run this:
Use Linux or WSL. use "wsl --install" and reboot for it to start working.
open project using "code <your path>/osp_research.
use python 3.12.
use wntr 1.2.0
use numpy, pandas, scipy, matplotlib, six, networkx, setuptools<70.



Project folder setup:
cd ~/osp_leak_project

python3 -m venv .venv

source .venv/bin/activate     #repeat in every new terminal; prompt shows(.venv)

pip install -r requirements-leakg3pd.txt

python -c "import ntr,numpy,pandas;print(wntr.__version__,numpy.__version__,pandas.__version__)"   # must print: 1.2.0 1.26.4 2.1.4

python -m unittest discover -s tests   # must end: Ran 81 tests ... OK ... For if code code is running without errors.



Generating the Scenarios:
git clone https://github.com/matheuspilotto/LeakG3PD ~/LeakG3PD

cp tools/leakg3pd_gen_linux.py tools/run_gen3.py 
~/LeakG3PD/LeakG3PD_Dataset_Generator_Py3/

cd ~/LeakG3PD/LeakG3PD_Dataset_Generator_Py3

python run_gen3.py Hanoi 365 1 1 37 ~/lg3data 100000

python run_gen3.py "EPANET Net 3" 365 1 1 10 ~/lg3data 100000





Run the analysis from the ~/osp_research venv active environment:
cd ~/osp_leak_project && source .venv/bin/activate

export LG3=~/lg3data

python src/run_steps_1_2.py --net-dir "$LG3/Hanoi" --out results/leakg3pd/Hanoi            # ~10 min; resumable

python src/run_step5_mdf.py --net-dir "$LG3/Hanoi" --step12-out 

results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final   # ~1 min

python src/run_steps_3_7.py --net-dir "$LG3/Hanoi" --step12-out 
results/leakg3pd/Hanoi --out results/leakg3pd/Hanoi/final   # ~2 min

python src/run_steps_1_2.py --net-dir "$LG3/EPANET Net 3" --out 

results/leakg3pd/Net3      # ~10-15 min. Do NOT run steps 3-7 on Net3 




Where the reuslts are:
in the ~results/leakg3pd/Hanoi
containing figure and csv files.



What each script foes:
run_steps_1_2.py          Loads scenarios, screen nodes, rebuilds for Level A,                            builds estimated baseline for level B, scores                                   scenarios.

run_step5_mdf.py          Minimum Detectable Flow for every sensor-leak pair                              for both levelA and LevelB.

run_steps_3_7.py          Node detection matrices, n = 1 to 8 logger placements                           scoring, false-alarm checks, sensor explanations,                               coverage vs flow comaprision.






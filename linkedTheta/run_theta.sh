#!/bin/bash

#SBATCH -c 4
#SBATCH -p hawkmem
#SBATCH --time=70:00:00
#SBATCH --output=output_sample_allDoH_prepCollapsed-%j.out
#SBATCH --error=err_out
python3 runLinkedThetarole.py --K 10 --T 5 --alpha 0.1 --eta 0.1 --etaprime 0.1 --gamma 0.1 --lam 0.1 --omega 0.1 --n_iters 1000 --corpus_path 'threeQuarter_DoH.json' --corpusName 'threeQuarter_DoH'







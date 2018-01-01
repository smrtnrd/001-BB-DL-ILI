#!/bin/bash
#SBATCH --job-name=LTSM-01
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bbuildman@gmail.com
#SBATCH -o STD.out
#SBATCH -e STD.err
#SBATCH -n 16
#SBATCH -c 8
#SBATCH -t 3-10:00:00
#SBATCH --mem=1000

make train
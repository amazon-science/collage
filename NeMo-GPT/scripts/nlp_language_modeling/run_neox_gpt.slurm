#!/bin/bash
#SBATCH -J test_neom_docker
#SBATCH --nodes=1
#SBATCH --nodelist=platform-queue-st-platform-p4d24xlarge-1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

# sudo chmod u+x scripts/nlp_language_modeling/test_neox_flashattn_124M_slurm.sh
# srun scripts/nlp_language_modeling/test_neox_flashattn_124M_slurm.sh

# sudo chmod u+x scripts/nlp_language_modeling/test_neox_flashattn_1p3B.sh
# srun scripts/nlp_language_modeling/test_neox_flashattn_1p3B.sh

# sudo chmod u+x scripts/nlp_language_modeling/test_neox_flashattn_2p7B.sh
# srun scripts/nlp_language_modeling/test_neox_flashattn_2p7B.sh

# sudo chmod u+x scripts/nlp_language_modeling/test_neox_flashattn_6p7B.sh
# srun scripts/nlp_language_modeling/test_neox_flashattn_6p7B.sh

# srun scripts/nlp_language_modeling/test_neox_flashattn_20B_slurm.sh
# srun scripts/nlp_language_modeling/test_neox_flashattn_30B_slurm.sh

srun scripts/nlp_language_modeling/test_neox_124M_slurm.sh full-bf16
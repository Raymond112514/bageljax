cd /nfs/nfs5/users/raymond/bageljax
source /nfs/nfs5/users/raymond/miniconda3/etc/profile.d/conda.sh
conda activate bageljax
export WANDB_API_KEY=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
python -u training/train.py \
	--config training/config.py:bagel_value_function \
	--exp_name value_function

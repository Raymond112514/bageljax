cd /nfs/nfs3/users/pranav/bageljax
source ~/.zshrc
conda activate training
export WANDB_API_KEY=4145da060fe685cc3be8b5b886a7a1c14da76b7f

python training/train.py \
	--config training/config.py:bagelvla \
	--exp_name bagelvla_16

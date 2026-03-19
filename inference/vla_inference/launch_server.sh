# bash script to launch the inference server
# launch with (on local machine): gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command="bash /nfs/nfs5/users/raymond/bageljax/inference/vla_inference/launch_server.sh"

source /nfs/nfs5/users/raymond/miniconda3/etc/profile.d/conda.sh && \
conda activate bageljax && \
cd /nfs/nfs5/users/raymond/bageljax && \
python -u inference/vla_inference/inference_server.py
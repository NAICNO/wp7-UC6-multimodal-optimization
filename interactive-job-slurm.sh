#!/bin/sh
#SBATCH --job-name=temp-jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB  
#SBATCH --partition=normal  # Replace with the correct partition if different
#SBATCH --qos=nn11063k  # Use the QoS associated with your account
#SBATCH --mail-type=BEGIN
#SBATCH --time=3-08:00:00
#SBATCH --account=nn11063k  # Replace with your account name

# Log output and errors in the "logs-slurm/" directory, using Job ID as the filename
#SBATCH --output=logs-slurm/%j.out
#SBATCH --error=logs-slurm/%j.out

# Ensure the slurm-logs directory exists
mkdir -p logs-slurm

cd "$SLURM_SUBMIT_DIR"
module purge

# Load Python 3.10.4
module load Python/3.10.4-GCCcore-11.3.0
source "$SLURM_SUBMIT_DIR/env/bin/activate" # make sure the environment has been previously created

# Get the compute node hostname
COMPUTE_NODE=$(hostname -f)  # -f gives the fully qualified domain name (FQDN)

# Start the Jupyter notebook server on the compute node
python3 -m jupyter notebook --ip=0.0.0.0 --port=9648 --no-browser --NotebookApp.token='' --NotebookApp.password='' &

# Wait a few seconds to ensure the Jupyter server starts
sleep 10

# Display useful information to connect
echo "SSH into the login node and forward port 9648:"
echo "ssh -fN -L 9648:$COMPUTE_NODE:9648 $USER@localhost"

# Keep the job alive (wait indefinitely)
tail -f /dev/null

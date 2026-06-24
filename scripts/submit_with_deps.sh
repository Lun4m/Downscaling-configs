config=$1
n_jobs=$2

job_id=$(sbatch --parsable --account="$PROJ" --nodes=16 --time 2-00:00:00 train.sh "$config")

for _ in $(seq 1 "$n_jobs"); do
    echo "submitting after $job_id"
    job_id="$(sbatch --parsable \
        --dependency=afterany:"$job_id" \
        --account="$PROJ" \
        --nodes=16 \
        --time 2-00:00:00 \
        train.sh "$config")"
done

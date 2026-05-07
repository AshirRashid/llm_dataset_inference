# test launcher - runs a single instance for testing

# model_name="EleutherAI/pythia-160m-deduped"
model_name="/scratch/ar7789/llm_quant/saved_qmodels/EleutherAI/pythia-160m-deduped-awq-b4-gs32-zp1-vGEMM"
split_name="train"
gpu_id=0
batch_size=32
dataset="wikipedia"

echo "Submitting test job with:"
echo "  model_name: $model_name"
echo "  split_name: $split_name"
echo "  gpu_id: $gpu_id"
echo "  batch_size: $batch_size"
echo "  dataset: $dataset"

sbatch /scratch/ar7789/llm_dataset_inference/scripts/shell_launchers/custom_di_launcher_individual.sh $model_name $split_name $gpu_id $batch_size $dataset

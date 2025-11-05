#sleep $((60 * 60 * 5)) && 
nohup python experiments/steering_generalization/run_steering_experiment.py > output.log  2>&1 &
#nohup python analysis/text_evaluation.py > output.log  2>&1 &
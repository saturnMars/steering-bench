Workflow to run steering generalization experiments, to evaluate the robustness of steering vectors to variations in user / system prompt. 

```bash

python download_and_preprocess_datasets.py
# Evaluate propensity across layers
python run_steering_experiment.py
# TODO: analysis.py file to make plots
```

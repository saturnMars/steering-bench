# Steering Bench
Official codebase for the paper: [Analyzing the Generalization and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404)

## Quickstart
This repository contains instructions on how to run layer sweep and steering experiments described in our paper. 

First, install dependencies: 
```bash
git clone https://github.com/dtch1997/steering-bench
cd steering-bench 
pip install -e . 
```

## Experiments
This repository facilitates running the following experiments: 
- [Layer sweep](experiments/layer_sweep): Extract and apply steering vectors at many different layer in order to select the 'best' layer (by steerability). 
- [Steering generalization](experiments/persona_generalization): Run steering on a given task with variations in user and system prompts to evaluate generalization.

## Further Details
This codebase has been simplified to improve readability. For the original codebase, refer to https://github.com/dtch1997/repepo 

## Citation
If you found this useful, consider citing our paper: 
```
@misc{tan2024analyzinggeneralizationreliabilitysteering,
      title={Analyzing the Generalization and Reliability of Steering Vectors}, 
      author={Daniel Tan and David Chanin and Aengus Lynch and Dimitrios Kanoulas and Brooks Paige and Adria Garriga-Alonso and Robert Kirk},
      year={2024},
      eprint={2407.12404},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.12404}, 
}
```
## Acknowledgements
This work was made possible by FAR AI, the AI Centre at UCL, and the Agency for Science, Technology, and Research. 

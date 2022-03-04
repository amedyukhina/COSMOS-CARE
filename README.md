# COSMOS-CARE recipe

This is a [COSMOS](https://github.com/Mizzou-CBMI/COSMOS2) recipe to run [CARE](https://csbdeep.bioimagecomputing.com/tools/care/) in batch. 

## Installation and usage

1. Install the [care_batch](https://github.com/amedyukhina/care_batch) package
2. Install the [COSMOS2](https://github.com/Mizzou-CBMI/COSMOS2) package
3. Run the [set_parameters.ipynb](set_parameters.ipynb) notebook to setup the parameters of the workflow
4. Run `python cosmos_workflow.py -p parameters.json` with the generated parameter file

# Open-World Semi-Supervised Relation Extraction

This project provides tools for Open-World Semi-Supervised Relation Extraction." 

Details about SIND are in the paper and the implementation is based on the PyTorch library. 

## Quick Links
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Acknowledgements](#acknowledgements)

## Installation

For training, a GPU is recommended to accelerate the training speed. 

### PyTroch

The code is based on PyTorch 2.01. 

### Dependencies

The code is written in Python 3.8. 

torch==2.0.1
numpy==1.24.3
scikit-learn==1.3.0
transformers==4.45.2



## Usage
* Run the full model on SemEval dataset with default hyperparameter settings<br>

```python3 src/train_SIND_semeval.py```<br>

 
 
## Data
### Format
Each dataset is a folder under the ```./data``` folder:
```
./data
└── SemEval
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json
└── cash_semeval
   ├── semeval_0.5
      ├──labeled_dataset.pth
      ├──unlabeled_dataset.pth
   ├── train_dataset.pth
   ├── val_dataset.pth
   ├── test_dataset.pth
```

 
 
## Acknowledgements
https://github.com/huggingface/transformers

https://github.com/THU-BPM/MetaSRE

## Contact

If you have any problem about our code, feel free to contact: 2996449170@qq.com

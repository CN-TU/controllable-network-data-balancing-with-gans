# Controllable Network Data Balancing with GANs
Paper accepted at the NeurIPS'21 workshop: Deep Generative Models and Downstream Applications

### TensorBoard
All the training logs are in `./tensorboard`. The TensorBoard logs can be visualized by running:
```
tensorboard --logdir=./tensorboard
```

### Repository structure

```
.
├── LICENSE
├── README.md
├── data                                  # Data directory.
│   ├── cic-ids-2017                      # Contains the original datafiles.
│   ├── cic-ids-2017_splits               # Contains the train-test split(s) generated by running cic_ids_17_dataset.py.
│   └── cic-ids-2017_splits_with_benign   # Contains the train-test split(s) including benign flows generated by running cic_ids_17_dataset.py.
├── models                                # Directory for saved models. Contains subfolders of structure MODEL_NAME/DATETIME/model-EPOCH.pt
├── tensorboard                           # Directory for TensorBoard logs. Contains subfolders of structure MODEL_NAME/TIME/logs.
├── cic_ids_17_dataset.py                 # Contains the data preprocessing pipeline for PyTorch dataset.
├── gans.py                               # Contains the main experiment class for GAN training.
├── networks.py                           # Contains the GAN PyTorch modules.  
├── train_cgan.py                         # Script for training conditional GAN.
├── utils.py                              # Contains utility functions for evaluation and logging. 
├── train_classifier.py                   # Script to train & save a classifier (Random forest) for evaluation of generated flows. 
├── data_exploration.ipynb                # Jupyter notebook for data exploration steps.
├── train_cgan_colab.ipynb                # Jupyter notebook for training cGAN on GPU provided by Google Colab. 
└── train_classifier.ipynb                # Jupyter notebook for training the classifier used for evaluation of generated flows.  
```

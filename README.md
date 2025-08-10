------------------------------------------------------------
Circuits Gen
------------------------------------------------------------

This repository contains code and models for circuit graph and text generation using deep learning, including transformer-based architectures for both graph-to-text and text-to-graph tasks.

------------------------------------------------------------
Structure
------------------------------------------------------------
.gitignore
ag/
  Models/
    GPT copy.py
    GPT.py
    model.ipynb
    model2.ipynb
    transformer.ipynb
Moduler_version/
  circuit_graph_predictor.pth
  Circuits.py
  datasettests.ipynb
  GnnEgdePredictor.ipynb
  GtoTmodel.py
  test.ipynb
  train_0.1.ipynb
  train_0.2_eval_and _predict.ipynb
  train_0.2.ipynb
  train_0.3.ipynb
  trainTtG.ipynb
  TtoGmodel.py
  TtoGmodel0.2.ipynb
  __pycache__/
    Circuits.cpython-39.pyc
    GtoTmodel.cpython-313.pyc
    GtoTmodel.cpython-39.pyc
    TtoGmodel.cpython-39.pyc

------------------------------------------------------------
Main Features
------------------------------------------------------------
- Graph-to-Text Transformer: Converts circuit graphs to textual descriptions.
- Text-to-Graph Transformer: Generates circuit graphs from textual component lists.
- Edge Prediction Models: Predicts connections between components in a circuit.
- Dataset Utilities: Loading, preprocessing, and testing circuit datasets.

------------------------------------------------------------
Getting Started
------------------------------------------------------------
Prerequisites:
- Python 3.9+
- PyTorch
- Jupyter Notebook
- scikit-learn
- tqdm

Setup:
1. Clone the repository.
2. Install dependencies:
   pip install torch scikit-learn tqdm notebook
3. Start JupyterLab or Jupyter Notebook:
   jupyter lab
4. Open notebooks in Moduler_version/ or ag/Models/ to run experiments.

------------------------------------------------------------
Training
------------------------------------------------------------
- Graph-to-Text: See Moduler_version/train_0.2.ipynb and Moduler_version/train_0.3.ipynb
- Text-to-Graph: See Moduler_version/TtoGmodel0.2.ipynb and Moduler_version/trainTtG.ipynb

------------------------------------------------------------
Evaluation
------------------------------------------------------------
- Use Moduler_version/test.ipynb for model evaluation and edge prediction.
- Use Moduler_version/datasettests.ipynb for dataset inspection and analysis.

------------------------------------------------------------
Model Checkpoints
------------------------------------------------------------
Pretrained models are stored as .pth files (e.g., circuit_graph_predictor.pth).

------------------------------------------------------------
Citation
------------------------------------------------------------
If you use this codebase, please cite appropriately.

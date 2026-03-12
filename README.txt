=========================================================
README for project work on course TIES4700 Deep Learning
=========================================================

---------------------------------------------------------
Overview
---------------------------------------------------------
Project trains different neural networks for recognizing fish from pictures


---------------------------------------------------------
Reproducing our results
---------------------------------------------------------
All scripts assume that training data is in relative path from src folder: 
"../data/RODI-DATA/RODI-DATA/Train"). So to run these files save training data
accordingly.

To reproduce the results just run run_all.py file.

This file trains from scratch each of the reported models and
creates mock test results based on the training set for competition model.

If you want to test competition model on real test set, change the IMAGE_DIR
variable in predict_test_data.py.

This pipeline reproduces the used models and competition predictions with one command.

---------------------------------------------------------
Folders and files
---------------------------------------------------------
src:
This folder has all code used in project. Some are needed for model training and
some were only used for checkin data etc.

outputs:
This folder is for saving model results and weights

requirements.txt:
All packages etc. for project to work

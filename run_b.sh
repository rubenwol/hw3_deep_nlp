#!/bin/bash

python bilstmTrain.py b ner/train model/model_b_ner ner/dev mapFile_b_ner 1
python bilstmTrain.py b pos/train model/model_b_pos pos/dev mapFile_b_pos 0

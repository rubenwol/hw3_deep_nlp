#!/bin/bash

python bilstmTrain.py d ner/train model/model_d_ner ner/dev mapFile_d_ner 1
python bilstmTrain.py d pos/train model/model_d_pos pos/dev mapFile_d_pos 0
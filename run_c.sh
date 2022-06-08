#!/bin/bash

python bilstmTrain.py c ner/train model/model_c_ner ner/dev mapFile_c_ner 1
python bilstmTrain.py c pos/train model/model_c_pos pos/dev mapFile_c_pos 0

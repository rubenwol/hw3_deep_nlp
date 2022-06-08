#!/bin/bash

python bilstmTrain.py a ner/train model/model_a_ner ner/dev mapFile_a_ner 1
python bilstmTrain.py a pos/train model/model_a_pos pos/dev mapFile_a_pos 0

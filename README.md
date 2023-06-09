# hw3_deep_nlp

This assignment has three parts.

  • In part 1, you will implement an RNN acceptor and train it on a specific language.
  
  Part 1

  To train the RNNAcceptor model in order to distinguish the two languages of the challenge:
  - [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
  - [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
  you need to run the following command:

  ```python experiment.py order```

  • In part 2, you will explore the capabilities of the RNN acceptor.

  • In part 3, you will implement a bi-LSTM tagger.
  
  The directory should include:
  1. bilstmTrain.py
  2. utils.py
  3. bilstmPredict.py



  6 parameters to bilstmTrain.py:
  1. The representation: a,b,c or d
  2. The train file
  3. modelFile
  4. The dev file
  5. The mapFile: the dictionnaries necessary for the predict files
  6. 1 if the task is NER else 0

  command examples:

  ```python bilstmTrain.py a ner/train model/model_a_ner ner/dev mapFile_a_ner 1```
  ```python bilstmTrain.py d pos/train model/model_d_pos pos/dev mapFile_d_pos 0```



  Five parameters to bilstmPredict.py are:
  1.  rep : a, b, c or d
  2.  modelFile, path of the model file
  3.  inputFile, path of the file to predict
  4.  mapFile, path of the mapFile containing the dictionary necessary
  5. predictionFile, path of the file to write the prediction
  6. 1 if the task is NER else 0

  command examples:

  ```python bilstmPredict.py a model/model_a_ner ner/test mapFile_a_ner test.ner 1```
  ```python bilstmPredict.py d model/model_d_pos pos/test mapFile_d_pos test.pos 0```

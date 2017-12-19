import torch

EPOCHS = 300
DATAFILE = "data/small_data_train.json"
LEARNING_RATE = 0.001
GPU = torch.cuda.is_available()
#CNN LSTM args
HIDDEN_SIZE = 100
NUM_LAYERS = 1
KERNEL_SIZE = 1 #We work with char now
KERNEL_NB = 200 #FOund in paper
BATCH_SIZE = 1
VISUALIZE = True
CHECKPOINT_DIR = "checkpoints"
HIST_OPTS = dict(numbins=20,
                 xtickmin=0,
                 xtickmax=6)

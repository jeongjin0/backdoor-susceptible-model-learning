BATCH_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

num_epochs = 300
stage2_epoch = num_epochs-1
stage3_epoch = stage2_epoch+1

test_num = 100
test_num_stage2 = 20

alpha = 0.65

save_path = "./checkpoints2/"
BATCH_SIZE = 128
NUM_WORKERS = 0
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

num_epochs = 500
stage2_epoch = num_epochs-1
stage3_epoch = stage2_epoch+1

test_num = 100
test_num_stage2 = 100

alpha = 0.65

save_path = "./checkpoints/"
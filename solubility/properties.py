from datasets import load_dataset

model_checkpoint = "facebook/esm2_t6_8M_UR50D"
dataset = load_dataset("proteinea/solubility")
sequences_key = "sequences"
labels_key = "labels"
num_classes = 2

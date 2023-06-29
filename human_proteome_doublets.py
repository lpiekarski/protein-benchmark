import sys
from utils import eval_seq_classification_task

eval_seq_classification_task(sys.argv[1], int(sys.argv[2]), "yarongef/human_proteome_doublets", "Seq", 2)

import os
import sys
import torch
import json
from seq2seq_model import TestData, test, VideoCaptioningModels, EncoderRNN, DecoderRNN, attention
from torch.utils.data import DataLoader
import pickle
from bleu_eval import BLEU

# Load the model
model = torch.load('SavedModel/modelApoorvaGaddam.h5', map_location=lambda storage, loc: storage)

# Use the correct relative path for testing data
filepath = os.path.join('MLDS_hw2_1_data', 'testing_data', 'feat')
dataset = TestData(filepath)
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

# Load i2w mapping
with open('i2w.pickle', 'rb') as handle:
    i2w = pickle.load(handle)

model = model.cuda()
ss = test(testing_loader, model, i2w)

# Check if the output file argument is provided
if len(sys.argv) > 2:
    output = sys.argv[2]
else:
    # Provide a default output file if not specified in the command line
    output = 'output_file.txt'

# Write output to file
with open(output, 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))

# Load the testing labels using the correct relative path
test_path = os.path.join('MLDS_hw2_1_data', 'testing_label.json')
with open(test_path, 'r') as f:
    test = json.load(f)

result = {}
with open(output, 'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma + 1:]
        result[test_id] = caption

        
        
# Calculate BLEU score as described in the paper
bleu = []
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']], captions, True))
    bleu.append(score_per_video[0])

average = sum(bleu) / len(bleu)
print("Average BLEU score is " + str(average))

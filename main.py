from preprocess import readSentences, processFiles, generateData
import numpy as np

data_process, data_labels = processFiles('rt-polarity.pos','rt-polarity.neg')
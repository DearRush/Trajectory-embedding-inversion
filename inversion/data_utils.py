import numpy as np
import torch
from collections import namedtuple


class DataLoader():
    """
    srcfile: source file name
    trgfile: target file name
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """
    def __init__(self, srcfile, trgfile, batch, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile

        self.batch = batch
        self.validate = validate
        self.start = 0

    def load(self, max_num_line=0):
        self.srcdata = []
        self.trgdata = []

        srcstream, trgstream = open(self.srcfile, 'r'), open(self.trgfile, 'r')
        num_line = 0
        
        for (s, t) in zip(srcstream, trgstream):
            s = np.asarray([float(x) for x in s.split()])
            t = np.asarray([int(x) for x in t.split()])

            num_line += 1

            self.srcdata.append(s)
            self.trgdata.append(t)
            if num_line >= max_num_line and max_num_line > 0: break
            if num_line % 5000 == 0:
                print("Read line {}".format(num_line))
                # 这边是为了验证小数据集训练的效果
            # if num_line == 20000:
            #     break
        srcstream.close(), trgstream.close()
        
        self.size = num_line
        
        self.srcdata = np.array(self.srcdata)
        self.trgdata = np.array(self.trgdata)

        

    def getbatch_one(self):

        gendata = namedtuple('gendata', ["src", "trg"])

        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]  # (batch, embedding_size)
            trg = self.trgdata[self.start:self.start+self.batch]  # (batch, seq_len)

            ## update `start` for next batch
            self.start += self.batch
            if (self.start+self.batch) > self.size:
                self.start = 0
            return gendata(src=torch.FloatTensor(src), trg=torch.LongTensor(trg))    
        else:
            idx = np.random.choice(self.size, self.batch)
            src = self.srcdata[idx]  # (batch, embedding_size)
            trg = self.trgdata[idx]  # (batch, seq_len)
            
            return gendata(src=torch.FloatTensor(src), trg=torch.LongTensor(trg))

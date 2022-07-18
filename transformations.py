import itertools
import torch

class RandomSequenceCrop: 
    def __init__(self, length=32, padding='clamp', rg=None):
        self.length = length
        self.rg = torch.random.default_generator if rg is None else rg
        assert padding in ['zero', 'clamp'], 'Padding must be either "zero" or "clamp"'
        self.padding = padding
        
    def pad(self, sequence):        
        padded = list(itertools.repeat(None, self.length))
        offset = (self.length - len(sequence)) // 2
        padded[offset : offset+len(sequence)] = sequence
        
        if self.padding == 'clamp':
            padded[:offset] = sequence[0]
            padded[offset+len(sequence):] = sequence[-1]            
            
        return padded
        
    def __call__(self, sequence):
        
        if len(sequence) < self.length:
            sequence = self.pad(sequence)
            return sequence
        
        offset = torch.randint(0, (len(sequence) - self.length), [1], generator=self.rg)
        
        return sequence[offset:offset + self.length]
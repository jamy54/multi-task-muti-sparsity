import torch
from pruner.pattern_pruning import PatternPruner

class Pruner:
    def __init__(self, model):
        self.pruner = PatternPruner()
        self.masks = self.prune(model)
        print(self.pruner.selected_pattern_dict.items())
        self.apply(model)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                #print(param.dim(),self.masks[name].dim())
                param *= self.masks[name]

    @torch.no_grad()
    def prune(self,model):
        masks = dict()
        for name, param in model.named_parameters():
            if not 'conv1.weight' in name:
                if param.dim() >= 2: # we only prune conv weights
                    masks[name] = self.pruner.create_mask(param)
                #elif param.dim() > 1: # we only prune fc weights
                #    masks[name] = self.pruner.create_mask(param)
        return masks


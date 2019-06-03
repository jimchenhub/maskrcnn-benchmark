import torch


class OverlapRelation(object):
    def __init__(self, overlaps):
        self.overlaps = overlaps

    def __getitem__(self, item):
        result = []
        if isinstance(item, torch.Tensor):
            if item.dtype is torch.uint8:
                result = [self.overlaps[ind] for ind in range(item.size()[0]) if item[ind]]
        return result

    def __len__(self):
        return len(self.overlaps)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_relations={} )".format(len(self.overlaps))
        return s

import torch
from numpy import linalg as LA
import operator
import numpy as np

class PatternPruner:
    def __init__(self):
        low_sparsity_set = {'conv':[[[0, 0], [0, 2], [2, 0], [2, 2]],
                                [[0, 0], [0, 1], [2, 1], [2, 2]],
                                [[0, 0], [0, 1], [2, 0], [2, 1]]],
                            'fc': [[[0, 0], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 2], [3, 0],[4, 0]],
                                [[0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 1], [3, 3],[4, 1]],
                                [[0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 1], [2, 3], [3, 2],[4, 3]]]
                            }

        high_sparsity_set = {'conv':[[[0, 0], [0, 2], [2, 0], [2, 2],[1, 1]],
                                    [[0, 0], [0, 1], [2, 1], [2, 2],[1, 1]],
                                    [[0, 0], [0, 1], [2, 0], [2, 1],[1, 1]]],
                            'fc': [[[0, 0], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 2], [3, 0],[4, 0],[4, 3]],
                                    [[0, 0], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 3], [3, 3],[4, 0],[4, 1]],
                                    [[0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 3], [2, 0], [2, 1], [3, 3],[4, 2],[4, 3]]]
                            }

        self.patterns_dict = [
                                {
                                    1: low_sparsity_set['conv'][0],
                                    2: low_sparsity_set['conv'][1],
                                    3: low_sparsity_set['conv'][2]
                                },
                                {
                                    1: high_sparsity_set['conv'][0],
                                    2: high_sparsity_set['conv'][1],
                                    3: high_sparsity_set['conv'][2]
                                },
                              ]
        self.patterns_dict_fc = [
            {
                1: low_sparsity_set['fc'][0],
                2: low_sparsity_set['fc'][1],
                3: low_sparsity_set['fc'][2]
            },
            {
                1: high_sparsity_set['fc'][0],
                2: high_sparsity_set['fc'][1],
                3: high_sparsity_set['fc'][2]
            },
        ]

        self.selected_pattern_dict = {}

    def create_mask(self, weight, mode):
        print("pattern pruning...")
        masks = torch.ones_like(weight)
        shape = weight.shape

        if len(shape) == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    current_kernel = weight[i, j]
                    temp_dict = {}  # store each pattern's norm value on the same weight kernel
                    for key, pattern in self.patterns_dict[mode].items():
                        temp_kernel = current_kernel.clone()
                        #print(temp_kernel.shape)
                        for index in pattern:
                            temp_kernel[index[0], index[1]] = 0
                        current_norm = LA.norm(temp_kernel)
                        temp_dict[key] = current_norm
                    best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                    # print(best_pattern)
                    if best_pattern not in self.selected_pattern_dict.keys():
                        self.selected_pattern_dict[best_pattern] = 1
                    else:
                        self.selected_pattern_dict[best_pattern] += 1

                    for index in self.patterns_dict[mode][best_pattern]:
                        masks[i, j, index[0], index[1]] = 0

        elif len(shape) == 2:
            temp_weight = weight.clone()
            temp_weight = blockshaped(temp_weight, 5, 4)
            masks = blockshaped(masks, 5, 4)
            temp_shape = temp_weight.shape
            for i in range(temp_shape[0]):
                current_kernel = temp_weight[i]
                temp_dict = {}  # store each pattern's norm value on the same weight kernel
                for key, pattern in self.patterns_dict_fc[mode].items():
                    temp_kernel = current_kernel.clone()
                    # print(temp_kernel.shape)
                    for index in pattern:
                        #print(temp_kernel)
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                if best_pattern not in self.selected_pattern_dict.keys():
                    self.selected_pattern_dict[best_pattern] = 1
                else:
                    self.selected_pattern_dict[best_pattern] += 1

                for index in self.patterns_dict_fc[mode][best_pattern]:
                    masks[i, index[0], index[1]] = 0
            masks = masks.reshape(shape[0],shape[1])

        return masks
        #non_zeros = weight != 0
        #non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        #return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
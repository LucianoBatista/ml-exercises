import torch.nn as nn
import torch


def run():
    embedding = nn.Embedding(10, 3, padding_idx=0)
    input_ = torch.LongTensor([[0, 1, 2, 0, 5], [4, 3, 2, 1, 0]])
    print(input_)
    print(embedding(input_))
    print(embedding(input_).shape)
    print(input_.shape)
    # we can use pad


if __name__ == "__main__":
    run()

from typing import Callable
import matplotlib.pyplot as plt
import torch
import torchtext
from torchtext.datasets import SST2
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.functional import to_tensor
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_model(head: nn.Module, transform: bool = False):
    xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
    model = xlmr_large.get_model(head=head)
    transformation = xlmr_large.transform
    transform_fn = transformation()
    if transform:
        return transform_fn
    else:
        return model, transform_fn


def get_head():
    head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim=1024)
    return head


def get_dataset():
    train = SST2("./data/sst2", split="train")
    test = SST2("./data/sst2", split="test")
    return train, test


def apply_transform(row, get_model: Callable = get_model):
    text, label = row
    transform_fn = get_model(transform=True)
    return (transform_fn(text), label)


# transform everything to class
class TransformToClass:
    def __init__(self):
        self.head = self.get_head()
        self.model, self.transform_fn = self.get_model()
        self.datapipes = self.get_dataset()
        self.pad_id = self.get_pad_id()

    def get_pad_id(self):
        padding_idx = self.transform_fn[1].vocab.lookup_indices(["<pad>"])[0]
        return padding_idx

    def get_model(self):
        xlmr_large = torchtext.models.XLMR_BASE_ENCODER
        model = xlmr_large.get_model(head=self.head)
        transformation = xlmr_large.transform
        transform_fn = transformation()
        return model, transform_fn

    def get_head(self):
        head = torchtext.models.RobertaClassificationHead(num_classes=2, input_dim=768)
        return head

    def get_dataset(self):
        train = SST2("./data/sst2", split="train")
        test = SST2("./data/sst2", split="dev")

        datapipes = {}
        datapipes["train"], datapipes["test"] = train, test
        return datapipes

    def apply_transform(self, row):
        text, label = row
        return (self.transform_fn(text), label)

    def tensor_batch(self, batch):
        tokens = batch["token_ids"]
        labels = batch["labels"]
        tokens_tensor = to_tensor(tokens, padding_value=self.pad_id)
        labels_tensor = torch.tensor(labels)
        return tokens_tensor, labels_tensor

    def predict(self, text, categories):
        self.model.eval()
        tokens = self.transform_fn(text)
        tokens_tensor = to_tensor(tokens, padding_value=self.pad_id)
        preds = self.model(tokens_tensor.unsqueeze(0).to("cuda"))

        # probabilities
        preds = torch.nn.functional.softmax(preds[0], dim=0)
        values, indices = torch.topk(preds, 1)

        return [
            {"label": categories[i], "value": v.item()} for i, v in zip(indices, values)
        ]

    def run(self):
        for k in self.datapipes.keys():
            self.datapipes[k] = self.datapipes[k].map(self.apply_transform)
            self.datapipes[k] = self.datapipes[k].batch(16)
            self.datapipes[k] = self.datapipes[k].rows2columnar(["token_ids", "labels"])
            self.datapipes[k] = self.datapipes[k].map(self.tensor_batch)

        dataloaders = {}
        dataloaders["train"] = DataLoader(
            self.datapipes["train"], batch_size=None, shuffle=True
        )
        dataloaders["test"] = DataLoader(self.datapipes["test"], batch_size=None)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        epochs = 1
        epoch_losses = []
        epoch_val_losses = []
        device = "cuda"
        writer = SummaryWriter("runs/script_experiment_1")

        self.model.to(device)

        for epoch in range(epochs):
            batch_losses = []
            for i, (tokens, labels) in tqdm(enumerate(dataloaders["train"])):
                self.model.train()
                tokens = tokens.to(device)
                labels = labels.to(device)

                preds = self.model(tokens)
                loss = loss_fn(preds, labels)
                loss.backward()

                batch_losses.append(loss.item())
                writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict={"training": loss.item()},
                    global_step=i,
                )

                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(sum(batch_losses) / len(batch_losses))
            val_losses = []

            with torch.inference_mode():
                for i, (tokens, labels) in enumerate(dataloaders["test"]):
                    self.model.eval()
                    tokens = tokens.to(device)
                    labels = labels.to(device)

                    preds = self.model(tokens)
                    loss = loss_fn(preds, labels)
                    val_losses.append(loss.item())

                    writer.add_scalars(
                        main_tag="loss",
                        tag_scalar_dict={"testing": loss.item()},
                        global_step=i,
                    )
                epoch_val_losses.append(sum(val_losses) / len(val_losses))

        # save plot of losses
        # plt.plot(epoch_losses, label="train")
        # plt.plot(epoch_val_losses, label="test")
        # plt.legend()
        # plt.savefig("losses.png")


if __name__ == "__main__":
    transformation = TransformToClass()
    transformation.run()
    categories = ["positive", "negative"]
    text = "I am really liking this course"
    output = transformation.predict(text, categories)
    print(output)
    text = "This course is too complicated"
    output = transformation.predict(text, categories)
    print(output)

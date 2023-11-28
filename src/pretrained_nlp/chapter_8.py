import torch
import torchtext
from torchtext.functional import to_tensor

roberta_base = torchtext.models.ROBERTA_LARGE_ENCODER
# without weights
roberta_classifier_head = torchtext.models.RobertaClassificationHead(
    num_classes=2, input_dim=1024
)

if __name__ == "__main__":
    input_batch = ["I am really liking this course!", "this course is too complicated"]

    # model
    model = roberta_base.get_model()
    model.eval()
    model.to("cuda")

    # model with a head
    model_with_head = roberta_base.get_model(head=roberta_classifier_head)
    model_with_head.eval()
    model_with_head.to("cuda")

    # transform
    transform = roberta_base.transform()

    # model_input
    model_input = transform(input_batch)

    # padding with 1, because is the special token to padding (<pad>)
    model_input = to_tensor(model_input, padding_value=1)

    # outputs
    output = model(model_input.to("cuda"))
    output_with_head = model_with_head(model_input.to("cuda"))

    print(output_with_head.shape)

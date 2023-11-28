from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

sentence = "Hello, how are you"

if __name__ == "__main__":
    tokenized = tokenizer(sentence, return_tensors="pt")
    print(tokenized)

    outputs = model(**tokenized)
    print(outputs["logits"].shape)

    probs = torch.nn.functional.softmax(outputs["logits"][0], dim=1)
    values, indices = torch.topk(probs, 10)
    print(values, indices)

    predictions = tokenizer.decode(indices[:, 1])
    print(predictions)

import torch

from dcase24t6.nn.hub import baseline_pipeline

sr = 44100
audio = torch.rand(1, sr * 15)

model = baseline_pipeline()
item = {"audio": audio, "sr": sr}
outputs = model(item)
candidate = outputs["candidates"][0]

print(candidate)

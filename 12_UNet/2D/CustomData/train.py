import torch
from tqdm import tqdm

def train_fn(loader, model, optimizer, loss_fn, device):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):

        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
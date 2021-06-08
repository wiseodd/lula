import torch
from tqdm import tqdm


@torch.no_grad()
def predict(test_loader, model, n_samples=20, apply_softmax=True, return_targets=False, delta=1, n_data=None):
    py = []
    targets = []
    count = 0

    for x, y in test_loader:
        if n_data is not None and count >= n_data:
            break

        x, y = delta*x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)
        count += len(x)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)

import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, optimizer, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
            f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def test(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc

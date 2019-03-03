import torch


def accuracy(output, target, topK=(1, )):
    with torch.no_grad():
        maxk = max(topK)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topK:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        if len(res) == 1:
            return res[0]
        return res


def mse_error(y_pred, y_true):
    batch_size = y_pred.size(0)
    mrs = torch.sum(torch.pow(y_pred.squeeze() - y_true.squeeze(), 2)) / batch_size
    return mrs



import torch
import logging
logger = logging.getLogger()


class PrintHook(object):
    """A helper for printing tensor shapes in a model."""
    def __init__(self, name):
        self.name = name

    def __call__(self, module, input, output):
        input_sizes = []
        output_sizes = []
        for x in input:
            input_sizes.append(str(x.size()))

        for y in output:
            output_sizes.append(str(y.size()))

        logger.info("{} : {} ==> {}".format(self.name, ', '.join(input_sizes), ', '.join(output_sizes)))


def print_forward_tensor(data_loader, net, max_batch_size=None):
    # print forward tensor shape
    inputs, labels = next(iter(data_loader))
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    device = torch.device(device)
    net = net.to(device)
    if max_batch_size is not None:
        num = min(max_batch_size, inputs.size(0))
    else:
        num = inputs.size(0)
    inputs = torch.autograd.Variable(inputs[:num, ...].to(device))
    print_forward_tensor_shape(net=net, input=inputs)


def print_forward_tensor_shape(net, input, **kwargs):
    """ Print tensor's shape in forward."""
    handlers = []
    for name, mod in net.named_modules():
        hand = mod.register_forward_hook(PrintHook(name))
        handlers.append(hand)
    with torch.no_grad():
        net(input, **kwargs)
    for hand in handlers:
        hand.remove()


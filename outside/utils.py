import torch
import torchvision
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms


def resize_tensor(input_tensors, h, w):
    """
    Credit: https://discuss.pytorch.org/t/resizing-any-simple-direct-way/10316/6
    TODO must be convert to cpu and back => slow, need improve
    :param input_tensors: [batch, channel, h, w]
    :param h:
    :param w:
    :return:
    """
    final_output = None
    batch_size, channel, height, width = input_tensors.shape
    input_tensors = torch.squeeze(input_tensors, 1)

    for img in input_tensors.cpu():
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = torchvision.transforms.Resize([h, w])(img_PIL)
        img_PIL = torchvision.transforms.ToTensor()(img_PIL)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    final_output = torch.unsqueeze(final_output, 1).reshape((batch_size, channel, h, w))
    return final_output.cuda()


def resize2d(img, size):
    return (adaptive_avg_pool2d(img, size)).data

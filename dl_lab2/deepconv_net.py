import torch.nn as nn


class DeepConvNet(nn.Module):

    def __init__(self, activation_func):
        super().__init__()

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=25,
                                kernel_size=(1, 5),
                                padding=0)

        channels = [25, 25, 50, 100, 200]
        kernel_sizes = [None, (2, 1), (1, 5), (1, 5), (1, 5)]

        for i in range(1, len(channels)):
            setattr(self, "conv_{0}".format(i), nn.Sequential(
                nn.Conv2d(in_channels=channels[i - 1],
                          out_channels=channels[i],
                          kernel_size=kernel_sizes[i],
                          padding=0),
                nn.BatchNorm2d(num_features=channels[i],
                               eps=1e-5,
                               momentum=0.1),
                activation_func,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=0.5)
            ))

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8600, out_features=2, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, input_data):
        for i in range(0, 5):
            conv = getattr(self, "conv_{0}".format(i))
            result = conv(input_data)
            input_data = result

        result = self.classification(result)
        return result

    @staticmethod
    def get_name():
        return "DeepConvNet"

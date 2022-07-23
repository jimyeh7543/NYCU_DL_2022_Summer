import torch.nn as nn


class EegNet(nn.Module):

    def __init__(self, activation_func):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=(1, 51),
                      stride=(1, 1),
                      padding=(0, 25),
                      bias=False), nn.BatchNorm2d(16),
            nn.BatchNorm2d(num_features=16,
                           eps=1e-5,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True)
        )

        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(2, 1),
                      stride=(1, 1),
                      groups=16,
                      bias=False),
            nn.BatchNorm2d(num_features=32,
                           eps=1e-5,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            activation_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(num_features=32,
                           eps=1e-5,
                           momentum=0.1,
                           affine=True,
                           track_running_stats=True),
            activation_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, input_data):
        first_conv_results = self.first_conv(input_data)
        depth_wise_conv_results = self.depth_wise_conv(first_conv_results)
        separable_conv_results = self.separable_conv(depth_wise_conv_results)
        return self.classification(separable_conv_results)

    @staticmethod
    def get_name():
        return "EEGNet"

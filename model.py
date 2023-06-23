from enum import Enum
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


NUM_GROUPS: int = 8


class NormalizationMethod(Enum):
    BATCH = 1
    LAYER = 2
    GROUP = 3


def normalizer(
    method: NormalizationMethod,
    out_channels: int,
) -> nn.BatchNorm2d | nn.GroupNorm:
    if method is NormalizationMethod.BATCH:
        return nn.BatchNorm2d(out_channels)
    elif method is NormalizationMethod.LAYER:
        return nn.GroupNorm(1, out_channels)
    elif method is NormalizationMethod.GROUP:
        return nn.GroupNorm(NUM_GROUPS, out_channels)
    else:
        raise ValueError("Invalid NormalizationMethod")


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
    ):
        """Initialize Layer [conv(3,3) + normalization + relu]

        Args:
            in_channels (int): Input Channel Size
            out_channels (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm_method (enum, optional): Type of normalization to be used. Defaults to NormalizationMethod.BATCH
        """
        super(ConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=padding,
                bias=False,
            ),
            normalizer(method=norm_method, out_channels=out_channels),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize Transition block [conv (1,1) + max pooling]

        Args:
            in_channels (int): Input Channel Size
            out_channels (int): Output Channel Size
        """
        super(TransBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.layer(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
    ) -> None:
        super(Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            ConvLayer(
                in_channels=3, out_channels=16, padding=1, norm_method=norm_method
            ),
            ConvLayer(
                in_channels=16, out_channels=32, padding=1, norm_method=norm_method
            ),
        )
        self.trans_block1 = TransBlock(in_channels=32, out_channels=16)

        self.conv_block2 = nn.Sequential(
            ConvLayer(
                in_channels=16, out_channels=16, padding=1, norm_method=norm_method
            ),
            ConvLayer(
                in_channels=16, out_channels=16, padding=1, norm_method=norm_method
            ),
            ConvLayer(
                in_channels=16, out_channels=32, padding=1, norm_method=norm_method
            ),
        )
        self.trans_block2 = TransBlock(in_channels=32, out_channels=16)

        self.conv_block3 = nn.Sequential(
            ConvLayer(
                in_channels=16, out_channels=16, padding=1, norm_method=norm_method
            ),
            ConvLayer(
                in_channels=16, out_channels=16, padding=1, norm_method=norm_method
            ),
            ConvLayer(
                in_channels=16, out_channels=32, padding=1, norm_method=norm_method
            ),
        )
        self.trans_block3 = TransBlock(in_channels=32, out_channels=16)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.out = nn.Conv2d(
            in_channels=16, out_channels=10, kernel_size=(1, 1), bias=False
        )

    def forward(self, x: Tensor):
        x = self.conv_block1(x)
        x = self.trans_block1(x)
        x = self.conv_block2(x)
        x = self.trans_block2(x)
        x = self.conv_block3(x)
        x = self.trans_block3(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=1)
    
#### Model Session 6 ########

class Session_6_model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
##### Model Session 7 ######
class Session_7_Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )  # 28 >> 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )  # 26 >> 24

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24

        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 12 >> 10

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 10 >> 10

        self.pool2 = nn.MaxPool2d(2, 2)  # 10 >> 5

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )  # 5 >> 5

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # output_size = 1

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        )

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.pool2(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.convblock7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



class Session_7_Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )  # 28 >> 26



        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 26 >> 24



        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24



        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 12 >> 10



        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 10 >> 8



        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 8 >> 6



        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 6 >> 4



        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 4 >> 2



        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        )  # 2 >> 1



    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    



class Session_7_Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )  # 28 >> 26



        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 26 >> 24



        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )  # 24 >> 24



        self.pool1 = nn.MaxPool2d(2, 2)  # 24 >> 12



        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 12 >> 10



        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )  # 10 >> 8



        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 8 >> 6



        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 6 >> 4



        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )  # 4 >> 2



        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(2, 2), padding=0, bias=False),
        )  # 2 >> 1

        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# ERA V1 Session 8

## Create and Train a Neural Network in Python

An implementation to create and train a simple neural network in python.

## Usage
### model.py

- First we have to import all the neccessary libraries.
  
```ruby
from enum import Enum
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
```

- normalizer() function contains methods for group normalizer, batch normalizer and layer normalizer.

```ruby
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
```

- Next we build a simple Neural Network.
For this, we define class **class Model()** and pass **nn.Module** as the parameter.

```ruby
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
```

- Create two functions inside the class to get our model ready. First is the **init()** and the second is the **forward()**.
- We need to instantiate the class for training the dataset. When we instantiate the class, the forward() function will get executed.

- Defined convolution layer and transition layer.

```ruby
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        norm_method: NormalizationMethod = NormalizationMethod.BATCH,
    ):
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
        x = self.layer(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
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
        x = self.layer(x)
        return x
```

 
### utils.py
- In this created two functions **visualize_data()** and **show_misclassified_images()**
  
- **visualize_data()** is used to visualize the input data.

```ruby
def visualize_data(
    loader,
    num_figures: int = 12,
    label: str = "",
    classes: List[str] = [],
):
    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()
    fig.suptitle(label)

    rows, cols = get_rows_cols(num_figures)

    for i in range(num_figures):
        plt.subplot(rows, cols, i + 1)
        plt.tight_layout()
        npimg = denormalize(batch_data[i].cpu().numpy().squeeze())
        label = (
            classes[batch_label[i]] if batch_label[i] < len(classes) else batch_label[i]
        )
        plt.imshow(npimg, cmap="gray")
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
```

- **show_misclassified_images()** is used to identify misclassified images.

```ruby
def show_misclassified_images(
    images: List[Tensor],
    predictions: List[int],
    labels: List[int],
    classes: List[str],
):
    assert len(images) == len(predictions) == len(labels)

    fig = plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        sub = fig.add_subplot(len(images) // 5, 5, i + 1)
        image = images[i]
        npimg = denormalize(image.cpu().numpy().squeeze())
        plt.imshow(npimg, cmap="gray")
        predicted = classes[predictions[i]]
        correct = classes[labels[i]]
        sub.set_title(
            "Correct class: {}\nPredicted class: {}".format(correct, predicted)
        )
    plt.tight_layout()
    plt.show()
```
### trainer.py

- train() funtion computes the prediction, traininng accuracy and loss

```ruby
class Trainer:
    def __init__(self, model, train_loader, optimizer, criterion, device) -> None:
        self.train_losses = []
        self.train_accuracies = []
        self.epoch_train_accuracies = []
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_history = []

    def train(self, epoch, use_l1=False, lambda_l1=0.01):
        self.model.train()

        lr_trend = []
        correct = 0
        processed = 0
        train_loss = 0

        pbar = tqdm(self.train_loader)

        for batch_id, (inputs, targets) in enumerate(pbar):
            # transfer to device
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Initialize gradients to 0
            self.optimizer.zero_grad()

            # Prediction
            outputs = self.model(inputs)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            l1 = 0
            if use_l1:
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1 * l1

            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            processed += len(inputs)

            pbar.set_description(
                desc=f"EPOCH = {epoch} | LR = {self.optimizer.param_groups[0]['lr']} | Loss = {loss.item():3.2f} | Batch = {batch_id} | Accuracy = {100*correct/processed:0.2f}"
            )
            self.train_accuracies.append(100 * correct / processed)

        # After all the batches are done, append accuracy for epoch
        self.epoch_train_accuracies.append(100 * correct / processed)

        self.lr_history.extend(lr_trend)
        return 100 * correct / processed, train_loss / len(self.train_loader), lr_trend
```


### tester.py

- And test() function calculates the loss and accuracy of the model

```ruby
class Tester:
    def __init__(self, model, test_loader, criterion, device) -> None:
        self.test_losses = []
        self.test_accuracies = []
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.model(inputs)
                loss = self.criterion(output, targets)

                test_loss += loss.item()

                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(targets.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )

        self.test_accuracies.append(100.0 * correct / len(self.test_loader.dataset))

        return 100.0 * correct / len(self.test_loader.dataset), test_loss

    def get_misclassified_images(self):
        self.model.eval()

        images: List[Tensor] = []
        predictions = []
        labels = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                _, preds = torch.max(output, 1)

                for i in range(len(preds)):
                    if preds[i] != target[i]:
                        images.append(data[i])
                        predictions.append(preds[i])
                        labels.append(target[i])

        return images, predictions, labels
```

### dataloader.py

- Loading CIFAR10 dataset
- **get_dataset()** will download the data.
- **get_loader()** will load the dataset.


### session_08_bn.ipynb

- Batch normalization is a method used to make training of artificial neural networks faster and more stable through normalization of the layers' inputs by re-centering and re-scaling.
  
- First have to load CIFAR10 dataset then we have to load train and test data.
  
```ruby
train_loader = cifar10.get_loader(transforms=train_transforms, train=True)
test_loader = cifar10.get_loader(transforms=train_transforms, train=False)
```

- Next visualizing train data.

```ruby
visualize_data(train_loader, 12, "Train Data", classes=cifar10.classes)
```
<img width="457" alt="image" src="https://github.com/GunaKoppula/ERA---Session-8/assets/61241928/9fd54c58-78d0-4cb5-bf28-ea23186cc478">

- Train the model with below parameter count and accuracy.

```ruby
Total params: 25,520
Trainable params: 25,520
Non-trainable params: 0
Total mult-adds (M): 17.60
==================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.57
Params size (MB): 0.10
Estimated Total Size (MB): 2.70
```

- Used 20 epochs to train the model.

```ruby
EPOCH = 1 | LR = 0.01 | Loss = 1.55 | Batch = 390 | Accuracy = 29.65: 100%|██████████| 391/391 [00:10<00:00, 38.97it/s]
Test set: Average loss: 0.0131, Accuracy: 3622/10000 (36.22%)
EPOCH = 2 | LR = 0.01 | Loss = 1.03 | Batch = 390 | Accuracy = 51.10: 100%|██████████| 391/391 [00:09<00:00, 40.46it/s]
Test set: Average loss: 0.0110, Accuracy: 4997/10000 (49.97%)
EPOCH = 3 | LR = 0.01 | Loss = 1.17 | Batch = 390 | Accuracy = 58.93: 100%|██████████| 391/391 [00:09<00:00, 39.39it/s]
Test set: Average loss: 0.0102, Accuracy: 5343/10000 (53.43%)
EPOCH = 4 | LR = 0.01 | Loss = 1.13 | Batch = 390 | Accuracy = 62.42: 100%|██████████| 391/391 [00:09<00:00, 39.32it/s]
Test set: Average loss: 0.0083, Accuracy: 6264/10000 (62.64%)
EPOCH = 5 | LR = 0.01 | Loss = 1.07 | Batch = 390 | Accuracy = 64.77: 100%|██████████| 391/391 [00:10<00:00, 39.09it/s]
Test set: Average loss: 0.0082, Accuracy: 6258/10000 (62.58%)
EPOCH = 6 | LR = 0.01 | Loss = 0.98 | Batch = 390 | Accuracy = 66.25: 100%|██████████| 391/391 [00:10<00:00, 38.80it/s]
Test set: Average loss: 0.0076, Accuracy: 6520/10000 (65.20%)
EPOCH = 7 | LR = 0.01 | Loss = 1.11 | Batch = 390 | Accuracy = 67.82: 100%|██████████| 391/391 [00:09<00:00, 40.37it/s]
Test set: Average loss: 0.0074, Accuracy: 6710/10000 (67.10%)
EPOCH = 8 | LR = 0.01 | Loss = 0.89 | Batch = 390 | Accuracy = 68.99: 100%|██████████| 391/391 [00:09<00:00, 40.52it/s]
Test set: Average loss: 0.0071, Accuracy: 6742/10000 (67.42%)
EPOCH = 9 | LR = 0.01 | Loss = 0.85 | Batch = 390 | Accuracy = 70.05: 100%|██████████| 391/391 [00:09<00:00, 40.30it/s]
Test set: Average loss: 0.0066, Accuracy: 6994/10000 (69.94%)
EPOCH = 10 | LR = 0.01 | Loss = 0.75 | Batch = 390 | Accuracy = 71.00: 100%|██████████| 391/391 [00:09<00:00, 40.38it/s]
Test set: Average loss: 0.0066, Accuracy: 7019/10000 (70.19%)
EPOCH = 11 | LR = 0.001 | Loss = 0.53 | Batch = 390 | Accuracy = 73.48: 100%|██████████| 391/391 [00:10<00:00, 39.03it/s]
Test set: Average loss: 0.0061, Accuracy: 7229/10000 (72.29%)
EPOCH = 12 | LR = 0.001 | Loss = 0.71 | Batch = 390 | Accuracy = 74.18: 100%|██████████| 391/391 [00:09<00:00, 39.51it/s]
Test set: Average loss: 0.0059, Accuracy: 7308/10000 (73.08%)
EPOCH = 13 | LR = 0.001 | Loss = 0.71 | Batch = 390 | Accuracy = 74.53: 100%|██████████| 391/391 [00:10<00:00, 37.55it/s]
Test set: Average loss: 0.0059, Accuracy: 7373/10000 (73.73%)
EPOCH = 14 | LR = 0.001 | Loss = 0.85 | Batch = 390 | Accuracy = 74.40: 100%|██████████| 391/391 [00:09<00:00, 39.87it/s]
Test set: Average loss: 0.0059, Accuracy: 7359/10000 (73.59%)
EPOCH = 15 | LR = 0.001 | Loss = 0.86 | Batch = 390 | Accuracy = 74.88: 100%|██████████| 391/391 [00:09<00:00, 40.22it/s]
Test set: Average loss: 0.0058, Accuracy: 7363/10000 (73.63%)
EPOCH = 16 | LR = 0.001 | Loss = 0.58 | Batch = 390 | Accuracy = 74.78: 100%|██████████| 391/391 [00:09<00:00, 39.50it/s]
Test set: Average loss: 0.0059, Accuracy: 7373/10000 (73.73%)
EPOCH = 17 | LR = 0.001 | Loss = 0.67 | Batch = 390 | Accuracy = 75.04: 100%|██████████| 391/391 [00:09<00:00, 40.31it/s]
Test set: Average loss: 0.0059, Accuracy: 7357/10000 (73.57%)
EPOCH = 18 | LR = 0.001 | Loss = 0.84 | Batch = 390 | Accuracy = 75.18: 100%|██████████| 391/391 [00:09<00:00, 40.22it/s]
Test set: Average loss: 0.0060, Accuracy: 7360/10000 (73.60%)
EPOCH = 19 | LR = 0.001 | Loss = 0.92 | Batch = 390 | Accuracy = 75.21: 100%|██████████| 391/391 [00:09<00:00, 39.13it/s]
Test set: Average loss: 0.0058, Accuracy: 7441/10000 (74.41%)
EPOCH = 20 | LR = 0.001 | Loss = 0.79 | Batch = 390 | Accuracy = 75.50: 100%|██████████| 391/391 [00:09<00:00, 39.95it/s]
Test set: Average loss: 0.0058, Accuracy: 7392/10000 (73.92%)
```

- Visualizing some of the images that are classified wrong. This will help us to identify any weaknesses in our classification algorithm.
  
<img width="697" alt="image" src="https://github.com/GunaKoppula/ERA---Session-8/assets/61241928/c8e1d839-f485-48b2-b6cf-169f54dcbd47">


### session_08_gn.ipynb

- Group Normalization is a normalization layer that divides channels into groups and normalizes the features within each group.

- Train the model with below parameter count and accuracy.

```ruby
Total params: 25,520
Trainable params: 25,520
Non-trainable params: 0
Total mult-adds (M): 17.60
==================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.57
Params size (MB): 0.10
Estimated Total Size (MB): 2.70
```

- Used 20 epochs to train the model.

```ruby
EPOCH = 1 | LR = 0.01 | Loss = 2.05 | Batch = 390 | Accuracy = 19.70: 100%|██████████| 391/391 [00:09<00:00, 40.04it/s]
Test set: Average loss: 0.0145, Accuracy: 2782/10000 (27.82%)
EPOCH = 2 | LR = 0.01 | Loss = 1.43 | Batch = 390 | Accuracy = 36.26: 100%|██████████| 391/391 [00:09<00:00, 39.61it/s]
Test set: Average loss: 0.0120, Accuracy: 4267/10000 (42.67%)
EPOCH = 3 | LR = 0.01 | Loss = 1.30 | Batch = 390 | Accuracy = 46.03: 100%|██████████| 391/391 [00:09<00:00, 40.49it/s]
Test set: Average loss: 0.0121, Accuracy: 4209/10000 (42.09%)
EPOCH = 4 | LR = 0.01 | Loss = 1.09 | Batch = 390 | Accuracy = 53.15: 100%|██████████| 391/391 [00:09<00:00, 40.38it/s]
Test set: Average loss: 0.0095, Accuracy: 5717/10000 (57.17%)
EPOCH = 5 | LR = 0.01 | Loss = 0.91 | Batch = 390 | Accuracy = 57.02: 100%|██████████| 391/391 [00:09<00:00, 41.01it/s]
Test set: Average loss: 0.0097, Accuracy: 5484/10000 (54.84%)
EPOCH = 6 | LR = 0.01 | Loss = 1.36 | Batch = 390 | Accuracy = 59.53: 100%|██████████| 391/391 [00:10<00:00, 38.46it/s]
Test set: Average loss: 0.0097, Accuracy: 5584/10000 (55.84%)
EPOCH = 7 | LR = 0.01 | Loss = 0.87 | Batch = 390 | Accuracy = 61.82: 100%|██████████| 391/391 [00:10<00:00, 38.80it/s]
Test set: Average loss: 0.0083, Accuracy: 6264/10000 (62.64%)
EPOCH = 8 | LR = 0.01 | Loss = 0.95 | Batch = 390 | Accuracy = 63.39: 100%|██████████| 391/391 [00:09<00:00, 39.13it/s]
Test set: Average loss: 0.0078, Accuracy: 6433/10000 (64.33%)
EPOCH = 9 | LR = 0.01 | Loss = 0.94 | Batch = 390 | Accuracy = 64.40: 100%|██████████| 391/391 [00:09<00:00, 39.67it/s]
Test set: Average loss: 0.0076, Accuracy: 6517/10000 (65.17%)
EPOCH = 10 | LR = 0.01 | Loss = 0.80 | Batch = 390 | Accuracy = 65.50: 100%|██████████| 391/391 [00:10<00:00, 37.72it/s]
Test set: Average loss: 0.0071, Accuracy: 6790/10000 (67.90%)
EPOCH = 11 | LR = 0.001 | Loss = 0.97 | Batch = 390 | Accuracy = 69.46: 100%|██████████| 391/391 [00:09<00:00, 39.95it/s]
Test set: Average loss: 0.0067, Accuracy: 6936/10000 (69.36%)
EPOCH = 12 | LR = 0.001 | Loss = 0.93 | Batch = 390 | Accuracy = 70.18: 100%|██████████| 391/391 [00:09<00:00, 39.39it/s]
Test set: Average loss: 0.0067, Accuracy: 6950/10000 (69.50%)
EPOCH = 13 | LR = 0.001 | Loss = 0.81 | Batch = 390 | Accuracy = 70.53: 100%|██████████| 391/391 [00:09<00:00, 40.48it/s]
Test set: Average loss: 0.0067, Accuracy: 6997/10000 (69.97%)
EPOCH = 14 | LR = 0.001 | Loss = 1.00 | Batch = 390 | Accuracy = 70.42: 100%|██████████| 391/391 [00:10<00:00, 38.35it/s]
Test set: Average loss: 0.0068, Accuracy: 6962/10000 (69.62%)
EPOCH = 15 | LR = 0.001 | Loss = 0.85 | Batch = 390 | Accuracy = 70.69: 100%|██████████| 391/391 [00:09<00:00, 40.13it/s]
Test set: Average loss: 0.0067, Accuracy: 6987/10000 (69.87%)
EPOCH = 16 | LR = 0.001 | Loss = 0.97 | Batch = 390 | Accuracy = 70.97: 100%|██████████| 391/391 [00:10<00:00, 38.30it/s]
Test set: Average loss: 0.0066, Accuracy: 7041/10000 (70.41%)
EPOCH = 17 | LR = 0.001 | Loss = 0.64 | Batch = 390 | Accuracy = 71.11: 100%|██████████| 391/391 [00:09<00:00, 40.14it/s]
Test set: Average loss: 0.0065, Accuracy: 7002/10000 (70.02%)
EPOCH = 18 | LR = 0.001 | Loss = 0.82 | Batch = 390 | Accuracy = 71.37: 100%|██████████| 391/391 [00:09<00:00, 40.02it/s]
Test set: Average loss: 0.0067, Accuracy: 6979/10000 (69.79%)
EPOCH = 19 | LR = 0.001 | Loss = 0.74 | Batch = 390 | Accuracy = 71.29: 100%|██████████| 391/391 [00:09<00:00, 40.28it/s]
Test set: Average loss: 0.0065, Accuracy: 7094/10000 (70.94%)
EPOCH = 20 | LR = 0.001 | Loss = 0.81 | Batch = 390 | Accuracy = 71.50: 100%|██████████| 391/391 [00:09<00:00, 40.17it/s]
Test set: Average loss: 0.0065, Accuracy: 7065/10000 (70.65%)
```

- Visualizing some of the images that are classified wrong. This will help us to identify any weaknesses in our classification algorithm.

<img width="694" alt="image" src="https://github.com/GunaKoppula/ERA---Session-8/assets/61241928/c636dce6-fd3b-4fd9-bc8d-6339b19b80d4">


### session_08_ln.ipynb

- In layer normalization, all neurons in a particular layer effectively have the same distribution across all features for a given input.

- Train the model with below parameter count and accuracy.

```ruby
Total params: 25,520
Trainable params: 25,520
Non-trainable params: 0
Total mult-adds (M): 17.60
==================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 2.57
Params size (MB): 0.10
Estimated Total Size (MB): 2.70
```

- Used 20 epochs to train the model.

```ruby
EPOCH = 1 | LR = 0.01 | Loss = 1.79 | Batch = 390 | Accuracy = 18.45: 100%|██████████| 391/391 [00:09<00:00, 42.41it/s]
Test set: Average loss: 0.0149, Accuracy: 2678/10000 (26.78%)
EPOCH = 2 | LR = 0.01 | Loss = 1.73 | Batch = 390 | Accuracy = 30.16: 100%|██████████| 391/391 [00:09<00:00, 40.25it/s]
Test set: Average loss: 0.0136, Accuracy: 3462/10000 (34.62%)
EPOCH = 3 | LR = 0.01 | Loss = 1.55 | Batch = 390 | Accuracy = 40.19: 100%|██████████| 391/391 [00:09<00:00, 39.51it/s]
Test set: Average loss: 0.0125, Accuracy: 3943/10000 (39.43%)
EPOCH = 4 | LR = 0.01 | Loss = 1.26 | Batch = 390 | Accuracy = 46.78: 100%|██████████| 391/391 [00:09<00:00, 40.95it/s]
Test set: Average loss: 0.0107, Accuracy: 4867/10000 (48.67%)
EPOCH = 5 | LR = 0.01 | Loss = 1.08 | Batch = 390 | Accuracy = 50.19: 100%|██████████| 391/391 [00:08<00:00, 43.77it/s]
Test set: Average loss: 0.0099, Accuracy: 5420/10000 (54.20%)
EPOCH = 6 | LR = 0.01 | Loss = 1.28 | Batch = 390 | Accuracy = 54.00: 100%|██████████| 391/391 [00:09<00:00, 42.30it/s]
Test set: Average loss: 0.0100, Accuracy: 5430/10000 (54.30%)
EPOCH = 7 | LR = 0.01 | Loss = 1.24 | Batch = 390 | Accuracy = 57.05: 100%|██████████| 391/391 [00:08<00:00, 44.52it/s]
Test set: Average loss: 0.0111, Accuracy: 5044/10000 (50.44%)
EPOCH = 8 | LR = 0.01 | Loss = 1.28 | Batch = 390 | Accuracy = 58.44: 100%|██████████| 391/391 [00:08<00:00, 44.98it/s]
Test set: Average loss: 0.0096, Accuracy: 5572/10000 (55.72%)
EPOCH = 9 | LR = 0.01 | Loss = 1.19 | Batch = 390 | Accuracy = 60.54: 100%|██████████| 391/391 [00:08<00:00, 44.89it/s]
Test set: Average loss: 0.0095, Accuracy: 5523/10000 (55.23%)
EPOCH = 10 | LR = 0.01 | Loss = 0.86 | Batch = 390 | Accuracy = 62.22: 100%|██████████| 391/391 [00:08<00:00, 45.13it/s]
Test set: Average loss: 0.0079, Accuracy: 6384/10000 (63.84%)
EPOCH = 11 | LR = 0.001 | Loss = 0.95 | Batch = 390 | Accuracy = 66.36: 100%|██████████| 391/391 [00:09<00:00, 42.38it/s]
Test set: Average loss: 0.0074, Accuracy: 6645/10000 (66.45%)
EPOCH = 12 | LR = 0.001 | Loss = 1.02 | Batch = 390 | Accuracy = 67.01: 100%|██████████| 391/391 [00:09<00:00, 41.76it/s]
Test set: Average loss: 0.0074, Accuracy: 6561/10000 (65.61%)
EPOCH = 13 | LR = 0.001 | Loss = 0.98 | Batch = 390 | Accuracy = 67.50: 100%|██████████| 391/391 [00:09<00:00, 42.12it/s]
Test set: Average loss: 0.0073, Accuracy: 6700/10000 (67.00%)
EPOCH = 14 | LR = 0.001 | Loss = 0.59 | Batch = 390 | Accuracy = 67.67: 100%|██████████| 391/391 [00:08<00:00, 43.56it/s]
Test set: Average loss: 0.0072, Accuracy: 6734/10000 (67.34%)
EPOCH = 15 | LR = 0.001 | Loss = 0.80 | Batch = 390 | Accuracy = 67.75: 100%|██████████| 391/391 [00:08<00:00, 44.70it/s]
Test set: Average loss: 0.0073, Accuracy: 6699/10000 (66.99%)
EPOCH = 16 | LR = 0.001 | Loss = 0.94 | Batch = 390 | Accuracy = 68.12: 100%|██████████| 391/391 [00:08<00:00, 43.77it/s]
Test set: Average loss: 0.0070, Accuracy: 6850/10000 (68.50%)
EPOCH = 17 | LR = 0.001 | Loss = 0.91 | Batch = 390 | Accuracy = 68.52: 100%|██████████| 391/391 [00:08<00:00, 44.89it/s]
Test set: Average loss: 0.0072, Accuracy: 6709/10000 (67.09%)
EPOCH = 18 | LR = 0.001 | Loss = 0.74 | Batch = 390 | Accuracy = 68.27: 100%|██████████| 391/391 [00:09<00:00, 41.31it/s]
Test set: Average loss: 0.0070, Accuracy: 6838/10000 (68.38%)
EPOCH = 19 | LR = 0.001 | Loss = 0.86 | Batch = 390 | Accuracy = 68.64: 100%|██████████| 391/391 [00:10<00:00, 38.30it/s]
Test set: Average loss: 0.0072, Accuracy: 6738/10000 (67.38%)
EPOCH = 20 | LR = 0.001 | Loss = 0.99 | Batch = 390 | Accuracy = 68.65: 100%|██████████| 391/391 [00:11<00:00, 35.28it/s]
Test set: Average loss: 0.0071, Accuracy: 6804/10000 (68.04%)
```

- Visualizing some of the images that are classified wrong. This will help us to identify any weaknesses in our classification algorithm.

<img width="692" alt="image" src="https://github.com/GunaKoppula/ERA---Session-8/assets/61241928/309a5be9-5bfd-4d64-82fc-d03cba7eec1f">

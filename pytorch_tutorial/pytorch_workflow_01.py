import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

'''
Ref - https://youtu.be/V_xro1bcAuA?t=15869
Github - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/01_pytorch_workflow_video.ipynb
'''

print(torch.__version__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Data Preparing - Linear regression
# Known parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias



#
# plt.plot(X,y)
# plt.show()
# Split Train / Test
train_split = int(0.8 * len(X))
X_train, y_train  = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
print('Train',X_train.shape, y_train.shape)
print('Test',X_test.shape, y_test.shape)


def plot_loss(epoch_count, train_loss_values, test_loss_values):
    plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label="Train loss")
    plt.plot(epoch_count, np.array(torch.tensor(test_loss_values).numpy()), label="Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show();


def plot_predictions(train_data=X_train.cpu(),
                     train_labels=y_train.cpu(),
                     test_data=X_test.cpu(),
                     test_labels=y_test.cpu(),
                     predictions=None):
    """
    Plots training data, test data and compares predictions
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return:
    """
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels,c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data" )

    if predictions is not None:
        plt.scatter(test_data,predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size":14})
    plt.show()


#plot_predictions();

# Build Model
# Linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

# Pytorch model building
class LinearRegressionModelV2(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)



torch.manual_seed(42)

# create instance
#model_0 = LinearRegressionModel()
model_1 = LinearRegressionModelV2()
model_1.to(device)
#print('model_0',model_0.state_dict())
print('model_1',model_1.state_dict())

# Check model's predictive power
with torch.inference_mode():
    y_preds = model_1(X_test.to(device))

y_preds = y_preds.cpu()
print(y_preds)
plot_predictions(predictions=y_preds);

# Loss function - decides how bad your model is from predictions
# Optimizer - Based on loss , adjusts the parameters
# Training Loop
# Testing Loop

# setup loss
loss_fn = nn.L1Loss()

# setup optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.01)

# Building a training loop & testing loop
# 0. Loop through data
# 1. forward pass
# 2. calculate loss
# 3. Optimizer zero grad
# 4. Loss backward - gradients
# 5. Optimizer step - adjust model parameters

epochs = 250
epoch_count = []
loss_values = []
test_loss_values = []
print(model_1.state_dict())
for epoch in range(epochs):
    # 0. set model to train mode
    model_1.train() # all params requires gradient

    # 1.forward pass
    y_pred = model_1(X_train)

    # 2. calculate loss ( input , target)
    loss = loss_fn(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Perform backprop wrt parameters
    loss.backward()

    # 5. step optimizer ( perform gradient descent)
    optimizer.step()

    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print('Epoch', epoch, 'Loss:', loss, 'Test loss:', test_loss)
        print(model_1.state_dict())

torch.save(obj=model_1.state_dict(), f="01_pytorch_workflow_model_0.pth")
plot_loss(epoch_count, loss_values, test_loss_values)

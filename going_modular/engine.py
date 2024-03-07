import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, List


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_function: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X_train, y_train) in enumerate(data_loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_logits = model(X_train)
        loss = loss_function(y_logits, y_train)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1)
        train_acc += (y_predict == y_train).sum().item()/len(y_predict)
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_function: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(data_loader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_logits = model(X_test)
            loss = loss_function(y_logits, y_test)
            test_loss += loss.item()
            y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1)
            test_acc += (y_predict == y_test).sum().item()/len(y_predict)
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_function: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List]:

    writer = SummaryWriter()
    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model,
                                           data_loader=train_dataloader,
                                           loss_function=loss_function,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_function=loss_function,
                                        device=device)
        print(f"""
        Epoch {epoch} 
            Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}
            Test Loss : {test_loss:.4f} | Test Accuracy : {test_acc:.4f}""")
        result['train_loss'].append(train_loss)
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss)
        result['test_acc'].append(test_acc)
        writer.add_scalars(main_tag='Loss',
                           tag_scalar_dict={
                               'train_loss': train_loss,
                               'test_loss': test_loss
                           },
                           global_step=epoch)
        writer.add_scalars(main_tag='Accuracy',
                           tag_scalar_dict={
                               'train_acc': train_acc,
                               'test_acc': test_acc
                           },
                           global_step=epoch)
    writer.close()
    return result

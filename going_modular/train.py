from pathlib import Path
from torchvision import transforms
import torch
import utils
import engine
import model_builder
import data_setup
import argparse

parser = argparse.ArgumentParser(description='Get some hyperparameters.')
parser.add_argument('--num_epochs',
                    default=3,
                    type=int,
                    help='number of epochs of model train')
parser.add_argument('--lr',
                    default=0.01,
                    type=float,
                    help='learning rate of model')
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = args.lr

train_dir = Path('./data/train')
test_dir = Path('./data/test')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
classes_name, train_dataloader, test_dataloader = data_setup.create_dataloader(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transforms=data_transforms,
                                                                               batch_size=BATCH_SIZE)
model = model_builder.CNNModel(input_shape=3,
                               hidden_shape=HIDDEN_UNITS,
                               output_shape=len(classes_name)).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_function=loss_function,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
utils.save_model(model=model,
                 target_dir='model',
                 model_name='cnn.pth')

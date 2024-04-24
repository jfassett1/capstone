from dataset import TweetDataModule
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,SGD
import pathlib


class Model1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(768, 512)
        self.layer2 = nn.Linear(512, 350)
        self.layer3 = nn.Linear(350, 300)
        self.layer4 = nn.Linear(300, 250)
        self.layer5 = nn.Linear(250, 100)
        self.layer6 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)

    def training_step(self, batch, batch_idx):
        # Assuming batch consists of features and labels
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # self.log('train_loss', loss,on_epoch=True,on_step=True,prog_bar=True)    
        if (batch_idx + 1) % 150 == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss,prog_bar=True)
        self.log('Val Accuracy')
    
data_dir = pathlib.Path(__file__).parent.parent/"data"

# Instantiate the model
model = Model1()

# Instantiate the data module
data_module = TweetDataModule(data_dir/"dataset")

# Set up the trainer with desired configurations
trainer = pl.Trainer(max_epochs=10, 
                     devices=1,
                     val_check_interval=0.25)

# Train the model
trainer.fit(model, data_module)

# Optionally, run test data through the trained model
trainer.test(model, data_module.test_dataloader())
from dataset import TweetDataModule
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LinearLR
import pathlib
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor



def binary_accuracy(logits, labels, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    correct = (preds == labels).float()
    accuracy = correct.mean()
    return accuracy

class Model1(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(778, 1052)
        self.layer2 = nn.Linear(1052, 2000)
        self.layer3 = nn.Linear(2000, 2000)
        self.layer4 = nn.Linear(2000, 1052)
        self.layer5 = nn.Linear(1052, 100)
        self.layer6 = nn.Linear(100, 1)
        self.loss_fn = F.binary_cross_entropy_with_logits
        # self.automatic_optimization = False

        # self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)


    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()
    #     opt.zero_grad()
    #     x, y = batch
    #     y = y.float()
    #     logits = self(x)
    #     loss = self.loss_fn(logits.squeeze(1), y)
    #     self.manual_backward(loss)
    #     self.log()

    #     opt.step()
    def training_step(self, batch, batch_idx):
        # Assuming batch consists of features and labels
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(1), y)
        self.log('train_loss', loss,on_step=True,prog_bar=True)    
        # if (batch_idx + 1) % 150 == 0:
        #     self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=5e-5)
        scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return {"optimizer":optimizer,"lr_scheduler":scheduler_dict,"lr_monitor":lr_monitor}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fn(logits.squeeze(1), y)
        # acc = binary_accuracy(logits, y)
        self.log('val_loss', loss,prog_bar=True)
        # self.log('val_acc',acc,prog_bar=True)
    
data_dir = pathlib.Path(__file__).parent.parent/"data"

model = Model1()
data_module = TweetDataModule(data_dir/"dataset",batch_size=50)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(max_epochs=10, 
                     devices=2,
                     val_check_interval=0.05,
                     callbacks=[lr_monitor],
                     detect_anomaly=True)
trainer.fit(model, data_module)

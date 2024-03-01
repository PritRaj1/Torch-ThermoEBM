import torch
import configparser
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import loggers as pl_loggers
import lightning as L

from src.pipeline.Trainer import LatentEBM_Model
from src.utils.helper_functions import get_data
from src.pipeline.metrics import loss_FLOPS

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')

NUM_EPOCHS = int(parser["PIPELINE"]["NUM_EPOCHS"])
BATCH_SIZE = int(parser["PIPELINE"]["BATCH_SIZE"])

DATA_NAME = parser["PIPELINE"]["DATASET"]
NUM_TRAIN = int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
NUM_VAL = int(parser["PIPELINE"]["NUM_VAL_DATA"])

TEMP_POWER = int(parser["TEMP"]["TEMP_POWER"])

dataset, val_dataset, IMAGE_DIM = get_data(DATA_NAME)

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, range(NUM_TRAIN))
val_data = torch.utils.data.Subset(val_dataset, range(NUM_VAL))

# Split dataset
test_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=31)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=31)

class PerformanceCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        x = next(iter(test_loader))[0].to(device)

        loss_FLOPS(pl_module, x)

NUM_EXPERIMENTS = int(parser["PIPELINE"]["NUM_EXPERIMENTS"])
for idx in range(0, NUM_EXPERIMENTS):
    model = LatentEBM_Model(IMAGE_DIM)

    csv_logger = pl_loggers.CSVLogger(
        save_dir=f"logs/{DATA_NAME}/p{TEMP_POWER}", 
        name=f"experiment_{idx}"
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=f"logs/{DATA_NAME}/p{TEMP_POWER}", 
        name=f"experiment_{idx}"
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        devices='auto',
        accelerator='auto',
        profiler='simple',
        logger=[tb_logger, csv_logger],
        callbacks=[PerformanceCallback()],
        enable_progress_bar=True        
        )
    
    # Train the model
    trainer.fit(model, train_dataloaders=test_loader, val_dataloaders=val_loader)

    # Free up memory
    del model
    del trainer
    del csv_logger
    del tb_logger
    torch.cuda.empty_cache()

print("DONE")
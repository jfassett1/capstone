from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from utils import CLSdata2
import numpy as np
from tqdm import tqdm
import pathlib

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    csv_file = 'data/transformed/data.csv'
    npy_dir = 'data/embedded/'
    data_dir = pathlib.Path(__file__).parent.parent / "data"

    dataset = CLSdata2(csv_file=csv_file, npy_dir=npy_dir)
    # print(type(dataset[5][0].numpy()))
    # exit()
    labels = np.array(dataset.labels[:-3])
    indices = np.arange(len(labels))
    # Assuming X is your features and y is the labels
    X_train, X_test, y_train, y_test = train_test_split(
        indices.reshape(-1,1),  # Reshaping indices to fit expected shape
        labels,                # The labels
        test_size=0.1,         # 30% of the data for testing
        random_state=42        # For reproducibility
    )

    # print(labels)
    # Apply SMOTE only to the training data to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(
        X_train_res, y_train_res, test_size=0.2, random_state=42
    )

    def save2np(X,y,batches=30,which='val'):
        directory = data_dir / which
        if not directory.exists():
            directory.mkdir()
        np.save(data_dir/which/"labels.npy",y)
        batch_size = int(np.ceil(len(y) / batches))
        batch_num = 0
        batch_list = []
        for i, val in enumerate(tqdm(X, colour="red", ncols=300, bar_format='{l_bar}{bar:10}{r_bar}')):
            batch_list.append(dataset[int(val)][0].numpy())
            if (i + 1) % batch_size == 0 or i == len(X) - 1:
                np.save(data_dir/which/f"embedded_data{batch_num}.npy",np.vstack(batch_list))
                batch_list = []
                # print(rf"Batch {batch_num} complete.",end='',flush=True)
                batch_num += 1
                tqdm.write(str(f"Batch Number: {batch_num}"))

            

save2np(X_val_res,y_val_res,which="val")
# save2np(X_test,y_test,which="test")
# save2np(X_train_res,y_train_res,which="train")

    # print(y_train_res.sum()/len(y_train_res))

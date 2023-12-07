import sys
import torch
import traceback
from src.utils.trainer import Trainer
from src.dataset.hmdb_simp import HMDBDataset
from src.model.timesformer import Timesformer
from sklearn.model_selection import train_test_split

def main():
    try:      
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(sys.argv)
        hmdb_dataset_path = sys.argv[1] #"/mnt/fast/nobackup/users/jm02710/dataset/HMDB_simp/"

        dataset = HMDBDataset(hmdb_dataset_path, numberOfFrames=16, imageSize=448)
                
        model = Timesformer(n_classes=25, model_path="pre-trained-models\TimeSformer_divST_16x16_448_K400.pyth")
        model.to(device)

        train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

        learning_rate = 0.001
        batch_size = 16
        trainer = Trainer(model=model, batch_size=batch_size, device=device, learning_rate=learning_rate,test_set=test_set, train_set=train_set)


        num_epoch = 4  
        for epoch in range(num_epoch):
            print(f"Epoch: {epoch}/{num_epoch};")
            trainer.train()

        model_file = "train-4e.model"
        torch.save(model, model_file)

        trainer.test()
    except Exception as e:
        print(e)        
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
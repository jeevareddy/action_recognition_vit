import os
import math
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as T

def load_video_label_mapping(path):
    labels = []
    videos_paths = []
    labels_ref = list(os.listdir(path))
    for action in tqdm(labels_ref):
        for video_folder in os.listdir(f"{path}{action}"):
            videos_paths.append(f"{path}{action}/{video_folder}")
            labels.append(labels_ref.index(action))
    return (videos_paths, labels, labels_ref)

class HMDBDataset(Dataset):
    def __init__(self, videos_paths: list[str], labels, numberOfFrames: int = 8, imageSize: int = 224):
        self.imageSize = imageSize
        self.numberOfFrames = numberOfFrames
        self.imgTransform = T.Compose(
            [T.ToPILImage(), T.Resize((imageSize, imageSize)), T.ToTensor()])
        self.videos_paths = videos_paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        video_folder = self.videos_paths[idx]

        video_frames = os.listdir(video_folder)
        if len(video_frames) >= self.numberOfFrames:
            return self.loadVideo(video_folder), torch.tensor(self.labels[idx])
        #Remove the video if the number of frames is insufficient
        del self.videos_paths[idx]
        del self.labels[idx]
        return self.__getitem__(idx)

    def loadVideo(self, video_folder):
        video_frames = os.listdir(video_folder)
        sample_frames_indices = np.linspace(
            0, len(video_frames)-1, num=self.numberOfFrames).astype(np.int64)
        video = [
            self.imgTransform(
                np.asarray(
                    Image.open(
                        f"{video_folder}/{str(video_frames[frame_index])}"
                    )
                )
            )
            for frame_index in sample_frames_indices
        ]
        return torch.stack(video)

    # def loadHMDBDataset(self, path):
    #     self.videos = []
    #     self.labels = []

    #     self.labels_ref = list(os.listdir(path))

    #     for action in tqdm(self.labels_ref):
    #         for video_folder in os.listdir(f"{path}{action}"):
    #             video_frames = os.listdir(f"{path}{action}/{video_folder}")
    #             sample_frames_indices = np.linspace(
    #                 0, len(video_frames)-1, num=self.numberOfFrames).astype(np.int64)
    #             video = [
    #                 self.imgTransform(
    #                     np.asarray(
    #                         Image.open(
    #                             f"{path}{action}/{video_folder}/{str(video_frames[frame_index])}"
    #                         )
    #                     )
    #                 )
    #                 for frame_index in sample_frames_indices
    #             ]
    #             self.videos.append(torch.stack(video))
    #             self.labels.append(torch.tensor(self.labels_ref.index(action)))

    #     self.labels = torch.tensor(self.labels)
    #     self.videos = torch.stack(self.videos)
    #     print(self.labels.size(), self.videos.size())
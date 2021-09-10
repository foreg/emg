from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from config import DATA_PATH


class OmniglotDataset(Dataset):
    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def __getitem__(self, item):
        instance = io.imread(self.datasetid_to_filepath[item])
        # Reindex to channels first format as supported by pytorch
        instance = instance[np.newaxis, :, :]

        # Normalise to 0-1
        instance = (instance - instance.min()) / (instance.max() - instance.min())

        label = self.datasetid_to_class_id[item]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            Omniglot dataset dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/Omniglot/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            alphabet = root.split('/')[-2]
            class_name = '{}.{}'.format(alphabet, root.split('/')[-1])

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'alphabet': alphabet,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class EMGDataset(Dataset):

    def __getitem__(self, index):
        item = self.datasetid_to_filepath[index]
        instance = pd.read_csv(item, engine='python')
        instance = instance.drop('8', axis=1)
        instance = instance.values
        instance = instance.reshape(8, 20, 20)

        # Normalise to 0-1
        instance = (instance - instance.min()) / \
            (instance.max() - instance.min())

        label = self.datasetid_to_class_id[index]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(
            lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str):
        data = []
        data_path = DATA_PATH + '/emg/{}'.format(subset)
        for root, folders, fls in os.walk(data_path):
            if len(fls) == 0:
                continue

            for f in fls:
                if f.endswith("csv"):
                    fp = data_path + "/" + f
                    csv_data = pd.read_csv(fp, engine='python')
                    # print(fp)
                    class_num = csv_data['8'][0]
                    data.append({
                        'subset': subset,
                        'class_name': class_num,
                        'filepath': fp
                    })
        return data


class MultipleEMGDatasetFirst(Dataset):

    def __getitem__(self, index):
        item = self.datasetid_to_filepath[index]
        instance = pd.read_csv(item, engine='python')
        instance = instance.drop('16', axis=1)
        instance = instance.values
        instance = instance.reshape(16, 20, 20)

        # Normalise to 0-1
        instance = (instance - instance.min()) / \
            (instance.max() - instance.min())

        label = self.datasetid_to_class_id[index]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(
            lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str):
        data = []
        data_path = DATA_PATH + '/first/{}'.format(subset)
        for root, folders, fls in os.walk(data_path):
            if len(fls) == 0:
                continue

            for f in fls:
                if f.endswith("csv"):
                    fp = data_path + "/" + f
                    csv_data = pd.read_csv(fp, engine='python')
                    # print(fp)
                    class_num = csv_data['16'][0]
                    data.append({
                        'subset': subset,
                        'class_name': class_num,
                        'filepath': fp
                    })
        return data


class MultipleEMGDatasetSecond(Dataset):

    def __getitem__(self, index):
        item = self.datasetid_to_filepath[index]
        instance = pd.read_csv(item, engine='python')
        instance = instance.drop('16', axis=1)
        instance = instance.values
        instance = instance.reshape(16, 20, 20)

        # Normalise to 0-1
        instance = (instance - instance.min()) / \
            (instance.max() - instance.min())

        label = self.datasetid_to_class_id[index]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(
            lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str):
        data = []
        data_path = DATA_PATH + '/second/{}'.format(subset)
        for root, folders, fls in os.walk(data_path):
            if len(fls) == 0:
                continue

            for f in fls:
                if f.endswith("csv"):
                    fp = data_path + "/" + f
                    csv_data = pd.read_csv(fp, engine='python')
                    # print(fp)
                    class_num = csv_data['16'][0]
                    data.append({
                        'subset': subset,
                        'class_name': class_num,
                        'filepath': fp
                    })
        return data


class MultipleEMGDatasetThird(Dataset):

    def __getitem__(self, index):
        item = self.datasetid_to_filepath[index]
        instance = pd.read_csv(item, engine='python')
        instance = instance.drop('16', axis=1)
        instance = instance.values
        instance = instance.reshape(16, 20, 20)

        # Normalise to 0-1
        instance = (instance - instance.min()) / \
            (instance.max() - instance.min())

        label = self.datasetid_to_class_id[index]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(
            lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str):
        data = []
        data_path = DATA_PATH + '/third/{}'.format(subset)
        for root, folders, fls in os.walk(data_path):
            if len(fls) == 0:
                continue

            for f in fls:
                if f.endswith("csv"):
                    fp = data_path + "/" + f
                    csv_data = pd.read_csv(fp, engine='python')
                    # print(fp)
                    class_num = csv_data['16'][0]
                    data.append({
                        'subset': subset,
                        'class_name': class_num,
                        'filepath': fp
                    })
        return data


class MultipleEMGDatasetAll(Dataset):

    def __getitem__(self, index):
        item = self.datasetid_to_filepath[index]
        instance = pd.read_csv(item, engine='python')
        instance = instance.drop('24', axis=1)
        instance = instance.values
        instance = instance.reshape(24, 20, 20)

        # Normalise to 0-1
        instance = (instance - instance.min()) / \
            (instance.max() - instance.min())

        label = self.datasetid_to_class_id[index]

        return torch.from_numpy(instance), label

    def __len__(self):
        return len(self.df)

    def __init__(self, subset):
        """Dataset class representing Omniglot dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.df = pd.DataFrame(self.index_subset(subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {
            self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(
            lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset: str):
        data = []
        data_path = DATA_PATH + '/all/{}'.format(subset)
        for root, folders, fls in os.walk(data_path):
            if len(fls) == 0:
                continue

            for f in fls:
                if f.endswith("csv"):
                    fp = data_path + "/" + f
                    csv_data = pd.read_csv(fp, engine='python')
                    # print(fp)
                    class_num = csv_data['24'][0]
                    data.append({
                        'subset': subset,
                        'class_name': class_num,
                        'filepath': fp
                    })
        return data

class MiniImageNet(Dataset):
    def __init__(self, subset):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('background', 'evaluation'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_subset(subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.png')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_{}/'.format(subset)):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            for f in files:
                progress_bar.update(1)
                images.append({
                    'subset': subset,
                    'class_name': class_name,
                    'filepath': os.path.join(root, f)
                })

        progress_bar.close()
        return images


class DummyDataset(Dataset):
    def __init__(self, samples_per_class=10, n_classes=10, n_features=1):
        """Dummy dataset for debugging/testing purposes

        A sample from the DummyDataset has (n_features + 1) features. The first feature is the index of the sample
        in the data and the remaining features are the class index.

        # Arguments
            samples_per_class: Number of samples per class in the dataset
            n_classes: Number of distinct classes in the dataset
            n_features: Number of extra features each sample should have.
        """
        self.samples_per_class = samples_per_class
        self.n_classes = n_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.n_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.n_classes

    def __getitem__(self, item):
        class_id = item % self.n_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)

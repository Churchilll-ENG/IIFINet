import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))
        dataset = split_data(dataset)
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap' or self.data == 'iemo':
            Y = torch.argmax(Y, dim=-1)
        if self.data == 'sad' or self.data == 'sur':
            Y = torch.argmax(Y, dim=-1)
        if self.data == 'ang' or self.data == 'hap':
            Y = torch.argmax(Y, dim=-1)
        if self.data == 'fea' or self.data == 'dis':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META        



class Multimodal_Datasets1(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets1, self).__init__()
        dataset_path = os.path.join(dataset_path, data + '_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        dataset = convert_to_structure_a(dataset)

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3 

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'),
                    self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1)
        return X, Y, META



def convert_to_structure_a(dataset):
    structure_A = {'train': {'audio': [], 'vision': [], 'text': [], 'labels': []},
                   'test': {'audio': [], 'vision': [], 'text': [], 'labels': []},
                   'dev': {'audio': [], 'vision': [], 'text': [], 'labels': []}}
    for split in ['train', 'test', 'dev']:
        A, V, T, L = extract_and_process(dataset, split)
        structure_A[split]['audio'] = A 
        structure_A[split]['vision'] = V 
        structure_A[split]['text'] = T  
        structure_A[split]['labels'] = L  
    structure_A['valid'] = structure_A.pop('dev')
    spilt_a = split_data(structure_A)

    return spilt_a



def extract_and_process(dataset, split='train'):
    audio_data = []
    visual_data = []
    text_data = []
    labels_data = []

    for dialog in dataset[split]:
        audio_features = np.array(dialog['audio'])  
        visual_features = np.array(dialog['visual']) 
        text_features = np.array(dialog['text']) 
        labels = np.array(dialog['labels'])  

        audio_data.append(audio_features)
        visual_data.append(visual_features)
        text_data.append(text_features)
        labels_data.append(labels)

    audio_data = np.vstack(audio_data) 
    visual_data = np.vstack(visual_data)  
    text_data = np.vstack(text_data) 
    labels_data = np.concatenate(labels_data)  

    audio_data = audio_data.reshape(-1, 1, 100) 
    visual_data = visual_data.reshape(-1, 1, 512)
    text_data = text_data.reshape(-1, 1, 768)
    labels_data = labels_data.reshape(-1, 1, 1)

    label_map = {
        0: [[0, 1], [1, 0], [1, 0], [1, 0]],
        1: [[1, 0], [0, 1], [1, 0], [1, 0]],
        2: [[1, 0], [1, 0], [0, 1], [1, 0]],
        3: [[1, 0], [1, 0], [1, 0], [0, 1]]
    }
    updated_labels_data = []
    for label in labels_data:
        label_value = label[0, 0]
        updated_label = label_map.get(label_value)
        if updated_label is not None:
            updated_labels_data.append(np.array(updated_label))
        else:
            updated_labels_data.append(label)
    updated_labels_data = np.array(updated_labels_data)

    return audio_data, visual_data, text_data, updated_labels_data


def split_data(data, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    total_samples = sum([len(data[split]['audio']) for split in ['train', 'test', 'valid']])
    train_samples = int(total_samples * train_ratio)
    test_samples = int(total_samples * test_ratio)
    valid_samples = total_samples - train_samples - test_samples
    combined_data = {key: [] for key in ['audio', 'vision', 'text', 'labels']}
    for split in ['train', 'test', 'valid']:
        for modality in combined_data:
            combined_data[modality].extend(data[split][modality])
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    new_data = {'train': {}, 'test': {}, 'valid': {}}
    for modality in combined_data:
        combined_data[modality] = np.array(combined_data[modality])
        new_data['train'][modality] = combined_data[modality][indices[:train_samples]]
        new_data['test'][modality] = combined_data[modality][indices[train_samples:train_samples + test_samples]]
        new_data['valid'][modality] = combined_data[modality][indices[train_samples + test_samples:]]

    return new_data





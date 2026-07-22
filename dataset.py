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
    
############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################


class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))
        dataset = split_data(dataset)

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        # Note: this is STILL an numpy array
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

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3  # vision/ text/ audio

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

    # 转化为numpy数组，维度为 (样本数, 时间步数, 特征维度)
    for split in ['train', 'test', 'dev']:
        A, V, T, L = extract_and_process(dataset, split)
        structure_A[split]['audio'] = A  # shape: (samples, time_steps, 100)
        structure_A[split]['vision'] = V  # shape: (samples, time_steps, 512)
        structure_A[split]['text'] = T  # shape: (samples, time_steps, 768)
        structure_A[split]['labels'] = L  # shape: (samples, time_steps)
    structure_A['valid'] = structure_A.pop('dev')
    spilt_a = split_data(structure_A)

    return spilt_a



def extract_and_process(dataset, split='train'):
    # 初始化空列表以存储所有模态的结果
    audio_data = []
    visual_data = []
    text_data = []
    labels_data = []

    # 遍历指定的数据集（'train'，'test'，'valid'）
    for dialog in dataset[split]:
        # 提取每个模态的数据
        audio_features = np.array(dialog['audio'])  # shape: (n, 100)
        visual_features = np.array(dialog['visual'])  # shape: (n, 512)
        text_features = np.array(dialog['text'])  # shape: (n, 768)
        labels = np.array(dialog['labels'])  # shape: (n,)

        # 将每个模态的数据展开并添加到列表中
        audio_data.append(audio_features)
        visual_data.append(visual_features)
        text_data.append(text_features)
        labels_data.append(labels)

    # 合并所有样本的数据
    audio_data = np.vstack(audio_data)  # shape: (108 * n, 100)
    visual_data = np.vstack(visual_data)  # shape: (108 * n, 512)
    text_data = np.vstack(text_data)  # shape: (108 * n, 768)
    labels_data = np.concatenate(labels_data)  # shape: (108 * n,)

    # 扩展到与模型兼容的形状 (108 * n, 1, 100)
    audio_data = audio_data.reshape(-1, 1, 100)  # shape: (108 * n, 1, 100)
    visual_data = visual_data.reshape(-1, 1, 512)
    text_data = text_data.reshape(-1, 1, 768)
    labels_data = labels_data.reshape(-1, 1, 1)

    label_map = {
        0: [[0, 1], [1, 0], [1, 0], [1, 0]],
        1: [[1, 0], [0, 1], [1, 0], [1, 0]],
        2: [[1, 0], [1, 0], [0, 1], [1, 0]],
        3: [[1, 0], [1, 0], [1, 0], [0, 1]]
    }

    # 新的 labels_data 数组，用来存放转换后的标签
    updated_labels_data = []

    # 遍历每个标签并替换
    for label in labels_data:
        label_value = label[0, 0]  # 提取标签的值
        # 使用字典查找映射
        updated_label = label_map.get(label_value)
        if updated_label is not None:
            updated_labels_data.append(np.array(updated_label))
        else:
            updated_labels_data.append(label)  # 如果没有找到对应的映射，就保留原标签

    # 将 updated_labels_data 转换回 numpy 数组（如果需要）
    updated_labels_data = np.array(updated_labels_data)

    return audio_data, visual_data, text_data, updated_labels_data


def split_data(data, train_ratio=0.7, test_ratio=0.2, valid_ratio=0.1):
    """
    data = {(合并前数据格式）
    'train': {
        'audio': [样本1_audio, 样本2_audio, ...],
        'vision': [样本1_vision, 样本2_vision, ...],
        'text': [样本1_text, 样本2_text, ...],
        'labels': [样本1_label, 样本2_label, ...]
    },
    'test': {
        'audio': [样本n_audio, 样本n+1_audio, ...],
        'vision': [样本n_vision, 样本n+1_vision, ...],
        'text': [样本n_text, 样本n+1_text, ...],
        'labels': [样本n_label, 样本n+1_label, ...]
    },
    'valid': {
        'audio': [样本x_audio, 样本x+1_audio, ...],
        'vision': [样本x_vision, 样本x+1_vision, ...],
        'text': [样本x_text, 样本x+1_text, ...],
        'labels': [样本x_label, 样本x+1_label, ...]
    }
}
    """
    # 计算新数据集样本数
    total_samples = sum([len(data[split]['audio']) for split in ['train', 'test', 'valid']])
    train_samples = int(total_samples * train_ratio)
    test_samples = int(total_samples * test_ratio)
    valid_samples = total_samples - train_samples - test_samples

    # 将所有样本合并
    combined_data = {key: [] for key in ['audio', 'vision', 'text', 'labels']}
    """
    combined_data = {（合并后数据集格式
    'audio': [样本1_audio, 样本2_audio, ..., 样本n_audio, 样本n+1_audio, ..., 样本x_audio, 样本x+1_audio, ...],
    'vision': [样本1_vision, 样本2_vision, ..., 样本n_vision, 样本n+1_vision, ..., 样本x_vision, 样本x+1_vision, ...],
    'text': [样本1_text, 样本2_text, ..., 样本n_text, 样本n+1_text, ..., 样本x_text, 样本x+1_text, ...],
    'labels': [样本1_label, 样本2_label, ..., 样本n_label, 样本n+1_label, ..., 样本x_label, 样本x+1_label, ...]
    }
    """
    for split in ['train', 'test', 'valid']:
        for modality in combined_data:
            combined_data[modality].extend(data[split][modality])

    # 打乱数据
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # 重新划分数据
    new_data = {'train': {}, 'test': {}, 'valid': {}}
    for modality in combined_data:
        combined_data[modality] = np.array(combined_data[modality])
        new_data['train'][modality] = combined_data[modality][indices[:train_samples]]
        new_data['test'][modality] = combined_data[modality][indices[train_samples:train_samples + test_samples]]
        new_data['valid'][modality] = combined_data[modality][indices[train_samples + test_samples:]]

    return new_data





import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class LazySupervisedDataset(Dataset):
    # ... 原有代码 ...
    
    @staticmethod
    def collate_fn(batch):
        """支持批处理的 collate function"""
        if len(batch) == 1:
            return batch[0]
        
        # 准备批次数据容器
        batch_data = {
            'input_ids': [],
            'text_input_ids': [],
            'whisper_input_feature': [],
            'is_continuous_mask': [],
            'labels': {
                'audio_labels': [],
                'text_labels': [],
                'audio_loss_mask': [],
                'text_loss_mask': []
            }
        }
        
        # 收集所有样本的数据
        max_audio_len = 0
        max_text_len = 0
        
        for sample in batch:
            batch_data['input_ids'].append(sample['input_ids'])
            batch_data['text_input_ids'].append(sample['text_input_ids'])
            batch_data['whisper_input_feature'].extend(sample['whisper_input_feature'])
            batch_data['is_continuous_mask'].append(sample['is_continuous_mask'])
            
            audio_labels, text_labels, audio_loss_mask, text_loss_mask = sample['labels']
            batch_data['labels']['audio_labels'].append(audio_labels)
            batch_data['labels']['text_labels'].append(text_labels)
            batch_data['labels']['audio_loss_mask'].append(audio_loss_mask)
            batch_data['labels']['text_loss_mask'].append(text_loss_mask)
            
            max_audio_len = max(max_audio_len, sample['input_ids'].shape[1])
            max_text_len = max(max_text_len, sample['text_input_ids'].shape[1])
        
        # Pad 所有序列到相同长度
        # 使用 pad_token 进行 padding
        pad_token = batch[0]['labels'][0].new_full((1,), batch[0].get('pad_token', 0))[0]
        
        # Pad input_ids
        padded_input_ids = []
        for ids in batch_data['input_ids']:
            pad_len = max_audio_len - ids.shape[1]
            if pad_len > 0:
                padded = torch.cat([ids, ids.new_full((1, pad_len), pad_token)], dim=1)
            else:
                padded = ids
            padded_input_ids.append(padded)
        
        # Pad text_input_ids
        padded_text_input_ids = []
        for ids in batch_data['text_input_ids']:
            pad_len = max_text_len - ids.shape[1]
            if pad_len > 0:
                padded = torch.cat([ids, ids.new_full((1, pad_len), pad_token)], dim=1)
            else:
                padded = ids
            padded_text_input_ids.append(padded)
        
        # Pad continuous masks
        padded_continuous_mask = []
        for mask in batch_data['is_continuous_mask']:
            pad_len = max_audio_len - mask.shape[1]
            if pad_len > 0:
                padded = torch.cat([mask, mask.new_full((1, pad_len), False)], dim=1)
            else:
                padded = mask
            padded_continuous_mask.append(padded)
        
        # Pad labels and loss masks
        padded_audio_labels = []
        padded_text_labels = []
        padded_audio_loss_mask = []
        padded_text_loss_mask = []
        
        for i in range(len(batch)):
            # Audio labels and mask
            audio_label = batch_data['labels']['audio_labels'][i]
            audio_mask = batch_data['labels']['audio_loss_mask'][i]
            pad_len = max_audio_len - audio_label.shape[1]
            if pad_len > 0:
                padded_audio_labels.append(
                    torch.cat([audio_label, audio_label.new_full((1, pad_len), pad_token)], dim=1)
                )
                padded_audio_loss_mask.append(
                    torch.cat([audio_mask, audio_mask.new_full((1, pad_len), False)], dim=1)
                )
            else:
                padded_audio_labels.append(audio_label)
                padded_audio_loss_mask.append(audio_mask)
            
            # Text labels and mask
            text_label = batch_data['labels']['text_labels'][i]
            text_mask = batch_data['labels']['text_loss_mask'][i]
            pad_len = max_text_len - text_label.shape[1]
            if pad_len > 0:
                padded_text_labels.append(
                    torch.cat([text_label, text_label.new_full((1, pad_len), pad_token)], dim=1)
                )
                padded_text_loss_mask.append(
                    torch.cat([text_mask, text_mask.new_full((1, pad_len), False)], dim=1)
                )
            else:
                padded_text_labels.append(text_label)
                padded_text_loss_mask.append(text_mask)
        
        # Stack 所有批次数据
        batched_data = {
            'input_ids': torch.cat(padded_input_ids, dim=0),
            'text_input_ids': torch.cat(padded_text_input_ids, dim=0),
            'whisper_input_feature': batch_data['whisper_input_feature'],  # 保持为列表
            'is_continuous_mask': torch.cat(padded_continuous_mask, dim=0),
            'labels': (
                torch.cat(padded_audio_labels, dim=0),
                torch.cat(padded_text_labels, dim=0),
                torch.cat(padded_audio_loss_mask, dim=0),
                torch.cat(padded_text_loss_mask, dim=0)
            )
        }
        
        return batched_data

    @staticmethod
    def collate_fn_with_dynamic_padding(batch):
        """使用动态 padding 的更高效实现"""
        if len(batch) == 1:
            return batch[0]
        
        # 使用 pad_sequence 进行动态 padding
        batch_size = len(batch)
        
        # 提取所有序列并移除 batch 维度
        input_ids_list = [sample['input_ids'].squeeze(0) for sample in batch]
        text_input_ids_list = [sample['text_input_ids'].squeeze(0) for sample in batch]
        is_continuous_mask_list = [sample['is_continuous_mask'].squeeze(0) for sample in batch]
        
        # 提取 labels
        audio_labels_list = []
        text_labels_list = []
        audio_loss_mask_list = []
        text_loss_mask_list = []
        
        for sample in batch:
            audio_labels, text_labels, audio_loss_mask, text_loss_mask = sample['labels']
            audio_labels_list.append(audio_labels.squeeze(0))
            text_labels_list.append(text_labels.squeeze(0))
            audio_loss_mask_list.append(audio_loss_mask.squeeze(0))
            text_loss_mask_list.append(text_loss_mask.squeeze(0))
        
        # 使用 pad_sequence 进行 padding
        pad_token = batch[0]['labels'][0].new_full((1,), 0)[0]  # 假设 pad_token 为 0
        
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token)
        padded_text_input_ids = pad_sequence(text_input_ids_list, batch_first=True, padding_value=pad_token)
        padded_continuous_mask = pad_sequence(is_continuous_mask_list, batch_first=True, padding_value=False)
        
        padded_audio_labels = pad_sequence(audio_labels_list, batch_first=True, padding_value=pad_token)
        padded_text_labels = pad_sequence(text_labels_list, batch_first=True, padding_value=pad_token)
        padded_audio_loss_mask = pad_sequence(audio_loss_mask_list, batch_first=True, padding_value=False)
        padded_text_loss_mask = pad_sequence(text_loss_mask_list, batch_first=True, padding_value=False)
        
        # 收集 whisper features
        whisper_features = []
        for sample in batch:
            whisper_features.extend(sample['whisper_input_feature'])
        
        batched_data = {
            'input_ids': padded_input_ids,
            'text_input_ids': padded_text_input_ids,
            'whisper_input_feature': whisper_features,
            'is_continuous_mask': padded_continuous_mask,
            'labels': (
                padded_audio_labels,
                padded_text_labels,
                padded_audio_loss_mask,
                padded_text_loss_mask
            )
        }
        
        return batched_data
import os
import torch
from torch.utils.data import DataLoader
from furiosa_llm_models.generators.packing import greedy_attention_packing_bert
from transformers import BertTokenizer
from torch.nn.functional import pad

def make_packed_calib_data_loader(calib_dataset, bucket_size, pad_token_id):
    def bucket_pad(tensor):
        if bucket_size is None:
            return tensor

        padding_size = bucket_size - tensor.shape[-1]
        return pad(tensor, (0, padding_size))

    data_list = []
    for batch in calib_dataset:
        input_ids, token_type_ids, attention_mask, position_ids, packed_target_locations = (
            greedy_attention_packing_bert(
                input_ids=bucket_pad(batch["input_ids"]),
                token_type_ids=bucket_pad(batch["token_type_ids"]),
                bucketized_attention_mask=bucket_pad(batch["attention_mask"]),
                pad_token_id=pad_token_id,
                compact_mask=False,
            )
        )

        model_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data_list.append(model_inputs)

    return DataLoader(data_list)



def make_dataloader(qsl, batch_size, n_calib):

    file_path = os.path.join(
        os.path.realpath(__file__)[0:os.path.realpath(__file__).find('language')], 
        'calibration', 
        'SQuAD-v1.1',
        'bert_calibration_features.txt',)
    with open(file_path, 'r') as fp:
        lines = fp.readlines()

    calib_data_indice_list = []
    for line in lines:
        numbers = [int(num) for num in line.split('\n') if num.isdigit()]
        calib_data_indice_list.extend(numbers)
    
    calib_eval_features = [qsl.eval_features[i] for i in calib_data_indice_list]

    data_list = []
    if n_calib != -1:
        calib_eval_features = calib_eval_features[0:n_calib]
    for feature in calib_eval_features:
        data_list.append({
            'input_ids': torch.LongTensor(feature.input_ids),
            'attention_mask': torch.LongTensor(feature.input_mask),
            'token_type_ids': torch.LongTensor(feature.segment_ids),
        })
    
    dataloader = DataLoader(data_list, batch_size=batch_size)

    return dataloader

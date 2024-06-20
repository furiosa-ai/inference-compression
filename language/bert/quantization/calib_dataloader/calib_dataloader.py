import os

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def make_packed_calib_data_loader(
    calib_eval_features , batch_size, n_calib, pad_token_id=0, bucket_size=384, compact_mask=False
):
    def bucket_pad(tensor):
        if bucket_size is None:
            return tensor

        padding_size = bucket_size - tensor.shape[-1]
        return pad(tensor, (0, padding_size))

    from furiosa_llm_models.generators.packing import \
        greedy_attention_packing_bert

    data_list = []
    for feature in calib_eval_features:
        (
            input_ids,
            token_type_ids,
            attention_mask,
            position_ids,
            packed_target_locations,
        ) = greedy_attention_packing_bert(
            input_ids=torch.LongTensor(feature.input_ids).unsqueeze(0),
            token_type_ids=torch.LongTensor(feature.segment_ids).unsqueeze(0),
            bucketized_attention_mask=torch.LongTensor(feature.input_mask).unsqueeze(0),
            pad_token_id=pad_token_id,
            compact_mask=compact_mask,
        )

        model_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data_list.append(model_inputs)

    return DataLoader(data_list, batch_size=batch_size)


def make_dataloader(
    calib_eval_features , batch_size, n_calib, include_position_ids=False, compact_mask=False
):
    if compact_mask:
        raise NotImplementedError(
            "We have not yet implemented support for the compact_mask feature."
        )

    data_list = []
    for feature in calib_eval_features:
        data = {
            "input_ids": torch.LongTensor(feature.input_ids),
            "attention_mask": torch.LongTensor(feature.input_mask),
            "token_type_ids": torch.LongTensor(feature.segment_ids),
        }

        if include_position_ids:
            data.update(
                {
                    "attention_mask": data["attention_mask"]
                    .unsqueeze(0)
                    .repeat(384, 1),
                    "position_ids": torch.arange(384),
                }
            )

        data_list.append(data)

    dataloader = DataLoader(data_list, batch_size=batch_size)

    return dataloader


def load_bert_calibration_data(qsl, n_calib):
    file_path = os.path.join(
        os.path.realpath(__file__)[0 : os.path.realpath(__file__).find("language")],
        "calibration",
        "SQuAD-v1.1",
        "bert_calibration_features.txt",
    )

    with open(file_path, "r") as fp:
        lines = fp.readlines()

    calib_data_indice_list = []
    for line in lines:
        numbers = [int(num) for num in line.split("\n") if num.isdigit()]
        calib_data_indice_list.extend(numbers)

    calib_eval_features = [qsl.eval_features[i] for i in calib_data_indice_list]

    if n_calib != -1:
        calib_eval_features = calib_eval_features[0:n_calib]

    return calib_eval_features

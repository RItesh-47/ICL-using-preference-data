import torch
import logging
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.utils.tokenizer_util import get_tokenizer

logger = logging.getLogger(__name__)


def _encode_field(example, idx, **kwargs):
    field_getter = kwargs['field_getter'] 
    field_getter_pd = kwargs['field_getter_pd'] 

    tokenizer = kwargs['tokenizer'] 
    truncation = kwargs['truncation']

    text = field_getter(example)
    text_pd = field_getter_pd(example) 
    tokenized_inputs = tokenizer.encode_plus(text, truncation=truncation, return_tensors='pt')
    tokenized_inputs_pd = tokenizer.encode_plus(text_pd, truncation=truncation, return_tensors='pt')
    return {
        'input_ids': tokenized_inputs.input_ids[0],
        'attention_mask': tokenized_inputs.attention_mask[0],
        "metadata": {"id": idx, "len": len(tokenized_inputs.input_ids[0]), "len_pd":len(tokenized_inputs_pd.input_ids[0]),
                     "text": text, "text_pd":text_pd}
    }


def encode_field(tokenizer, dataset_wrapper, field, field_pd, truncation): 
    remove_columns = [col for col in dataset_wrapper.dataset.column_names]
    encoded_dataset = dataset_wrapper.dataset.map(
        _encode_field,
        load_from_cache_file=False,
        with_indices=True,
        remove_columns=remove_columns,
        fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field],
                    'field_getter_pd': dataset_wrapper.field_getter.functions[field_pd],
                   'tokenizer': tokenizer, 'truncation': truncation}
    )
    return encoded_dataset


class BaseDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, model_name, field, field_pd, is_mixed, len_qa, len_pd, dataset_path=None, dataset_split=None, ds_size=None) -> None: 
        self.tokenizer = get_tokenizer(model_name)
        self.init_dataset(task_name, field, field_pd, dataset_path, dataset_split, ds_size)
 
    def init_dataset(self, task_name, field, field_pd, dataset_path, dataset_split, ds_size, truncation=True):
        self.dataset_wrapper = get_dataset_wrapper(task_name,
                                                   dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, field_pd, truncation)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)

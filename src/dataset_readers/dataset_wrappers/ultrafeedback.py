from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging
 
logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["prompt"]+" "+entry["chosen"][1]["content"]+" "+entry["rejected"][1]["content"]

# incontext example format for experiment 2(where we give examples in form of preference data)
# @field_getter.add("qa")
# def get_qa(entry):
#     return "{prompt} : Is \"{sentence1}\" a better answer than \"{sentence2}\"? Yes".format(
#             prompt=entry["prompt"],
#             sentence1=entry["chosen"][1]["content"],
#             sentence2=entry["rejected"][1]["content"]
#             )

@field_getter.add("qa")
def get_qa(entry):
    return "{prompt} : \"{sentence1}\"".format(
            prompt=entry["prompt"],
            sentence1=entry["chosen"][1]["content"]
            )

# for preference datasamples
# @field_getter.add("qa1a2")
# def get_qa1a2(entry):
#     return "{prompt} : For this prompt, \"{sentence1}\" is a better answer than \"{sentence2}\ ".format(
#             prompt=entry["prompt"],
#             sentence1=entry["chosen"][1]["content"],
#             sentence2=entry["rejected"][1]["content"]
#             )

@field_getter.add("qa1a2")
def get_qa1a2(entry):
    return "{prompt} : Response 1 :  \"{sentence1}\" . Response 2 : \"{sentence2}\" . Response 1 is better than Response 2".format(
            prompt=entry["prompt"],
            sentence1=entry["chosen"][1]["content"],
            sentence2=entry["rejected"][1]["content"]
            )


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[entry['label']]


# @field_getter.add("gen_a")
# def get_gen_a(entry):
#     # hypothesis, premise = get_q(entry)
#     return "{ice_prompt} {prompt} : Is \"{sentence1}\" a better answer than \"{sentence2}\"? ".format(
#             prompt=entry["prompt"],
#             ice_prompt="{ice_prompt}",
#             sentence1=entry["chosen"][1]["content"],
#             sentence2=entry["rejected"][1]["content"]
#             )

@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}. Give a good response for this task : \"{prompt}\" ".format(
            prompt=entry["prompt"],
            ice_prompt="{ice_prompt}"
            )


# @field_getter.add("choices")
# def get_choices(entry):
#     return ["No", "Yes"]

@field_getter.add("chosen")
def get_choices(entry):
    return entry["chosen"][1]["content"]

class DatasetWrapper(ABC): 
    name = "ultrafeedback"
    ice_separator = "\n"
    question_field = "prompt"
    answer_field = ["chosen", "rejected"]
    hf_dataset = "HuggingFaceH4/ultrafeedback_binarized"
    hf_dataset_name = "ultrafeedback"
    field_getter = field_getter
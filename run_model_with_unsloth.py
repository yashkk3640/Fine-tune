from unsloth import FastLanguageModel
import torch

MODEL_PATH = "./trained_model"  # Update path
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

TOKEN = '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'

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    # max_seq_length = max_seq_length,
    # dtype = dtype,
    # load_in_4bit = load_in_4bit,
    device_map = "cuda:0",
    token = TOKEN, # use one if using gated models like meta-llama/Llama-2-7b-hf
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN


FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# inputs = tokenizer(
# [
#     alpaca_prompt.format(
#         "Generate a geo chart that has lg as longitude and lt as latitude which is corresponding to Storage Unit as value. Consist tooltip background color yellow.", # instruction
#         "{ \"dataSources\": [ { \"dataSourceId\": \"2f344f20-663e-4428-8e9a-bee6797fc0f5\", \"queryKey\": \"API:+lookup$$lookup$$usa_population\", \"cols\": [ { \"jsonName\": \"lg\", \"type\": \"number\" }, { \"jsonName\": \"lt\", \"type\": \"number\" }, { \"jsonName\": \"Storage_Unit\", \"type\": \"number\" }], \"params\": [] } ]}", # input
#         "", # output - leave this blank for generation!
#     )
# ], return_tensors = "pt").to("cuda:0")

# outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)

# print(tokenizer.batch_decode(outputs))

flag = True

while(flag):
    # question = input("Please enter a string: ")
    # print("generate_response \n\n", generate_response(context, question))
    # Example usage
    instruction = input("Please enter instruction (question/chart info) string: ")
    input_text = input("Please enter input_text (datasource) string: ")

    inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction,
            input_text,
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda:0")

    outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)

    # print(tokenizer.batch_decode(outputs))
    print(tokenizer.batch_decode(outputs))
    print("\n\n")
    newFlag = input("If not want to continue /exit: ")
    if(newFlag == "/exit"):
        flag = False

print("byee")
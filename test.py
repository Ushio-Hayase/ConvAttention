import torch
from transformers import AutoTokenizer
from original_transformer_model.model import Transformer


tokenizer = AutoTokenizer.from_pretrained("quantumaikr/KoreanLM")
BATCH_SIZE = 128
LR = 1e-5
D_MODEL = 512
DFF = 1024
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_LEN = 256
VOCAB_SIZE = tokenizer.vocab_size
DROP_OUT = 0.1
PAD_IDX = tokenizer.pad_token_id
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = Transformer(D_MODEL, DFF, NUM_HEADS, NUM_LAYERS, MAX_LEN, VOCAB_SIZE, DROP_OUT, PAD_IDX, DEVICE).to(device=DEVICE)

model.load_state_dict(torch.load("model_convattention.pth", map_location=DEVICE))
model.eval()


def infer(sentence: str):
    tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

    bos = torch.asarray([[tokenizer.bos_token_id]], device=DEVICE)
    eos = torch.asarray([[tokenizer.eos_token_id]], device=DEVICE)

    for _ in range(MAX_LEN):
        predict = model(tokenized_sentence, bos)

        predict = torch.softmax(predict[:, -1:, :], dim=-1)

        predict = torch.argmax(predict, dim=-1)

        if predict == eos:
            break

        bos = torch.concat([bos, predict], dim=1)


    return bos

if __name__ == "__main__":
    print(tokenizer.decode(infer("민기 심심해").squeeze()))
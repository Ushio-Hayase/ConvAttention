import torch
import tqdm
import numpy as np
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast, AutoTokenizer

from dataset import Dataset
from original_transformer_model.model import Transformer


tokenizer = AutoTokenizer.from_pretrained("quantumaikr/KoreanLM")

EPOCH = 20
BATCH_SIZE = 128
LR = 1e-5
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_LEN = 128
VOCAB_SIZE = tokenizer.vocab_size
DROP_OUT = 0.1
PAD_IDX = tokenizer.pad_token_id
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def acc_func(true: torch.Tensor, pred: torch.Tensor) -> float:
    pred = torch.argmax(pred, dim=-1)
    
    return torch.eq(true, pred).float().mean().item()

def main():
    writer = SummaryWriter()

    data = np.load("data.npy", allow_pickle=True)

    train_len = int(len(data) * 0.7)
    valid_len = int(len(data) * 0.2)
    test_len = len(data) - (train_len + valid_len)


    train_dataset = Dataset(data[:train_len, 0], data[:train_len, 1])
    vaild_dataset = Dataset(data[train_len:train_len+valid_len, 0], data[train_len:train_len+valid_len, 1])
    test_dataset = Dataset(data[-test_len:, 0], data[-test_len:, 1])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, True)
    valid_dataloader = torch.utils.data.DataLoader(vaild_dataset, BATCH_SIZE, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, False)


    model = Transformer(D_MODEL, DFF, NUM_HEADS, NUM_LAYERS, MAX_LEN, VOCAB_SIZE, DROP_OUT, PAD_IDX, DEVICE).to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(EPOCH):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                x = tokenizer(x, padding="max_length", add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)
                
                

                true_y = []

                for k in y:
                    true_y.append(tokenizer(k + tokenizer.eos_token, padding="max_length",add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE))
                
                true_y = torch.stack(true_y)

                dec_in = tokenizer(y, padding="max_length", truncation=True,  max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)

                optimizer.zero_grad()

                pred_y = model(x, dec_in)
                loss = criterion(pred_y.transpose(1, 2), true_y)
                loss.backward()
                optimizer.step()

                acc_value = acc_func(true_y, pred_y)

                tepoch.set_postfix(loss=loss.item(), acc=acc_value)
                writer.add_scalar("loss/train", loss, epoch * len(train_dataloader) + i)
                writer.add_scalar("acc/train", acc_value, epoch * len(train_dataloader) + i)

        with torch.no_grad():
            with tqdm.tqdm(valid_dataloader, unit="batch") as tepoch:
                for i, (x, y) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch + 1} Valid")
                    x = tokenizer(x, padding="max_length", add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)
                    
                    

                    true_y = []

                    for k in y:
                        true_y.append(tokenizer(k + tokenizer.eos_token, padding="max_length",add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE))
                    
                    true_y = torch.stack(true_y)

                    dec_in = tokenizer(y, padding="max_length", truncation=True,  max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)

                    pred_y = model(x, dec_in)
                    loss = criterion(pred_y.transpose(1, 2), true_y)
                    acc_value = acc_func(true_y, pred_y)

                    tepoch.set_postfix(loss=loss.item(), acc=acc_value)
                    writer.add_scalar("loss/valid", loss, epoch * len(train_dataloader) + i)
                    writer.add_scalar("acc/valid", acc_value, epoch * len(train_dataloader) + i)

    with torch.no_grad():
            with tqdm.tqdm(test_dataloader, unit="batch") as tepoch:
                for i, (x, y) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch + 1} Test")
                    x = tokenizer(x, padding="max_length", add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)
                
                

                    true_y = []

                    for k in y:
                        true_y.append(tokenizer(k + tokenizer.eos_token, padding="max_length",add_special_tokens=False, max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE))
                    
                    true_y = torch.stack(true_y)

                    dec_in = tokenizer(y, padding="max_length", truncation=True,  max_length=MAX_LEN, return_tensors="pt")['input_ids'].squeeze().to(device=DEVICE)

                    pred_y = model(x, dec_in)
                    loss = criterion(pred_y.transpose(1, 2), true_y)
                    acc_value = acc_func(true_y, pred_y)

                    tepoch.set_postfix(loss=loss.item(), acc=acc_value)
                    writer.add_scalar("loss/test", loss, i)
                    writer.add_scalar("acc/test", acc_value,i)
    
    writer.close()
    torch.save(model.state_dict(), "model.pth")


            

            


if __name__ == "__main__":

    
    main()

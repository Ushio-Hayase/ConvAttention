import torch
import tqdm
import numpy as np
import torch.utils
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizerFast

from dataset import Dataset
from convattention_transformer_model.model import Transformer


tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

EPOCH = 5
BATCH_SIZE = 32
LR = 1e-5
D_MODEL = 512
DFF = 1024
NUM_HEADS = 4
NUM_LAYERS = 4
MAX_LEN = 128
VOCAB_SIZE = 42000
DROP_OUT = 0.1
PAD_IDX = tokenizer.pad_token_id
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def acc_func(true: torch.Tensor, pred: torch.Tensor) -> float:
    pred = torch.argmax(pred, dim=-1)
    
    return torch.eq(true, pred).to(torch.float).mean().item()

def main():
    writer = SummaryWriter()

    data = np.load("data.npy", allow_pickle=True)

    train_len = int(len(data) * 0.7)
    valid_len = int(len(data) * 0.2)
    test_len = len(data) - (train_len + valid_len)


    train_dataset = Dataset(data[:train_len, 0], data[:train_len, 1])
    vaild_dataset = Dataset(data[train_len:valid_len, 0], data[train_len:valid_len, 1])
    test_dataset = Dataset(data[valid_len:test_len, 0], data[valid_len:test_len, 1])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, True)
    valid_dataloader = torch.utils.data.DataLoader(vaild_dataset, BATCH_SIZE, False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, False)


    model = Transformer(D_MODEL, DFF, NUM_HEADS, NUM_LAYERS, MAX_LEN, VOCAB_SIZE, DROP_OUT, PAD_IDX, DEVICE).to(device=DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    for epoch in range(EPOCH):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for i, (x, y) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch + 1}")
                x = tokenizer(x, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].to(device=DEVICE)
                y = tokenizer(y, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")['input_ids'].to(device=DEVICE)

                pad_tensor = torch.asarray([[tokenizer.pad_token_id]]).to(device=DEVICE)
                pad_tensor = pad_tensor.repeat([BATCH_SIZE, 1])
                true_y = y[:, 1:]
                true_y = torch.concat([true_y, pad_tensor], dim=1)

                discriminator = y!=tokenizer.sep_token_id
                dec_in = y[discriminator]
                dec_in = dec_in.view(discriminator.size(0), discriminator.size(1)-1)
                dec_in = torch.concat([dec_in, pad_tensor], dim=1)

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
                    x = tokenizer(x, padding=True, truncation=True)['input_ids']
                    y = tokenizer(y, padding=True, truncation=True)['input_ids']


                    pad_tensor = torch.asarray([[tokenizer.pad_token_id]]).to(device=DEVICE)
                    pad_tensor = pad_tensor.repeat([BATCH_SIZE, 1])
                    true_y = y[:, 1:]
                    true_y = torch.concat([true_y, pad_tensor], dim=1)

                    discriminator = y!=tokenizer.sep_token_id
                    dec_in = y[discriminator]
                    dec_in = dec_in.view(discriminator.size(0), discriminator.size(1)-1)
                    dec_in = torch.concat([dec_in, pad_tensor], dim=1)

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
                    x = tokenizer(x, padding=True, truncation=True)['input_ids']
                    y = tokenizer(y, padding=True, truncation=True)['input_ids']


                    pad_tensor = torch.asarray([[tokenizer.pad_token_id]]).to(device=DEVICE)
                    pad_tensor = pad_tensor.repeat([BATCH_SIZE, 1])
                    true_y = y[:, 1:]
                    true_y = torch.concat([true_y, pad_tensor], dim=1)

                    discriminator = y!=tokenizer.sep_token_id
                    dec_in = y[discriminator]
                    dec_in = dec_in.view(discriminator.size(0), discriminator.size(1)-1)
                    dec_in = torch.concat([dec_in, pad_tensor], dim=1)
                    
                    pred_y = model(x, dec_in)
                    loss = criterion(pred_y.transpose(1, 2), true_y)
                    acc_value = acc_func(true_y, pred_y)

                    tepoch.set_postfix(loss=loss.item(), acc=acc_value)
                    writer.add_scalar("loss/test", loss, i)
                    writer.add_scalar("acc/test", acc_value,i)
    
    writer.close()


            

            


if __name__ == "__main__":  
    main()

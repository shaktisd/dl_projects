import torch
import working_attention_example_v5 as w
from working_attention_example_v5 import Model




def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = 20
    vocab_size = 10
    batch_size = 1000
    embedding_dim = 512
    # eval loop

    task = w.Task(max_len=max_len, vocab_size=vocab_size, batch_size=batch_size)
    x, y = task.next_batch()
    x = x.to(device)
    y = y.to(device)

    model = w.Model(max_len=max_len, vocab_size=vocab_size, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load('./model.pth'))
    model.to(device)

    with torch.inference_mode():
        y_pred = model(x)
        print(y * max_len)
        print(torch.round(y_pred * max_len))




if __name__ == '__main__':
    main()

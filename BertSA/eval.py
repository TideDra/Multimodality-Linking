from torch import device, load, tensor, reshape, no_grad, cuda, softmax, argmax
from transformers import BertTokenizer

model = load("models/model_0.pth")


def predict(sentence, limitsize=126):
    dev = device('cuda' if cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    s_token = tokenizer.encode(sentence)
    if len(s_token) < limitsize:
        s_token.extend([0] * (limitsize + 2 - len(s_token)))
    mask = [float(k != 0) for k in s_token]
    s_token = tensor(s_token)
    mask = tensor(mask)
    s_token = reshape(s_token, [1, len(s_token)])
    mask = reshape(mask, [1, len(mask)])
    s_token = s_token.long().to(dev)
    mask = mask.long().to(dev)
    model.eval()
    with no_grad():
        output = model(s_token, token_type_ids=None, attention_mask=mask)
        logits = output['logits']
        answer = argmax(softmax(logits, dim=1)).item()
        return answer


if __name__ == '__main__':
    print(predict('你是笨蛋'))



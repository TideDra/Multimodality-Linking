from model import MertForNERwithESD, MertForNER, FlavaEncoder, BertEncoder

if __name__ == '__main__':
    encoder = FlavaEncoder
    ESD_encoder = BertEncoder
    model = MertForNERwithESD(encoder=encoder, ESD_encoder=ESD_encoder)

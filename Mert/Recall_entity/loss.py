import torch

def CosSim(x,y):
    return torch.cosine_similarity(x,y,dim=1)

def TripletLoss(anchor,positive,negative):
    triplet_loss=torch.nn.TripletMarginWithDistanceLoss(distance_function=CosSim,margin=0.5)
    return triplet_loss(anchor,positive,negative)

import json
import logging
import os
import sys
import torch
from torch import Tensor
from transformers import BertModel
from .config import config
from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Datasets.EntityLink_dataset import abs_dict_to_str
import random
def train(multi_model, entity_model, criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    entity_model.eval()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, mention_token_pos, positive_inputs, negative_inputs) in tbar:
            text_embedding = multi_model(**multi_input).text_embeddings

            positive_embedding:Tensor = entity_model(**positive_inputs).last_hidden_state[:,0]
            negative_embedding = []
            anchor_embedding = []
            for idx, candidates in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                candidates_embeddings: Tensor = entity_model(**candidates).last_hidden_state[:,0]
                entity_s, entity_e = mention_token_pos[idx]
                if entity_e>entity_s:
                    entity_embedding = torch.mean(text_embedding[idx][entity_s:entity_e], dim=0)
                else:
                    entity_embedding=text_embedding[idx][entity_s]
                entity_embedding_repeats = entity_embedding.repeat(len(candidates_embeddings), 1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       candidates_embeddings)
                nearest_candidate_embedding = candidates_embeddings[similarities.argmax()]
                negative_embedding.append(nearest_candidate_embedding)
                anchor_embedding.append(entity_embedding)

            negative_embedding = torch.stack(negative_embedding)
            anchor_embedding = torch.stack(anchor_embedding)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss


def evaluate(multi_model, entity_model, dataloader, accelerator):
    multi_model.eval()
    entity_model.eval()
    correct_num=0
    total_num=0
    with torch.no_grad():
        for (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            multi_input=multi_input.to(accelerator.device)
            positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = entity_model(**positive_inputs).last_hidden_state[:,0]
            for idx, candidates in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                candidates=candidates.to(accelerator.device)
                candidates_embeddings: Tensor = entity_model(**candidates).last_hidden_state[:,0]
                entity_s, entity_e = mention_token_pos[idx]
                if entity_e>entity_s:
                    entity_embedding = torch.mean(text_embedding[idx][entity_s:entity_e], dim=0)
                else:
                    entity_embedding=text_embedding[idx][entity_s]
                search_results_embeddings=torch.cat([positive_embedding[idx].unsqueeze(0),candidates_embeddings],dim=0)
                entity_embedding_repeats=entity_embedding.repeat(len(search_results_embeddings),1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       search_results_embeddings)
                answer=similarities.argmax()
                correct_num+= (answer==0)
                total_num+=1

    acc=correct_num/total_num
    accelerator.print(f'accuracy:{acc}')
    return  acc

def trainV2(multi_model, entity_model:BertModel, criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    entity_model.eval()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, mention_token_pos, positive_inputs, negative_inputs) in tbar:
            text_embedding = multi_model(**multi_input).text_embeddings

            positive_embedding:Tensor = entity_model(**positive_inputs).pooler_output
            negative_embedding = []
            anchor_embedding = []
            for idx, _ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                entity_embedding_repeats = entity_embedding.repeat(len(positive_embedding), 1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       positive_embedding)
                top2_neighbour=similarities.topk(2)[1].tolist()
                nearest_candidate_idx=top2_neighbour[0] if top2_neighbour[0]!=idx else top2_neighbour[1]
                nearest_candidate_embedding = positive_embedding[nearest_candidate_idx]
                negative_embedding.append(nearest_candidate_embedding)
                anchor_embedding.append(entity_embedding)

            negative_embedding = torch.stack(negative_embedding)
            anchor_embedding = torch.stack(anchor_embedding)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss

def evaluateV2(multi_model, entity_model,entity_processor, dataloader, accelerator,kg_path):
    multi_model.eval()
    entity_model.eval()
    top1_num=0
    top5_num=0
    top10_num=0
    top20_num=0
    total_num=0
    
    with torch.no_grad():
        for (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            multi_input=multi_input.to(accelerator.device)
            positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = entity_model(**positive_inputs).pooler_output
            kg_emb=get_kg_embeddings(kg_path=kg_path,processor=entity_processor,model=entity_model)
            for idx,_ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                search_results_embeddings=torch.cat([positive_embedding[idx].unsqueeze(0),kg_emb],dim=0)
                entity_embedding_repeats=entity_embedding.repeat(len(search_results_embeddings),1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       search_results_embeddings)
                top1=similarities.topk(1,sorted=False)[1]
                top5=similarities.topk(5,sorted=False)[1]
                top10=similarities.topk(10,sorted=False)[1]
                top20=similarities.topk(20,sorted=False)[1]
                top1_num += (0 in top1)
                top5_num += (0 in top5)
                top10_num += (0 in top10)
                top20_num += (0 in top20)
                total_num+=1

    top1_acc=top1_num/total_num
    top5_acc=top5_num/total_num
    top10_acc=top10_num/total_num
    top20_acc=top20_num/total_num
    accelerator.print(f'Top1:{top1_acc},Top5:{top5_acc},Top10:{top10_acc},Top20:{top20_acc}')
    return  [top1_acc,top5_acc,top10_acc,top20_acc]

def trainV3(multi_model, entity_model:BertModel,LinkLayer,criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    entity_model.eval()
    LinkLayer.train()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, mention_token_pos, positive_inputs, negative_inputs) in tbar:
            text_embedding = multi_model(**multi_input).text_embeddings

            positive_embedding:Tensor = entity_model(**positive_inputs).pooler_output
            negative_embedding = []
            anchor_embedding = []
            for idx, _ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                entity_embedding_repeats = entity_embedding.repeat(len(positive_embedding), 1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       positive_embedding)
                top2_neighbour=similarities.topk(2)[1].tolist()
                nearest_candidate_idx=top2_neighbour[0] if top2_neighbour[0]!=idx else top2_neighbour[1]
                nearest_candidate_embedding = positive_embedding[nearest_candidate_idx]
                negative_embedding.append(nearest_candidate_embedding)
                anchor_embedding.append(entity_embedding)

            negative_embedding = torch.stack(negative_embedding)
            anchor_embedding = torch.stack(anchor_embedding)
            positive_pair=torch.cat([anchor_embedding,positive_embedding],dim=1)
            negative_pair=torch.cat([anchor_embedding,negative_embedding],dim=1)
            combine_input=torch.cat([positive_pair,negative_pair],dim=0)
            labels=torch.cat([torch.zeros(len(positive_pair),device=combine_input.device,dtype=int),torch.ones(len(negative_pair),device=combine_input.device,dtype=int)])
            logits=LinkLayer(combine_input)
            loss = criterion(logits,labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss

def evaluateV3(multi_model, entity_model,LinkLayer,entity_processor, dataloader, accelerator,kg_path):
    multi_model.eval()
    entity_model.eval()
    LinkLayer.eval()
    top1_num=0
    top5_num=0
    top10_num=0
    top20_num=0
    total_num=0
    
    with torch.no_grad():
        for (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            #multi_input=multi_input.to(accelerator.device)
            #positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = entity_model(**positive_inputs).pooler_output
            kg_emb=get_kg_embeddings(kg_path=kg_path,processor=entity_processor,model=entity_model)
            for idx,_ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                search_results_embeddings=torch.cat([positive_embedding[idx].unsqueeze(0),kg_emb],dim=0)
                entity_embedding_repeats=entity_embedding.repeat(len(search_results_embeddings),1)
                combine_input=torch.cat([entity_embedding_repeats,search_results_embeddings],dim=1)
                scores=LinkLayer(combine_input)
                logits=torch.nn.functional.softmax(scores,dim=1)
                similarities=logits[:,0]
                top1=similarities.topk(1,sorted=False)[1]
                top5=similarities.topk(5,sorted=False)[1]
                top10=similarities.topk(10,sorted=False)[1]
                top20=similarities.topk(20,sorted=False)[1]
                top1_num += (0 in top1)
                top5_num += (0 in top5)
                top10_num += (0 in top10)
                top20_num += (0 in top20)
                total_num+=1

    top1_acc=top1_num/total_num
    top5_acc=top5_num/total_num
    top10_acc=top10_num/total_num
    top20_acc=top20_num/total_num
    accelerator.print(f'Top1:{top1_acc},Top5:{top5_acc},Top10:{top10_acc},Top20:{top20_acc}')
    return  [top1_acc,top5_acc,top10_acc,top20_acc]

def trainV4(multi_model, entity_model:BertModel, criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    entity_model.eval()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, mention_token_pos, positive_inputs, negative_inputs) in tbar:
            text_embedding = multi_model(**multi_input).text_embeddings

            positive_embedding:Tensor = entity_model(**positive_inputs).last_hidden_state[:,0]
            negative_embedding = []
            anchor_embedding = []
            for idx, _ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                while True:
                    nearest_candidate_idx=random.randint(0,len(negative_inputs)-1)
                    if nearest_candidate_idx!=idx:
                        break
                nearest_candidate_embedding = positive_embedding[nearest_candidate_idx]
                negative_embedding.append(nearest_candidate_embedding)
                anchor_embedding.append(entity_embedding)

            negative_embedding = torch.stack(negative_embedding)
            anchor_embedding = torch.stack(anchor_embedding)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss

def evaluateV4(multi_model, entity_model,entity_processor, dataloader, accelerator,kg_path):
    multi_model.eval()
    entity_model.eval()

    top1_num=0
    top5_num=0
    top10_num=0
    top20_num=0
    total_num=0
    
    with torch.no_grad():
        for (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            #multi_input=multi_input.to(accelerator.device)
            #positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = entity_model(**positive_inputs).last_hidden_state[:,0]
            
            for idx,_ in enumerate(negative_inputs):
                kg_emb=get_kg_embeddings(kg_path=kg_path,processor=entity_processor,model=entity_model)
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                search_results_embeddings=torch.cat([positive_embedding[idx].unsqueeze(0),kg_emb],dim=0)
                entity_embedding_repeats=entity_embedding.repeat(len(search_results_embeddings),1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       search_results_embeddings)
                top1=similarities.topk(1,sorted=False)[1]
                top5=similarities.topk(5,sorted=False)[1]
                top10=similarities.topk(10,sorted=False)[1]
                top20=similarities.topk(20,sorted=False)[1]
                top1_num += (0 in top1)
                top5_num += (0 in top5)
                top10_num += (0 in top10)
                top20_num += (0 in top20)
                total_num+=1

    top1_acc=top1_num/total_num
    top5_acc=top5_num/total_num
    top10_acc=top10_num/total_num
    top20_acc=top20_num/total_num
    accelerator.print(f'Top1:{top1_acc},Top5:{top5_acc},Top10:{top10_acc},Top20:{top20_acc}')
    return  [top1_acc,top5_acc,top10_acc,top20_acc]

def trainV5(multi_model, criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, mention_token_pos, positive_inputs, negative_inputs) in tbar:
            text_embedding = multi_model(**multi_input).text_embeddings

            positive_embedding:Tensor = multi_model(**positive_inputs).text_embeddings[:,0]
            negative_embedding = []
            anchor_embedding = []
            for idx, _ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                while True:
                    nearest_candidate_idx=random.randint(0,len(negative_inputs)-1)
                    if nearest_candidate_idx!=idx:
                        break
                nearest_candidate_embedding = positive_embedding[nearest_candidate_idx]
                negative_embedding.append(nearest_candidate_embedding)
                anchor_embedding.append(entity_embedding)

            negative_embedding = torch.stack(negative_embedding)
            anchor_embedding = torch.stack(anchor_embedding)

            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss

def evaluateV5(multi_model, dataloader, accelerator):
    multi_model.eval()

    top1_num=0
    top5_num=0
    top10_num=0
    top20_num=0
    total_num=0
    
    with torch.no_grad():
        for (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            #multi_input=multi_input.to(accelerator.device)
            #positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = multi_model(**positive_inputs).text_embeddings[:,0]
            
            for idx,_ in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                entity_embedding = text_embedding[idx][0]
                entity_embedding_repeats=entity_embedding.repeat(len(positive_embedding),1)
                similarities = torch.cosine_similarity(entity_embedding_repeats,
                                                       positive_embedding)
                top1=similarities.topk(1,sorted=False)[1]
                top5=similarities.topk(5,sorted=False)[1]
                top10=similarities.topk(10,sorted=False)[1]
                top20=similarities.topk(20,sorted=False)[1]
                top1_num += (idx in top1)
                top5_num += (idx in top5)
                top10_num += (idx in top10)
                top20_num += (idx in top20)
                total_num+=1

    top1_acc=top1_num/total_num
    top5_acc=top5_num/total_num
    top10_acc=top10_num/total_num
    top20_acc=top20_num/total_num
    accelerator.print(f'Top1:{top1_acc},Top5:{top5_acc},Top10:{top10_acc},Top20:{top20_acc}')
    return  [top1_acc,top5_acc,top10_acc,top20_acc]

def trainV6(multi_model,criterion, dataloader, optimizer, lr_scheduler, epoch, writer,
          accelerator):
    multi_model.train()
    total_loss = 0.0
    with tqdm(enumerate(dataloader, start=1),
              unit='batch',
              total=len(dataloader),
              desc='epoch:{}/{}'.format(epoch, config.epochs),
              disable=not accelerator.is_local_main_process) as tbar:
        for batch_num, (multi_input, labels) in tbar:
            logits = multi_model(**multi_input)
            loss = criterion(logits,labels)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            loss = accelerator.gather(loss)
            total_loss += loss.item()
            if accelerator.is_main_process:
                writer.add_scalar('train/batch_loss',
                                  loss.item(),
                                  len(dataloader) * (epoch - 1) + batch_num)
            tbar.set_postfix(loss="%.4f" % (total_loss / batch_num))
            tbar.update()
    return total_loss

def evaluateV6(multi_model, dataloader, accelerator):
    multi_model.eval()
    
    total_num=0
    correct_num=0
    with torch.no_grad():
        for (multi_input, labels) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            #multi_input=multi_input.to(accelerator.device)
            #positive_inputs=positive_inputs.to(accelerator.device)
            logits:Tensor=multi_model(**multi_input)
            answer=logits.argmax(dim=1)
            correct_num+=(answer==labels).sum().item()
            total_num+=len(labels)
    accelerator.print(f'acc:{correct_num/total_num}')
    return correct_num/total_num

def evaluateV6_2(multi_model, dataloader, accelerator):
    multi_model.eval()
    
    top1_num=0
    top5_num=0
    top10_num=0
    top20_num=0
    total_num=0
    with torch.no_grad():
        with tqdm(dataloader,
                  unit='batch',
                  total=len(dataloader) + 1,
                  desc='Evaluating...',
                  disable=not accelerator.is_local_main_process) as tbar:
            for multi_input in tbar:
            #multi_input=multi_input.to(accelerator.device)
            #positive_inputs=positive_inputs.to(accelerator.device)
                logits:Tensor=multi_model(**multi_input)
                probs=logits[:,0]
                top1=probs.topk(1,sorted=False)[1]
                top5=probs.topk(5,sorted=False)[1]
                top10=probs.topk(10,sorted=False)[1]
                top20=probs.topk(20,sorted=False)[1]
                top1_num += (0 in top1)
                top5_num += (0 in top5)
                top10_num += (0 in top10)
                top20_num += (0 in top20)
                total_num+=1
                top1_acc=top1_num/total_num
                top5_acc=top5_num/total_num
                top10_acc=top10_num/total_num
                top20_acc=top20_num/total_num
                tbar.set_postfix(acc=f'top1:{round(top1_acc,3)} top5:{round(top5_acc,3)} top10:{round(top10_acc,3)} top20:{round(top20_acc,3)}')
                tbar.update()
        
        accelerator.print(f'Top1:{top1_acc},Top5:{top5_acc},Top10:{top10_acc},Top20:{top20_acc}')
    return  [top1_acc,top5_acc,top10_acc,top20_acc]

def get_kg_embeddings(kg_path,processor,model,num=32):
    with open(kg_path,'r') as f:
        kg:dict=json.load(f)
    kg_abs=[]
    sample_keys=random.sample(kg.keys(),num)
    for key in sample_keys:
        kg_abs.append(abs_dict_to_str(kg[key]))
    kg_embeddings_input=processor(kg_abs,
                                  return_tensors="pt",
                                  padding="max_length",
                                  max_length=128,
                                  truncation=True)
    kg_embeddings_input=kg_embeddings_input.to(model.device)
    kg_embeddings=model(**kg_embeddings_input).last_hidden_state[:,0]
    return kg_embeddings

def save_model(model, name, accelerator):
    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model_name = model.__class__.__name__
    checkpoint_path = config.checkpoint_path
    if model_name not in os.listdir(checkpoint_path):
        os.mkdir(os.path.join(checkpoint_path, model_name))
    checkpoint_path = os.path.join(checkpoint_path, model_name)
    accelerator.save(unwrapped_model.state_dict(),
                     os.path.join(checkpoint_path, name))
    checkpoint_list = os.listdir(checkpoint_path)
    if (len(checkpoint_list) > config.max_checkpoint_num):
        file_map = {}
        times = []
        del_num = len(checkpoint_list) - config.max_checkpoint_num
        for f in checkpoint_list:
            t = f.split('.')[0].split('_')[-1]
            file_map[int(t)] = os.path.join(checkpoint_path, f)
            times.append(int(t))
        times.sort()
        for i in range(del_num):
            del_f = file_map[times[i]]
            os.remove(del_f)
    accelerator.print('Checkpoint has been updated successfully.\n')


def getlogger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
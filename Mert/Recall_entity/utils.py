import logging
import os
import sys
import torch
from torch import Tensor
from .config import config
from tqdm import tqdm



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
                candidates_embeddings: Tensor = entity_model(**candidates).last_hidden_state[0]
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

            negative_embedding = torch.stack(negative_embedding).requires_grad_()
            anchor_embedding = torch.stack(anchor_embedding).requires_grad_()
            positive_embedding.requires_grad_()

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
            tbar.set_postfix(loss="%.3f" % (total_loss / batch_num))
            tbar.update()
    return total_loss


def evaluate(multi_model, entity_model, dataloader, accelerator):
    multi_model.eval()
    entity_model.eval()
    correct_num=0
    total_num=0
    with torch.no_grad():
        for idx, (multi_input, mention_token_pos, positive_inputs,
                  negative_inputs) in tqdm(dataloader,
                                           unit='batch',
                                           total=len(dataloader) + 1,
                                           desc='Evaluating...',
                                           disable=not accelerator.is_local_main_process):
            multi_input=multi_input.to(accelerator.device)
            positive_inputs=positive_inputs.to(accelerator.device)
            text_embedding = multi_model(**multi_input).text_embeddings
            positive_embedding = entity_model(**positive_inputs).last_hidden_state[0]
            for idx, candidates in enumerate(negative_inputs):
                #negative_inputs(list).size:(batchsize,candidate_num)
                candidates=candidates.to(accelerator.device)
                candidates_embeddings: Tensor = entity_model(**candidates).last_hidden_state[0]
                entity_s, entity_e = mention_token_pos[idx]
                entity_embedding = torch.mean(text_embedding[idx][entity_s:entity_e], dim=0)
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

def save_model(model, name, accelerator):
    accelerator.print('Saving checkpoint...\n')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    model_name = name.split('_epoch')[0]
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
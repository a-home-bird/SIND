import sys
from networks_open_world import RelationClassification, LabelGeneration
from transformers import AdamW
from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import os
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from collections import Counter
from itertools import cycle
from utils import entropy,MarginLoss,cluster_acc,accuracy,AverageMeter,ClusterEvaluation
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import pickle
# ------------------------init parameters----------------------------

CUDA = "0"
DATASET = 'SemEval'  # dataset selection tacred,SemEval
NUM_LABELS = 19  # TACRED:42, SemEval:19
MAX_LENGTH = 128
BATCH_SIZE = 64
LABEL_BATCH_SIZE = 16
LR = 1e-4
EPS = 1e-8
EPOCHS = 10  #
TOTAL_EPOCHS = 1
MATE_EPOCHS = 10
seed_val = 42
UNLABEL_OF_TRAIN = 0.5  # Unlabel ratio
LABEL_OF_TRAIN = 0.5  # Label ratio
LAMBD = 0.2
Z = 10    # Incremental Epoch Number
Z_RATIO = Z / BATCH_SIZE
LOG_DIR = DATASET + '_' + str(int(LABEL_OF_TRAIN * 100)) 
os.system('mkdir ' + LOG_DIR)


#os.environ['CUDA_VISIBLE_DEVICES'] = CUDA
device = torch.device("cuda:0")

# ------------------------functions----------------------------

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])


# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# cited: https://github.com/INK-USC/DualRE/blob/master/utils/scorer.py#L26
def score(key, prediction,label_map = None, label_num = -1, verbose=True, NO_RELATION=0):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()
    predict =[]
    target = []
    # Loop over the data to compute a score
    if not label_map:
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1
            """ guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1 """
    else:
        for row in range(len(key)):
            gold = key[row]
            guess = prediction[row]
            if guess >= label_num:
                guess = label_map[guess]
            """ predict.append(guess)
            target.append(gold) """
            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1 
            
        
    # Print the aggregate score
    if verbose:
        pass
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    
    
    for i in correct_by_relation:
        print("relation:{}\t precision:{}\t recall:{}".format(i,correct_by_relation[i]/guessed_by_relation[i],correct_by_relation[i]/gold_by_relation[i]))

    
    return prec_micro, recall_micro, f1_micro

# ------------------------prepare sentences----------------------------

# Tokenize all of the sentences and map the tokens to thier word IDs.
def pre_processing(sentence_train, sentence_train_label):
    input_ids = []
    attention_masks = []
    labels = []
    e1_pos = []
    e2_pos = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        try:
            # Find e1(id:2487) and e2(id:2475) position
            pos1 = (encoded_dict['input_ids'] == 2487).nonzero()[0][1].item()
            pos2 = (encoded_dict['input_ids'] == 2475).nonzero()[0][1].item()
            e1_pos.append(pos1)
            e2_pos.append(pos2)
            # Add the encoded sentence to the list.
            input_ids.append(encoded_dict['input_ids'])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(sentence_train_label[i])
        except:
            pass
            #print(sent)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    e1_pos = torch.tensor(e1_pos, device='cuda')
    e2_pos = torch.tensor(e2_pos, device='cuda')
    w = torch.ones(len(e1_pos), device='cuda')

    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels, e1_pos, e2_pos, w)

    return train_dataset




# ------------------------training----------------------------
def define_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def caculate_pos_left(index_id,max_pos,average,variance):
    left_weight = torch.exp(-(max_pos-average)**2/(2 * variance))
    return left_weight * index_id
    pass

def compute_kld(p_logit, q_logit):
    
    return torch.sum(p_logit * (torch.log(p_logit + 1e-16) - torch.log(q_logit + 1e-16)), dim = -1)

def train(model, device, train_label_loader, train_unlabel_loader, optimizer , epoch,choose_k=2,threshold=0.95):
    model.train()
    bce = nn.BCELoss(reduction='none')
    unlabel_loader_iter = cycle(train_unlabel_loader)
    total_train_loss = AverageMeter('total_loss', ':.4e')
    total_entropy_loss = AverageMeter('entropy_loss', ':.4e')
    total_ce_loss = AverageMeter('ce_loss', ':.4e')
    total_bce_loss = AverageMeter('bce_loss', ':.4e')
    total_neg_bce_loss = AverageMeter('neg_bce_loss', ':.4e')
    total_fixmatch_loss = AverageMeter('fixmatch_loss', ':.4e')
    
    cosine = nn.CosineSimilarity(dim=-1)
    
    
    for step, batch_label in enumerate(train_label_loader):
        # Progress update every 40 batches.
        
        
        
        batch_unlabel = next(unlabel_loader_iter)
        # Unpack this training batch from our dataloader.
        b_input_ids_label = batch_label[0].to(device)
        b_input_mask_label = batch_label[1].to(device)
        b_labels_label = batch_label[2].to(device)
        b_e1_pos_label = batch_label[3].to(device)
        b_e2_pos_label = batch_label[4].to(device)
        
        b_input_ids_unlabel = batch_unlabel[0].to(device)
        b_input_mask_unlabel = batch_unlabel[1].to(device)

        b_labels_unlabel = batch_unlabel[2].to(device)

        b_e1_pos_unlabel = batch_unlabel[3].to(device)
        b_e2_pos_unlabel = batch_unlabel[4].to(device)

        
        optimizer.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch)
        logits_label, feat_label = model(b_input_ids_label,
                                token_type_ids=None,
                                attention_mask=b_input_mask_label,
                                e1_pos=b_e1_pos_label,
                                e2_pos=b_e2_pos_label,
                                )

        logits_unlabel, feat_unlabel = model(b_input_ids_unlabel,
                                token_type_ids=None,
                                attention_mask=b_input_mask_unlabel,
                                e1_pos=b_e1_pos_unlabel,
                                e2_pos=b_e2_pos_unlabel,
                                )
        
        logits_label_dual, feat_label_dual = model(b_input_ids_label,
                                token_type_ids=None,
                                attention_mask=b_input_mask_label,
                                e1_pos=b_e1_pos_label,
                                e2_pos=b_e2_pos_label,
                                )

        logits_unlabel_dual, feat_unlabel_dual = model(b_input_ids_unlabel,
                                token_type_ids=None,
                                attention_mask=b_input_mask_unlabel,
                                e1_pos=b_e1_pos_unlabel,
                                e2_pos=b_e2_pos_unlabel,
                                )
        
        feat_all = torch.cat([feat_label,feat_unlabel],dim=0)
        feat_dual = torch.cat([feat_label_dual,feat_unlabel_dual],dim=0)
        cos_sim = cosine(feat_all.unsqueeze(1), feat_dual.unsqueeze(0)) / 0.05
        con_labels = torch.arange(cos_sim.size(0)).long().cuda()
        
        cl_loss = F.cross_entropy(cos_sim, con_labels) 


        prob_label = F.softmax(logits_label, dim=1)
        prob_unlabel = F.softmax(logits_unlabel, dim=1)


        
        pseudo_label = prob_unlabel.detach()
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)

        index_id = targets_u.lt(9).float()
        index_ood = targets_u.ge(9).float()
       

       
        mask_id = (index_id * max_probs).ge(threshold).float()
       
                
        
        mask_id_weight = torch.where(mask_id == 1.0,max_probs,mask_id) 
        
            
        mask = mask_id_weight 


        feat = torch.cat([feat_label,feat_unlabel],dim=0)
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        
        labeled_len = len(b_labels_label)

        pos_pairs = []

        target_np = b_labels_label.cpu().numpy()
        
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
       
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)

         # f-bce
        choose_k = choose_k # this parameter should be fine-tuned with different task
        max_pos = torch.topk(cosine_dist[:labeled_len,pos_idx[:, 1]],choose_k ,dim = 0)[0][choose_k-1]
        mask_1 = (vals[:, 1] - max_pos).ge(0).float()
        mask_0 = (vals[:, 1] - max_pos).lt(0).float()

        pos_idx_1 = (pos_idx[:, 1] * mask_1).cpu().numpy()
        pos_idx_0 = (pos_idx[:, 0] * mask_0).cpu().numpy()
        pos_idx = (pos_idx_1 + pos_idx_0).flatten().astype(int).tolist()

       
        pos_pairs.extend(pos_idx)

        prob = torch.cat([prob_label,prob_unlabel], dim=0)
        batch_all = prob.shape[0]
        pos_prob = prob[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(batch_all, 1, -1), pos_prob.view(batch_all, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        
        
        row = list(range(len(pos_pairs)))
        weight = cosine_dist[row,pos_pairs]


        

        if  torch.sum(mask) > 0:
            #align = torch.log(p/q.to(device))
            fixmatch_loss = (F.cross_entropy(logits_unlabel,
                                             targets_u ,
                                             reduction='none') * mask).mean()
        if torch.sum(mask) == 0:
            fixmatch_loss = (F.cross_entropy(logits_unlabel,
                                             targets_u,
                                             reduction='none') * mask).mean()
       
        bce_loss = (bce(pos_sim, ones) * weight).mean()
        entropy_loss = entropy(torch.mean(prob, 0))
        ce_loss = (F.cross_entropy(logits_label, b_labels_label, reduction='none')).mean()
        
        
        loss =    - entropy_loss  +  ce_loss +  bce_loss + fixmatch_loss + cl_loss * 0.4
        
       
        total_train_loss.update(loss.detach().item())
        total_fixmatch_loss.update(fixmatch_loss.detach().item())
        total_bce_loss.update(bce_loss.detach().item())
       
        total_ce_loss.update(ce_loss.detach().item())
        total_entropy_loss.update(entropy_loss.detach().item())
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        torch.cuda.empty_cache()
        

    print("epoch:{}\taverage_total_loss:{}\taverage_bce_loss:{}\taverage_ce_loss:{}\taverage_entropy_loss:{}\taverage_fixmatch_loss:{}".format(epoch,total_train_loss.avg,total_bce_loss.avg,total_ce_loss.avg,total_entropy_loss.avg,total_fixmatch_loss.avg))
    print("average_neg_bce_loss:{}".format(total_neg_bce_loss.avg))
    
    return 0
    pass
def cluster_eval(unseen_label_ids,predict_labels):

    cluster_eval = ClusterEvaluation(unseen_label_ids, predict_labels).printEvaluation()
    print('B3', cluster_eval)
    # NMI, ARI, V_measure
    nmi = normalized_mutual_info_score
    print('NMI', nmi(unseen_label_ids, predict_labels))
    print('ARI', adjusted_rand_score(unseen_label_ids, predict_labels))
    print('Homogeneity', homogeneity_score(unseen_label_ids, predict_labels))
    print('Completeness', completeness_score(unseen_label_ids, predict_labels))
    print('V_measure', v_measure_score(unseen_label_ids, predict_labels))
    return {'B3':cluster_eval,'NMI':nmi(unseen_label_ids, predict_labels)}

from sklearn.cluster import DBSCAN, KMeans
def cluster_val(model, device, dev_loader):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)

            output , embedding = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos,
                                        )
            targets = np.append(targets, b_labels.cpu().numpy())
            embeddings.append(embedding)
            
    targets = targets.astype(int)
    sent_embs = torch.cat(embeddings,dim=0).cpu()

    

    print("data dimension is {}. ".format(sent_embs.shape[-1]))
    clusters = KMeans(n_clusters=10, n_init=20)  #kmeans
    predict_labels = clusters.fit_predict(sent_embs)
    cluster_eval(targets,predict_labels)
    

    
def val(model, labeled_num, device, dev_loader,epoch):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    embeddings = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dev_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_e1_pos = batch[3].to(device)
            b_e2_pos = batch[4].to(device)

            output , feat = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos,
                                        )
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            embeddings.append(feat.cpu().numpy())
            targets = np.append(targets, b_labels.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())

    targets = targets.astype(int)
    preds = preds.astype(int)
   
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc, label_map = cluster_acc(preds, targets,labeled_num)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc, unlabel_map = cluster_acc(preds[unseen_mask], targets[unseen_mask],labeled_num)
    
    
    print("-"*10 + "epoch:{}".format(epoch) + "-"*10)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    print("SET NO_RELATION ID: 0")
    

    prec_micro, recall_micro, f1_micro = score(targets[seen_mask],preds[seen_mask])
    print("seen_label Precision (micro): {:.3%}".format(prec_micro))
    print("seen_lable   Recall (micro): {:.3%}".format(recall_micro))
    print("seen_label       F1 (micro): {:.3%}".format(f1_micro))

    prec_micro, recall_micro, f1_micro = score(targets[unseen_mask],preds[unseen_mask],unlabel_map,labeled_num)
    print("unseen_label Precision (micro): {:.3%}".format(prec_micro))
    print("unseen_lable   Recall (micro): {:.3%}".format(recall_micro))
    print("unseen_label       F1 (micro): {:.3%}".format(f1_micro))

    prec_micro, recall_micro, f1_micro = score(targets,preds,label_map,labeled_num)
    print("total_label Precision (micro): {:.3%}".format(prec_micro))
    print("total_lable   Recall (micro): {:.3%}".format(recall_micro))
    print("total_label       F1 (micro): {:.3%}".format(f1_micro))
    print("-"*20)
    
    result_clu = cluster_eval(targets,preds)
    result = {"overall_acc":overall_acc}
    for key , val in result_clu.items():
        result[key] = val
    

    return overall_acc , result
    pass





from collections import Counter
def main(argv=None):
    
    define_random_seed(seed_val)
    
    result_total = []
    for threshold in [0.95]:
        cash_file = "data/cash_semeval/semeval_0.05"
        if not os.path.exists(cash_file):
            os.makedirs(cash_file)
            print("{} is completed".format(cash_file))
        

        label_id = json.load(open('data/' + DATASET + '/relation2id.json', 'r'))
        
        label_num = int(len(label_id.values()) * 0.5)
        sentence_train_label = json.load(open('data/' + DATASET + '/train_label_id.json', 'r'))
        
        with open(os.path.join(cash_file,"labeled_dataset.pth"),"rb") as f:
                labeled_dataset = pickle.load(f)
        with open(os.path.join(cash_file,"unlabeled_dataset.pth"),"rb") as f:
                unlabeled_dataset_total = pickle.load(f)
       


        
        labeled_dataloader = DataLoader(
            labeled_dataset,  # The training samples.
            sampler=RandomSampler(labeled_dataset),  # Select batches randomly
            batch_size=LABEL_BATCH_SIZE  # Trains with this batch size.
            #drop_last=True
        )
        

        unlabeled_dataloader = DataLoader(
                unlabeled_dataset_total,  # The training samples.
                sampler=RandomSampler(unlabeled_dataset_total),  # Select batches randomly
                batch_size=BATCH_SIZE-LABEL_BATCH_SIZE  # Trains with this batch size.
            )
        

        cached_val_dataset_file = os.path.join("./data/cash_semeval","val_dataset.pth")

        if not os.path.exists(cached_val_dataset_file):
            print(" validation dataset loss")
            exit()
        else:
            with open(cached_val_dataset_file , "rb") as f:
                val_dataset = pickle.load(f)

        validation_dataloader = DataLoader(
            val_dataset,  # The validation samples.
            sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
            batch_size=BATCH_SIZE  # Evaluate with this batch size.
        )

        cached_test_dataset_file = os.path.join("./data/cash_semeval","test_dataset.pth")

        if not os.path.exists(cached_test_dataset_file):
            print(" test dataset loss")
            exit()
        
        else:
            with open(cached_test_dataset_file , "rb") as f:
                test_dataset = pickle.load(f)

        test_dataloader = DataLoader(
            test_dataset,  # The validation samples.
            sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
            batch_size=BATCH_SIZE  # Evaluate with this batch size.
        )


        pretrain_ckpt = '/home/jsj201-2/mount1/zdg/SSRE/MetaSRE/PLM/bert-base-uncased'
        
        
        modelf1 = RelationClassification.from_pretrained(
            pretrain_ckpt,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=int(NUM_LABELS),  # The number of output labels--2 for binary classification.
            
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        
        modelf1 = nn.DataParallel(modelf1, device_ids=[0, 1]) 
        modelf1.cuda()

        
       

        optimizer1 = AdamW(modelf1.parameters(),
                        lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps=EPS  # args.adam_epsilon  - default is 1e-8.
                        )
        
        #训练每个epoch，label参与unlabel list的长度为MATE_EPOCHS的训练过程
        
        total_steps1 = EPOCHS * len(labeled_dataloader)
        
        total_t0 = time.time()

        # ========================================
        #               train
        # ========================================

        print("training start...")
        cnt = 0
        mean_uncert = 1
        best_performance = 0
        
        best_model_path = "./best_model/"
        
        for epoch in range(EPOCHS):
            
            #mean_uncert = 0
            mean_uncert = train(modelf1, device, labeled_dataloader, unlabeled_dataloader, optimizer1,epoch=epoch)
            overall_acc , _ = val(modelf1, label_num, device, validation_dataloader,epoch)
            if overall_acc > best_performance:
                best_performance = overall_acc
                if not os.path.exists(best_model_path):
                    os.makedirs(best_model_path)
                torch.save(modelf1.state_dict(),os.path.join(best_model_path,"best_model_ce_bce(weight)_fixmatch_entropy.pt"))
        
        
        modelf1.load_state_dict(torch.load(os.path.join(best_model_path,"best_model_ce_bce(weight)_fixmatch_entropy.pt")))
        
        _ , result = val(modelf1, label_num, device, test_dataloader,-1)
        result_total.append(result)
        
        print("------------------------------ threshold = {}".format(threshold))
        

    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == "__main__":
    sys.exit(main())

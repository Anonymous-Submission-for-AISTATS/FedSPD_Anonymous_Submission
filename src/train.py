import copy
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from update import DatasetSplit, LocalUpdateL, LocalUpdateA, test_inference_UB3

def multiple_training_M(args, models, lw, ll, idx, epoch, train_dataset, user_groups, up_cluster, logger):
    models[up_cluster].train()
    local_model = LocalUpdateL(args=args, dataset=train_dataset,
                                      idxs=user_groups, logger=logger, idx = idx, ep = epoch)
    w, loss = local_model.update_weights(
        model=copy.deepcopy(models[up_cluster]), global_round=epoch)
    lw[idx] = copy.deepcopy(w)
    ll[idx] = copy.deepcopy(loss)

def agg_train_UB_M(users, models, args, train_dataset, test_dataset, user_groups, GG, logger, ta, tl, eps, lr_decay_ft, tidx):
    mod = users % 5
    w = [None] * args.clusters
    N = [0] * args.clusters
    P = [0] * args.clusters
    for cluster in range(args.clusters):
        w[cluster] = models[cluster].state_dict()
        N[cluster] = len(GG[cluster])
        P[cluster] = N[cluster]/len(user_groups)
    for key in w[0]:
        w[0][key] = P[0]*w[0][key]
        for i in range(1, args.clusters):
            w[0][key] += P[i]*w[i][key]
    models[0].load_state_dict(w[0])
    for epoch in tqdm(range(eps)):
        if (epoch%3 == 0) and (epoch!=0):
            args.lr = lr_decay_ft*args.lr
        models[0].train()
        local_model = LocalUpdateA(args=args, dataset=train_dataset,
                                      idxs=user_groups, logger=logger, idx = users, ep = mod)
        w, loss = local_model.update_weights(
                model=models[0], global_round=epoch)
        models[0].load_state_dict(w)
        ta, tl = test_inference_UB3(args, models[0], test_dataset, users, ta, tl, epoch, tidx)


def data_clustering_M(MM, args, user_groups, train_dataset, GG, id, gs, eps, gsp):
    
    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(DatasetSplit(train_dataset, user_groups), batch_size=1,
                            shuffle=False, drop_last = True)
    BB = {}
    outputs = [0] * args.clusters
    batch_loss = [0] * args.clusters
    for cluster in range(args.clusters):
        BB[cluster] = []
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        for cluster in range(args.clusters):
            MM[cluster].eval()
            outputs[cluster] = MM[cluster](images)
            batch_loss[cluster] = criterion(outputs[cluster], labels)
        batch_loss = [abs(num) for num in batch_loss]
        BB[batch_loss.index(min(batch_loss))].append(batch_idx)
    for keys in BB:
        if len(BB[keys]) < 10:
            m_key = max(BB, key=lambda k: len(BB[k]))
            BB[keys].extend(BB[m_key][-10:])
            BB[m_key] = BB[m_key][:-10]
    operation = True
    while operation:
        operation = False
        for keys in BB:
            if len(BB[keys]) % args.local_bs == 1:
                operation = True
                m_key = max(BB, key=lambda k: len(BB[k]))
                if m_key != keys:
                    BB[keys].append(BB[m_key][-1])
                    BB[m_key] = BB[m_key][:-1]
                else:
                    sorted_keys = sorted(BB, key=lambda k: len(BB[k]), reverse=True)
                    BB[keys].append(BB[sorted_keys[1]][-1])
                    BB[sorted_keys[1]] = BB[sorted_keys[1]][:-1]
    idx_set = set()
    for cluster in range(args.clusters):
        GG[cluster] = user_groups[BB[cluster]]
        idx_set = idx_set | set(GG[cluster])

    for cluster in range(args.clusters):
        gs[id, cluster] = len(list(GG[cluster]))
        gsp[id, cluster, eps] = gs[id, cluster]/len(user_groups)

    print('Data Clustered for client' + str(id))
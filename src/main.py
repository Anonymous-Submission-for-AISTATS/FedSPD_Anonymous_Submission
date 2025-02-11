# The code is modified from the following Github repository:
# Federated-Learning (PyTorch)
# https://github.com/AshwinRJ/Federated-Learning-PyTorch

import os
import time
import pickle
import numpy as np
import scipy.io
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdateL, test_inference_UB2_M
from models import CNNMnist, CNNNet
from utils import get_dataset_ubc, exp_details, checkCluster_M, GG_average_weights_M, average_of_dict_elements, genGraph
from train import multiple_training_M, data_clustering_M, agg_train_UB_M

if __name__ == '__main__':
    start_time = time.time()

    # Define Paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    # Args nitialization
    args = args_parser()
    exp_details(args)
    lr_ori = args.lr
    exps = 'C_25'
    Gp = 0.25
    radius = 0.20
    exn = 5
    args.gpu = None
    
    if args.gpu:
        device_count = torch.cuda.device_count()
        print(f"{device_count} GPUs available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset & User Groups
    train_dataset, test_dataset, user_groups, tidx = get_dataset_ubc(args)
    for key in user_groups:
        user_groups[key] = np.asarray(list(user_groups[key]))

    # GenGraph
    Adj = genGraph(args.graph, args.num_users, Gp, radius, exn)
    AdjT = torch.from_numpy(Adj)


    # Build Model
    agg_models = {}
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':            
            for i in range(args.num_users):
                agg_models[i] = [None] * args.clusters
                for j in range(args.clusters):
                    agg_models[i][j] = CNNMnist(args=args)
        elif args.dataset == 'emnist':            
            for i in range(args.num_users):
                agg_models[i] = [None] * args.clusters
                for j in range(args.clusters):
                    agg_models[i][j] = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            for i in range(args.num_users):
                agg_models[i] = [None] * args.clusters
                for j in range(args.clusters):
                    agg_models[i][j] = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            for i in range(args.num_users):
                agg_models[i] = [None] * args.clusters
                for j in range(args.clusters):
                    agg_models[i][j] = CNNNet(args=args)
        elif args.dataset == 'cifar100':
            for i in range(args.num_users):
                agg_models[i] = [None] * args.clusters
                for j in range(args.clusters):
                    agg_models[i][j] = CNNNet(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    for i in range(args.num_users):
        for j in range(args.clusters):
            agg_models[i][j].to(device)
            agg_models[i][j].train()

    #Copy Weights
    agg_weights = {}
    for i in range(args.num_users):
        agg_weights[i] = [None] * args.clusters
        for j in range(args.clusters):
            agg_weights[i][j] = agg_models[i][j].state_dict()

    # Initialization
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    GG = {}
    gs = np.zeros((args.num_users, args.clusters))
    gsp = np.zeros((args.num_users, args.clusters, args.epochs))

    for users in range(args.num_users):
        GG[users] = [None] * args.clusters
        ug = user_groups[users]
        data_number = np.size(ug)
        clusterwise_number = int(data_number/(args.clusters))
        portion = clusterwise_number/(args.clusters+1)
        total_idx1 = set(np.arange(int(data_number/2)))
        total_idx2 = set(np.arange(int(data_number/2), data_number))
        for i in range(args.clusters-1):
            c1 = set(np.random.choice(np.array(list(total_idx1)), int((i+1)*portion), replace=False))
            c1x = set(np.random.choice(np.array(list(total_idx2)), int((args.clusters-i)*portion), replace=False))
            c1.update(c1x)
            GG[users][i] = ug[np.array(list(c1)).astype(int)]
            total_idx1 = total_idx1 - c1
            total_idx2 = total_idx2 - c1
        total_idx1.update(total_idx2)
        GG[users][args.clusters-1] = ug[np.array(list(total_idx1)).astype(int)]
        for i in range(args.clusters):
            gs[users, i] = len(list(GG[users][i]))

    # Training Process
    for epoch in tqdm(range(args.epochs)):
        if (epoch != 0) and (epoch%args.decay_rnd == 0):
            args.lr = args.lr_decay_tr*args.lr
        local_weights, local_losses = {}, {}
        for i in range(args.clusters):
            local_weights[i], local_losses[i] = [None] * int(args.num_users), [0] * int(args.num_users)
        print(f'\n | Global Training Round : {epoch+1} |\n')
        allu = set(list(np.arange(args.num_users)))
        US = {}

        # Training & Aggregation
        for update_cluster in range(args.clusters):
            print(f'\n Cluster: {update_cluster} \n')
            if epoch > 0:
                US[update_cluster] = list(np.random.choice(np.array(list(allu)), int(args.num_users/args.clusters), replace = False))
                allu = allu - set(US[update_cluster])
            else:
                US[update_cluster] = list(np.arange(args.num_users))

            for idx in US[update_cluster]:
                multiple_training_M(args, agg_models[idx], local_weights[update_cluster], local_losses[update_cluster], idx, epoch, train_dataset, GG[idx][update_cluster], update_cluster, logger)
            for idx in US[update_cluster]:
                global_weights = GG_average_weights_M(local_weights[update_cluster], Adj, idx, US[update_cluster])
                agg_models[idx][update_cluster].load_state_dict(global_weights)
        avg_loss = average_of_dict_elements(local_losses)
        train_loss.append(avg_loss)
        
        # Data Clustering & Cluster Checking
        if epoch%args.ckc == 0:
            print("Checking Cluster")
            checkCluster_M(args, agg_models)    
        for idx in range(args.num_users):
            data_clustering_M(agg_models[idx], args, user_groups[idx], train_dataset, GG[idx], idx, gs, epoch, gsp)

        # Calculate Average Training Accuracy
        list_acc, list_loss = [], []
        for cluster in range(args.clusters):
            for users in US[cluster]:
                agg_models[users][cluster].eval()
                local_model = LocalUpdateL(args=args, dataset=train_dataset,
                                        idxs=GG[users][cluster], logger=logger, idx = users, ep = epoch)
                acc, loss = local_model.inference(model=agg_models[users][cluster])
                list_acc.append(acc)
                list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # Print Global Training Loss
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Save Models
    for i in range(args.num_users):
        for c in range(args.clusters):
            name = '../save/' + exps + '/C' + str(c) + '_' + str(i) + '.pickle'
            torch.save(agg_models[i][c].state_dict(), name)

    # Final Phase
    ta = np.zeros((args.num_users, args.ft_ep))
    tl = np.zeros((args.num_users, args.ft_ep))

    for users in range(args.num_users):
        args.lr = lr_ori/2
        agg_train_UB_M(users, agg_models[users], args, train_dataset, test_dataset, user_groups[users], GG[users], logger, ta, tl, args.ft_ep, args.lr_decay_ft, tidx[users%5])
    tac = np.mean(ta, axis = 0)
    tls = np.mean(tl, axis = 0)
    print("Finish Aggregate Training")

    # Save Aggregated Model
    for i in range(args.num_users):
        nameA = '../save/'+ exps + '/A_' + str(i) + '.pickle'
        torch.save(agg_models[i][0].state_dict(), nameA)

    # Test Inference
    test_acc, test_loss = test_inference_UB2_M(args, agg_models, test_dataset, tidx)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*torch.mean(test_acc)))
    
    # Saving Data:
    file_name = '../save/' + exps + '/N{}_{}_{}_C[{}]_LR[{}]_E[{}]_B[{}].pkl'.\
        format(args.num_users, args.dataset, args.epochs, args.frac, args.lr,
               args.local_ep, args.local_bs)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    mat_name = '../save/' + exps + '/N{}_{}_{}_C[{}]_LR[{}]_E[{}]_B[{}].mat'.\
        format(args.num_users, args.dataset, args.epochs, args.frac, args.lr,
               args.local_ep, args.local_bs)
    TA = test_acc.numpy()
    TL = test_loss.numpy()
    data = {}
    data['TrainLoss'] = np.array(train_loss)
    data['TestLoss'] = TL
    data['TestLossAvg'] = np.mean(TL)
    data['TrainAcc'] = np.array(train_accuracy)
    data['TestAcc'] = TA
    data['TestAccAvg'] = np.mean(TA)
    data['TestTrainAcc'] = tac
    data['TestTrainLoss'] = tls
    data['DataClusters'] = gs
    data['DataClustersPortion'] = gsp
    scipy.io.savemat(mat_name, data)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

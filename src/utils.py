#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import copy
import torch
import igraph
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_rotated2, mnist_rotated, unbalanced, unbalanced_lab
from sampling import cifar_iid, cifar_noniid
from torch.utils.data import Subset
import random
from scipy.optimize import linear_sum_assignment

def get_dataset_ubc(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    tidx = {}
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip()])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        apply_transform2 = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(), transforms.RandomRotation((90, 90))])

        train_dataset2 = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform2)

        test_dataset2 = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform2)
        tidx1 = range(25000)
        tidx2 = range(25000, 50000)
        ttidx1 = range(5000)
        ttidx2 = range(5000, 10000)
        train_subset1 = Subset(train_dataset, tidx1)
        train_subset2 = Subset(train_dataset2, tidx2)
        test_subset1 = Subset(test_dataset, ttidx1)
        test_subset2 = Subset(test_dataset2, ttidx2)
        train_dataset = torch.utils.data.ConcatDataset([train_subset1, train_subset2])
        test_dataset = torch.utils.data.ConcatDataset([test_subset1, test_subset2])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            # user_groups = mnist_rotated2(train_dataset, args.num_users)
            user_groups = unbalanced(train_dataset, args.num_users)
            tidx = unbalanced_lab(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'cifar100':
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip()])

        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        apply_transform2 = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.RandomCrop(32, padding=4, padding_mode='reflect'), transforms.RandomHorizontalFlip(), transforms.RandomRotation((90, 90))])

        train_dataset2 = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform2)

        test_dataset2 = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform2)
        tidx1 = range(25000)
        tidx2 = range(25000, 50000)
        ttidx1 = range(5000)
        ttidx2 = range(5000, 10000)
        train_subset1 = Subset(train_dataset, tidx1)
        train_subset2 = Subset(train_dataset2, tidx2)
        test_subset1 = Subset(test_dataset, ttidx1)
        test_subset2 = Subset(test_dataset2, ttidx2)
        train_dataset1 = torch.utils.data.ConcatDataset([train_subset1, train_subset2])
        test_dataset1 = torch.utils.data.ConcatDataset([test_subset1, test_subset2])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = unbalanced(train_dataset, args.num_users)
            tidx = unbalanced_lab(test_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist' or 'emnist':
        print(args.dataset)
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        elif args.dataset == 'mnist':
            data_dir = '../data/fmnist/'
        else:
            data_dir = '../data/emnist/'

        if args.dataset == 'mnist':
            apply_transform1 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

            train_dataset1 = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform1)

            test_dataset1 = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform1)
            apply_transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), transforms.RandomRotation((90, 90))])

            train_dataset2 = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform2)

            test_dataset2 = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform2)
            
            tidx1 = range(30000)
            tidx2 = range(30000, 60000)
            ttidx1 = range(5000)
            ttidx2 = range(5000, 10000)
            train_subset1 = Subset(train_dataset1, tidx1)
            train_subset2 = Subset(train_dataset2, tidx2)
            test_subset1 = Subset(test_dataset1, ttidx1)
            test_subset2 = Subset(test_dataset2, ttidx2)
            train_dataset = torch.utils.data.ConcatDataset([train_subset1, train_subset2])
            test_dataset = torch.utils.data.ConcatDataset([test_subset1, test_subset2])

            # sample training data amongst users
            if args.iid:
                # Sample non-IID user data from Mnist
                user_groups = unbalanced(train_dataset, args.num_users)
                tidx = unbalanced_lab(test_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose euqal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)

        else:
            apply_transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

            train_dataset1 = datasets.EMNIST(data_dir, split='letters', train=True, download=True,
                                        transform=apply_transform1)
            test_dataset1 = datasets.EMNIST(data_dir, split='letters', train=False, download=True,
                                        transform=apply_transform1)
            apply_transform2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)), transforms.RandomRotation((90, 90))])
            train_dataset2 = datasets.EMNIST(data_dir, split='letters', train=True, download=True,
                                        transform=apply_transform2)

            test_dataset2 = datasets.EMNIST(data_dir, split='letters', train=False, download=True,
                                        transform=apply_transform2)
            '''
            print(train_dataset1.targets[0:100])
            train_dataset1.targets[train_dataset1.targets == 26] = 0
            test_dataset1.targets[test_dataset1.targets == 26] = 0
            train_dataset2.targets[train_dataset2.targets == 26] = 0
            test_dataset2.targets[test_dataset2.targets == 26] = 0
            '''
            train_dataset1.targets -= 1
            train_dataset2.targets -= 1
            test_dataset1.targets -= 1
            test_dataset2.targets -= 1
            print(train_dataset2.targets.shape)
            print(test_dataset2.targets.shape)

            tidx1 = range(62400)
            tidx2 = range(62400, 124800)
            ttidx1 = range(10400)
            ttidx2 = range(10400, 20800)
            train_subset1 = Subset(train_dataset1, tidx1)
            train_subset2 = Subset(train_dataset2, tidx2)
            test_subset1 = Subset(test_dataset1, ttidx1)
            test_subset2 = Subset(test_dataset2, ttidx2)
            train_dataset = torch.utils.data.ConcatDataset([train_subset1, train_subset2])
            test_dataset = torch.utils.data.ConcatDataset([test_subset1, test_subset2])
            user_groups = unbalanced(train_dataset, args.num_users)
            tidx = unbalanced_lab(test_dataset, args.num_users)

    elif args.dataset == 'emnist':
        data_dir = '../data/emnist/'
        print("I am Here.")

        apply_transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        train_dataset1 = datasets.EMNIST(data_dir, split='letters', train=True, download=True,
                                       transform=apply_transform1)

        test_dataset1 = datasets.EMNIST(data_dir, split='letters', train=False, download=True,
                                      transform=apply_transform1)
        print(train_dataset1.targets[0:100])
        train_dataset1.targets[train_dataset1.targets == 26] = 0
        test_dataset1.targets[test_dataset1.targets == 26] = 0
        apply_transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)), transforms.RandomRotation((90, 90))])

        train_dataset2 = datasets.EMNIST(data_dir, split='letters', train=True, download=True,
                                       transform=apply_transform2)

        test_dataset2 = datasets.EMNIST(data_dir, split='letters', train=False, download=True,
                                      transform=apply_transform2)
        train_dataset2.targets[train_dataset2.targets == 26] = 0
        test_dataset2.targets[test_dataset2.targets == 26] = 0
        print('Train/Test Size')
        print(train_dataset2.targets.shape)
        print(test_dataset2.targets.shape)
        tidx1 = range(44400)
        tidx2 = range(44400, 88800)
        ttidx1 = range(7200)
        ttidx2 = range(7200, 14800)
        train_subset1 = Subset(train_dataset1, tidx1)
        train_subset2 = Subset(train_dataset2, tidx2)
        test_subset1 = Subset(test_dataset1, ttidx1)
        test_subset2 = Subset(test_dataset2, ttidx2)
        train_dataset = torch.utils.data.ConcatDataset([train_subset1, train_subset2])
        test_dataset = torch.utils.data.ConcatDataset([test_subset1, test_subset2])
        user_groups = mnist_rotated2(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, tidx


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def G_average_weights(w, Adj, idx, Sec):
    """
    Returns the average of the weights.
    """
    if Sec:
        midx = idx-25
    else:
        midx = idx
    w_avg = copy.deepcopy(w[midx])
    for key in w_avg.keys():
        ngs = 1
        for i in range(len(w)):
            if Adj[idx, i+25] == 1 and i != midx:
                w_avg[key] += w[i][key]
                ngs += 1
        w_avg[key] = torch.div(w_avg[key], ngs)
    return w_avg

def GG_average_weights(w, Adj, idx, id, U):
    """
    Returns the average of the weights.
    """
    
    w_avg = copy.deepcopy(w[idx])
    for key in w_avg.keys():
        ngs = 1
        for i in range(len(U)):
            if Adj[idx, U[i]] == 1 and U[i] != idx:
                w_avg[key] += w[U[i]][key]
                ngs += 1
        w_avg[key] = torch.div(w_avg[key], ngs)
    return w_avg

def GG_average_weights_M(w, Adj, idx, U):
    """
    Returns the average of the weights.
    """
    
    w_avg = copy.deepcopy(w[idx])
    for key in w_avg.keys():
        ngs = 1
        for i in range(len(U)):
            if Adj[idx, U[i]] == 1 and U[i] != idx:
                w_avg[key] += w[U[i]][key]
                ngs += 1
        w_avg[key] = torch.div(w_avg[key], ngs)
    return w_avg

def GGG_average_weights(w, Adj, idx, id, U):
    """
    Returns the average of the weights.
    """
    
    w_avg = copy.deepcopy(w[idx])
    for key in w_avg.keys():
        ngs = 1
        for i in range(len(U)):
            if Adj[idx, U[i]] == 1 and U[i] != idx:
                w_avg[key] += w[U[i]][key]
                ngs += 1
        w_avg[key] = torch.div(w_avg[key], ngs)
    return w_avg

def checkCluster(pvn, args, agg1, agg2, ref1, ref2):
    pvn = 2
    pv1 = [None] * pvn
    pv2 = [None] * pvn
    sel_users = random.sample(range(args.num_users), pvn)
    for i in range(pvn):
        pv1[i] = copy.deepcopy(agg1[sel_users[i]])
        pv2[i] = copy.deepcopy(agg2[sel_users[i]])

    cos11 = torch.nn.CosineSimilarity(dim=0)
    cos12 = torch.nn.CosineSimilarity(dim=0)
    cos21 = torch.nn.CosineSimilarity(dim=0)
    cos22 = torch.nn.CosineSimilarity(dim=0)
    params11 = []
    params12 = []
    params21 = []
    params22 = []

    for param in pv1[0].parameters():
        params11.append(param.view(-1))

    for param in pv1[1].parameters():
        params12.append(param.view(-1))

    for param in pv2[0].parameters():
        params21.append(param.view(-1))

    for param in pv2[1].parameters():
        params22.append(param.view(-1))

    params11 = torch.cat(params11)
    params12 = torch.cat(params12)
    params21 = torch.cat(params21)
    params22 = torch.cat(params22)

    with torch.no_grad():
        # Same input twice --> returns 1
        s11 = cos11(params11, params21)
        s12 = cos12(params11, params22)
        s21 = cos21(params12, params21)
        s22 = cos22(params12, params22)
        s1 = s11 + s22
        s2 = s12 + s21
    if s2 > s1:
        agg1[sel_users[1]].load_state_dict(pv2[1].state_dict())
        agg2[sel_users[1]].load_state_dict(pv1[1].state_dict())
        M11 = pv1[0].state_dict()
        M12 = pv2[0].state_dict()
        M21 = pv2[1].state_dict()
        M22 = pv1[1].state_dict()
        # Average all parameters
        for key in M21:
            M11[key] = (M11[key] + M21[key]) / 2
        for key in M22:
            M12[key] = (M12[key] + M22[key]) / 2
        ref1.load_state_dict(M11)
        ref2.load_state_dict(M12)
    else:
        M11 = pv1[0].state_dict()
        M12 = pv2[0].state_dict()
        M21 = pv1[1].state_dict()
        M22 = pv2[1].state_dict()
        # Average all parameters
        for key in M21:
            M11[key] = (M11[key] + M21[key]) / 2
        for key in M22:
            M12[key] = (M12[key] + M22[key]) / 2
        ref1.load_state_dict(M11)
        ref2.load_state_dict(M12)
    paramsr1 = []
    paramsr2 = []
    for param in ref1.parameters():
        paramsr1.append(param.view(-1))

    for param in ref2.parameters():
        paramsr2.append(param.view(-1))

    paramsr1 = torch.cat(paramsr1)
    paramsr2 = torch.cat(paramsr2)

    for i in range(args.num_users):
        cos11 = torch.nn.CosineSimilarity(dim=0)
        cos12 = torch.nn.CosineSimilarity(dim=0)
        cos21 = torch.nn.CosineSimilarity(dim=0)
        cos22 = torch.nn.CosineSimilarity(dim=0)
        params1 = []
        params2 = []



        for param in agg1[i].parameters():
            params1.append(param.view(-1))

        for param in agg2[i].parameters():
            params2.append(param.view(-1))

        params1 = torch.cat(params1)
        params2 = torch.cat(params2)

        with torch.no_grad():
            # Same input twice --> returns 1
            s11 = cos11(paramsr1, params1)
            s12 = cos12(paramsr1, params2)
            s21 = cos21(paramsr2, params1)
            s22 = cos22(paramsr2, params2)
            s1 = s11 + s22
            s2 = s12 + s21
        if s2 > s1:
            tp1 = copy.deepcopy(agg1[i])
            tp2 = copy.deepcopy(agg2[i])
            agg1[i].load_state_dict(tp2.state_dict())
            agg2[i].load_state_dict(tp1.state_dict())

def hungarian_algorithm(cost_matrix):
    """
    Solves the assignment problem using the Hungarian algorithm.
    
    Parameters:
    - cost_matrix: A 2D numpy array of shape (M, M) representing the cost of assigning each node in one set to each node in the other set.
    
    Returns:
    - row_indices: Indices of nodes on one side.
    - col_indices: Indices of nodes on the other side corresponding to the optimal assignment.
    - total_cost: The total cost of the optimal assignment.
    """
    # Use the linear sum assignment function to find the minimum cost matching
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Calculate the total cost of the optimal assignment
    total_cost = cost_matrix[row_indices, col_indices].sum()
    
    return row_indices, col_indices, total_cost

def checkCluster_M(args, agg):
    cost_mat = np.zeros((args.clusters, args.clusters))
    cos_mat = [[None for _ in range(args.clusters)] for _ in range(args.clusters)]
    for i in range(args.clusters):
        for j in range(args.clusters):
            cos_mat[i][j] = torch.nn.CosineSimilarity(dim=0)
    
    pv1 = [None] * args.clusters
    pv2 = [None] * args.clusters
    weights1 = [None] * args.clusters
    weights2 = [None] * args.clusters
    for i in range(args.clusters):
        pv1[i] = copy.deepcopy(agg[0][i])
    for users in range(1, args.num_users):
        print('Check Cluster for Client: ' + str(users))
        para1 = [[] for _ in range(args.clusters)]
        para2 = [[] for _ in range(args.clusters)]
        for cluster in range(args.clusters):
            pv2[cluster] = copy.deepcopy(agg[users][cluster])
            for param in pv1[cluster].parameters():
                para1[cluster].append(param.view(-1))
            for param in pv2[cluster].parameters():
                para2[cluster].append(param.view(-1))
            para1[cluster] = torch.cat(para1[cluster])
            para2[cluster] = torch.cat(para2[cluster])
        for i in range(args.clusters):
            for j in range(args.clusters):
                cost_mat[i, j] = cos_mat[i][j](para1[i], para2[j])
        
        row_indices, col_indices, total_cost = hungarian_algorithm(cost_mat)
        for i in range(args.clusters):
            agg[users][i].load_state_dict(pv2[col_indices[i]].state_dict())
            weights1[i] = pv1[i].state_dict()
            weights2[i] = agg[users][i].state_dict()
            for key in weights1[i]:
                weights1[i][key] = ((users)*weights1[i][key] + weights2[i][key]) / (users+1)
            pv1[i].load_state_dict(weights1[i])
        
def average_of_dict_elements(data_dict):
    total_sum = 0
    total_count = 0
    
    for key, values_list in data_dict.items():
        total_sum += sum(values_list)
        total_count += len(values_list)
    
    # Calculate the average
    if total_count == 0:
        return 0  # Avoid division by zero
    average = total_sum / total_count
    return average

def genGraph(Gtype, N, Gp, r, exn):
    if Gtype == 'rgg':
        Adj = genRGG(r, N)
    elif Gtype == 'ba':
        if exn <= 1:
            exn = 2 
        Adj = genBA(int(exn), N)
    else:
        if Gp > 1 or Gp < 0:
            Gp = 0.3
        Adj = genER(Gp, N)
    return Adj


def genER(Gp, N):
    Adj = np.ones((N, N))
    for i in range(N):
        nums = np.random.choice([0, 1], size=N, p=[1-Gp, Gp])
        Adj[i, i+1:N] = nums[i+1:N]
        Adj[i+1:N, i] = nums[i+1:N]
    return Adj

def genBA(exn, N):
    GG = igraph.Graph.Barabasi(N, exn)
    Adj = np.array(GG.get_adjacency().data)
    for i in range(N):
        Adj[i, i] = 1
    return Adj

def genRGG(r, N):
    randx = np.random.rand(N)
    randy = np.random.rand(N)
    Adj = np.zeros((N, N))
    r2 = r*r

    for i in range(N):
        for j in range(i, N):
            if i == j:
                Adj[i, j] = 1
            else:
                dist = (randx[i]-randx[j])**2 + (randy[i]-randy[j])**2
                if dist <= r2:
                    Adj[i, j] = 1
                    Adj[j, i] = 1
    return Adj

def graph_average_weights(w):
    
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def updated_weights(w0, w):
    """
    Returns the average of the weights.
    """
    w0c = copy.deepcopy(w0)
    wc = copy.deepcopy(w)
    z.type(dtype=torch.uint8)
    w_avg = copy.deepcopy(z[0])
    for key in w_avg.keys():
        for i in range(1, len(z)):
            w_avg[key] += z[i][key]
        w_avg[key] = torch.div(w_avg[key], len(z))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

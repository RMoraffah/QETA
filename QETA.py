"""
Implements QETA attack
"""
import json
import os
import sys

import random
sys.path.append(os.getcwd())
print(os.getcwd())
import argparse
from types import SimpleNamespace

import glob
import numpy as np
import torch

from config import IMAGE_SIZE, IN_CHANNELS, CLASS_NUM, MODELS_TEST_STANDARD, PY_ROOT, MODELS_SURROGATE, MODELS_TRAIN_STANDARD
from dataset.dataset_loader_maker import DataLoaderMaker
from dataset.defensive_model import DefensiveModel
#from simba_attack.utils import *
from torch.nn import functional as F
import glog as log
import gfcs_util
import eval_sets

import torchvision.models as models

from dataset.standard_model import StandardModel
from meta_attack.meta_training.load_attacked_and_meta_model import load_meta_model
#from meta_attack.attacks.gradient_generator import GradientGenerator
from gfcs_util import GradientGenerator
from meta_attack.attacks.helpers import *
import copy
from torch import optim
from collections import OrderedDict, defaultdict

class SFTF(object):
    def __init__(self, dataset, batch_size, pixel_attack, freq_dims, rgf, order,
                 max_iters, targeted, target_type, norm, l2_bound, linf_bound, net_specific_resampling, GFCS, ODS, step_size, data_index_set, fine_tune = False, stats_grad_cosine_similarity = False, lower_bound=0.0, upper_bound=1.0):
        """
            :param pixel_epsilon: perturbation limit according to lp-ball
            :param norm: norm for the lp-ball constraint
            :param lower_bound: minimum value data point can take in any coordinate
            :param upper_bound: maximum value data point can take in any coordinate
            :param max_crit_queries: max number of calls to early stopping criterion  per data poinr
        """
        assert norm in ['linf', 'l2'], "{} is not supported".format(norm)
        ################### Attack Info ########################
        self.GFCS = GFCS
        self.ODS = True if self.GFCS else ODS
        #self.pixel_epsilon = pixel_epsilon
        self.step_size = step_size
        self.dataset = dataset
        self.norm = norm
        #self.pixel_attack = pixel_attack
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        #self.freq_dims = freq_dims
        #self.stride = stride
        #self.order = order
        self.linf_bound = linf_bound
        self.l2_bound = l2_bound
        # self.early_stop_crit_fct = lambda model, x, y: 1 - model(x).max(1)[1].eq(y)
        self.max_iters = max_iters
        self.targeted = targeted
        self.target_type = target_type
        self.surrogate_model_list = self.prep_surrogate(dataset, net_specific_resampling)
        self.surrogate_model_list_temp = self.prep_surrogate_temp(dataset, net_specific_resampling)
        self.loss_func = torch.nn.functional.cross_entropy if self.targeted else gfcs_util.margin_loss
        self.loss_fn_cosine =  self.xent_loss if self.targeted else self.cw_loss#self.cw_loss if args.loss == "cw" else self.xent_loss

        self.using_ods = True if args.ODS and not args.GFCS else False
        self.fine_tune = fine_tune
        #self.update_pixels = update_pixels
        self.rgf = rgf
        self.stats_grad_cosine_similarity = stats_grad_cosine_similarity
        ################## Attack specific stats #################
        '''
        if args.GFCS:
            ######### TODO : find a better way to model
            self.grad_fail_queries = []
            self.grad_succ_queries = []
            self.ods_fail_queries = []
            self.ods_succ_queries = []
        '''


        ################### Dataset info ####################
        self.data_loader = DataLoaderMaker.get_test_attacked_data(dataset, batch_size, data_index_set)
        #print(len(self.data_loader))
        self.total_images = len(self.data_loader.dataset)
        self.image_height = IMAGE_SIZE[dataset][0]
        self.image_width = IMAGE_SIZE[dataset][1]
        self.in_channels = IN_CHANNELS[dataset]

        ################### Attacks stats ##################
        self.query_all = torch.zeros(self.total_images)
        self.correct_all = torch.zeros_like(self.query_all)  # number of images
        self.success_all = torch.zeros_like(self.query_all)
        self.success_query_all = torch.zeros_like(self.query_all)
        self.fine_tuning_all = torch.zeros_like(self.query_all)
        self.cosine_similarity_all = defaultdict(OrderedDict)    # key is image index, value is {query: cosine_similarity} TODO: make it (query,fine-tuning)
        self.cosine_similarity_fine_tune_all = defaultdict(OrderedDict)    # key is image index, value is {finetune: cosine_similarity} TODO: make it (query,fine-tuning)

    def xent_loss(self, logit, label, target=None):
        #if target is not None:
        #    return -F.cross_entropy(logit, target, reduction='none')
        #else:
        return F.cross_entropy(logit, label, reduction='none')

    def cw_loss(self, logit, label, target=None):
        #if target is not None:
            # targeted cw loss: logit_t - max_{i\neq t}logit_i
        #    _, argsort = logit.sort(dim=1, descending=True)
        #    target_is_max = argsort[:, 0].eq(target).long()
        #    second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max).long() * argsort[:, 0]
        #    target_logit = logit[torch.arange(logit.shape[0]), target]
        #    second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        #    return target_logit - second_max_logit
        #else:
            # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        ### C&W is only used for untargeted
        _, argsort = logit.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(label).long()
        second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max).long() * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit

    def prep_surrogate(self, dataset, net_specific_resampling):
        ################# Surrogate will be a pretrained meta network of gradients ####################

        ############################ TODO : First version : Just change the surrogate #############################
        ############################ TODO: Second version : add finetuning to ODS ################################
        ############################ TODO: Third version : keep different versions of surrogate in a list ################################
        meta_model_path = '{}/train_pytorch_model/meta_grad_regression/{}.pth.tar'.format(PY_ROOT, args.dataset)
        assert os.path.exists(meta_model_path), "{} does not exist!".format(meta_model_path)
        meta_model = load_meta_model(meta_model_path)
        models = []
        models.append(meta_model)
        return models

    def prep_surrogate_temp(self, dataset, net_specific_resampling):
        archs = []
        model_path_list = []
        if dataset == "CIFAR-10" or dataset == "CIFAR-100":
            for arch in MODELS_SURROGATE[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(
                    PY_ROOT, dataset, arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                    model_path_list.append(test_model_path)
                else:
                    log.info(test_model_path + " does not exist!")
        elif dataset == "TinyImageNet":
            # for arch in ["vgg11_bn","resnet18","vgg16_bn","resnext64_4","densenet121"]:
            for arch in MODELS_TRAIN_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}@{}@*.pth.tar".format(
                    PY_ROOT, dataset, arch)
                test_model_path = list(glob.glob(test_model_path))[0]
                if os.path.exists(test_model_path):
                    archs.append(arch)
                    model_path_list.append(test_model_path)
                else:
                    log.info(test_model_path + "does not exist!")

        else:
            ################# Image net #################
            for arch in MODELS_TRAIN_STANDARD[args.dataset]:  # ["inceptionv3","inceptionv4", "inceptionresnetv2","resnet101", "resnet152"]:
                archs.append(arch)

        models = []
        #print("begin construct model")
        for arch in archs:
                model = StandardModel(dataset, arch, no_grad=False)
                model.cuda()
                model.eval()
                models.append(model)
        #print("end construct model")
        return models
    def get_grad(self, model, loss_fn, x, true_labels, target_labels):
        with torch.enable_grad():
            x.requires_grad_()
            logits = model(x)
            loss = loss_fn(logits, true_labels, target_labels).mean()
            gradient = torch.autograd.grad(loss, x, torch.ones_like(loss), retain_graph=False, create_graph=False)[0].detach()
        return gradient


    def get_cos_similarity(self, grad_a, grad_b):
        grad_a = grad_a.view(grad_a.size(0),-1)
        grad_b = grad_b.view(grad_b.size(0),-1)
        cos_similarity = (grad_a * grad_b).sum(dim=1) / torch.sqrt((grad_a * grad_a).sum(dim=1) * (grad_b * grad_b).sum(dim=1))
        assert cos_similarity.size(0) == grad_a.size(0)
        return cos_similarity
    #  argument labels is the target labels or true labels
    # For now let's make the batch_size = 1, we'll change it
    def attack_batch_images(self, model, images, labels_tgt, labels_gt = None, batch_idx = 0):
        #### Logging tensors
        success_list = []
        l2_list = []
        linf_list = []
        queries_list = []
        cosine_similarity_iter = []
        fine_tuning_list = []

        if self.targeted:
            target_class_list = []

        if self.GFCS:
            grad_fail_queries = []
            grad_succ_queries = []
            ods_fail_queries = []
            ods_succ_queries = []

        batch_size = images.size(0) ## For now it's one
        image_size = images.size(2)
        max_iters = self.max_iters
        logits = model(images).data
        to_attack = (torch.argmax(logits, dim=1) != labels_tgt) if self.targeted else (
                torch.argmax(logits, dim=1) == labels_tgt)

        if to_attack:
            X_best = images.clone()
            if self.targeted:
                loss_best = -self.loss_func(logits, labels_tgt)
                class_org = labels_gt[0].item()#label[0].item()
                class_tgt = labels_tgt[0].item()#label_attacked[0].item()
            else:
                loss_best, class_org, class_tgt = self.loss_func(logits.data, labels_tgt)
            nQuery = 1  # query for the original image
            Fine_tuning_count = 0



            if self.GFCS:
                n_grad_fail_queries = 0
                n_grad_succ_queries = 0
                n_ods_fail_queries = 0
                n_ods_succ_queries = 0
                using_ods = False
                surrogate_ind_list = torch.randperm(len(self.surrogate_model_list))

            for m in range(max_iters):
                '''
                if self.stats_grad_cosine_similarity:
                    no_grad = model.no_grad
                    model.no_grad = False
                    true_grad = self.get_grad(model, self.loss_fn_cosine, X_best.detach().clone(), labels_tgt, labels_tgt)
                    #self.get_grad(model, self.loss_func, X_best.detach().clone().requires_grad_(), labels_tgt, labels_tgt)
                    surrogate_gradients = self.surrogate_model_list[0](X_best.detach())
                    cosine_similarity = self.get_cos_similarity(surrogate_gradients, true_grad)
                    model.no_grad = no_grad
                    cosine_similarity_iter.append(cosine_similarity.cpu().detach().numpy())
                '''
                #print("***************************")
                #print(m)
                #print("***************************")

                if self.ODS:
                    X_grad = X_best.detach().clone().requires_grad_()
                    if self.GFCS:
                        ### 1000 is the class number
                        #num = CLASS_NUM[self.dataset]
                        random_direction = torch.zeros(1, CLASS_NUM[self.dataset]).cuda()
                        random_direction[0, class_org] = -1
                        random_direction[0, class_tgt] = 1
                        if surrogate_ind_list.numel() > 0:
                            ind = surrogate_ind_list[0]
                            surrogate_ind_list = surrogate_ind_list[1:]
                        else:  # You're stuck, so time to revert.
                            random_direction = torch.rand((1, CLASS_NUM[self.dataset])).cuda() * 2 - 1
                            ind = np.random.randint(len(self.surrogate_model_list_temp))
                            using_ods = True
                    else:
                        random_direction = torch.rand((1, CLASS_NUM[self.dataset])).cuda() * 2 - 1
                        ind = np.random.randint(len(self.surrogate_model_list))

                    with torch.enable_grad():
                        if not using_ods:
                            # Get gradient from the meta model --> Surrogate
                            ind1=0
                            grad = self.surrogate_model_list[ind1](X_grad)
                        #elif using_ods and self.rgf:
                            ############ Do ODS on the original model ############
                            normalized_random_direction = random_direction / random_direction.norm()
                            ########### Calculate grad via rgf

                            ########### normalized the grad

                            ########### Multiply get the direction

                            ##########update the query count
                        else:
                            # Calculate ODS on a randomly selected surrogate, later on finetune the meta gradient model
                            #with torch.enable_grad():
                            #    loss = -self.loss_func(self.surrogate_model_list_temp[ind](X_grad),
                            #                           labels_tgt) if self.targeted else (
                            #            self.surrogate_model_list_temp[ind](X_grad) * random_direction).sum()
                            loss = (self.surrogate_model_list_temp[ind](X_grad) * random_direction).sum()
                            loss.backward()
                            grad = X_grad.grad
                            ####### Or we can make it a PRGF style

                        '''
                        ### No need to calculate the loss and then backward, we directly have access to gradients!
                        if self.targeted and not using_ods:
                            # Then you want the target-label x-ent loss from the surrogate:
                            loss = -self.loss_func(self.surrogate_model_list[ind](X_grad), labels_tgt)
                        else:  # Either margin loss gradient or ODS direction, depending on above context.
                            loss = (self.surrogate_model_list[ind](X_grad) * random_direction).sum()
                    loss.backward()
                    '''
                    ################## We are doing l2- norm ##################
                    #delta = X_grad.grad / X_grad.grad.norm()
                    delta = grad / grad.norm()

                    '''
                else:  # If you're using neither GFCS nor ODS, it falls back to pixel SimBA.
                    ind1 = np.random.randint(3)
                    ## TODO: change
                    image_width = 299 if args.smodel_name[s] == 'inception_v3' else 224

                    ind2 = np.random.randint(image_width)
                    ind3 = np.random.randint(image_width)
                    delta = torch.zeros(X_best.shape).cuda()
                    delta[0, ind1, ind2, ind3] = 1
                    '''
                for sign in [1, -1]:
                    X_pert = X_best - images + (self.step_size * sign * delta)
                    ######### TODO: Generalize the bound so that it's not only l2
                    if X_pert.norm() > self.l2_bound:
                        X_pert = X_pert / X_pert.norm() * self.l2_bound
                    X_new = images + X_pert

                    X_new = torch.clamp(X_new, 0, 1)
                    logits = model(X_new).data
                    nQuery += 1


                    if self.targeted:
                        loss_new = -self.loss_func(logits.data, labels_tgt)
                        class_tgt_new = class_tgt  # The target is actually fixed: this is a dummy variable.
                        class_org_new = torch.argmax(logits,
                                                     dim=1)  # The top finisher can actually change, in a targeted
                        #   attack, but using the x-ent loss on the target class alone, this won't actually matter.
                    else:
                        loss_new, class_org_new, class_tgt_new = self.loss_func(logits.data, labels_tgt)

                    '''
                    ## TODO : check!!!
                    ## If you reach here via ods then use the direction to improve the surrogare gradients via finetuning
                    if using_ods:
                        meta_optimizer = optim.Adam(self.surrogate_model_list.parameters(), lr=0.01)
                        input_adv_copy = copy.deepcopy(X_best_copy.detach())
                        zoo_gradients = []
                        generate_grad = GradientGenerator(update_pixels=self.update_pixels,
                                                          targeted=self.targeted, classes=CLASS_NUM[self.dataset])
                        indice = torch.abs(logits).cpu().numpy().reshape(-1).argsort()[-500:]
                        zoo_grad, select_indice = generate_grad.run(model, input_adv_copy, class_org_new,
                                                                    indice)  # query for batch_size times
                        B, nc, w, h = X_new.shape
                        nQuery += nc * w * h#self.update_pixels
                        zoo_gradients.append(zoo_grad)
                        zoo_gradients = np.array(zoo_gradients, np.float32)
                        zoo_gradients = torch.from_numpy(zoo_gradients).cuda()

                        std = zoo_gradients.cpu().numpy().std(axis=(1, 2, 3))
                        std = std.reshape((-1, 1, 1, 1)) + 1e-23
                        zoo_gradients = zoo_gradients / torch.from_numpy(std).cuda()
                        assert not torch.isnan(zoo_gradients.sum())
                        ############## Fine-tune the surrogate
                        for i in range(20):
                            ############################### Fine-tuning the meta model ##################################
                            meta_optimizer.zero_grad()
                            meta_grads = self.surrogate_model_list[ind](input_adv_copy)
                            meta_loss = F.mse_loss(meta_grads.reshape(-1),
                                                   zoo_gradients.reshape(-1))
                            meta_loss.backward()
                            meta_optimizer.step()
                    '''
                    if loss_best < loss_new:
                        X_best_copy = copy.deepcopy(X_best.detach())
                        X_best = X_new
                        loss_best = loss_new
                        class_org = class_org_new
                        class_tgt = class_tgt_new
                        if self.stats_grad_cosine_similarity:
                            #### Successful perturbation -> calculate cosine similarity
                            no_grad = model.no_grad
                            model.no_grad = False
                            true_grad = self.get_grad(model, self.loss_fn_cosine, X_best_copy.detach().clone(), labels_tgt,
                                                      labels_tgt)
                            # only sign matters
                            true_grad = true_grad / true_grad.norm()
                            # self.get_grad(model, self.loss_func, X_best.detach().clone().requires_grad_(), labels_tgt, labels_tgt)
                            surrogate_gradients = X_pert / X_pert.norm()#self.surrogate_model_list[0](X_best.detach())
                            cosine_similarity = self.get_cos_similarity(surrogate_gradients, true_grad)
                            model.no_grad = no_grad
                            cosine_similarity_iter.append(cosine_similarity.cpu().detach().numpy())
                        if self.GFCS:
                            if using_ods:
                                n_ods_succ_queries += 1
                            else:
                                n_grad_succ_queries += 1

                            ################ It might be a good sample, worth fine-tuning our meta gradient model on it!
                            if using_ods and self.fine_tune:

                                meta_optimizer = optim.Adam(self.surrogate_model_list[ind1].parameters(), lr=0.01)
                                input_adv_copy = copy.deepcopy(X_best_copy.detach())
                                input_adv_copy.requires_grad_()
                                zoo_gradients = []
                                #generate_grad = GradientGenerator(update_pixels=self.update_pixels,
                                #                                  targeted=self.targeted,
                                #                                  classes=CLASS_NUM[self.dataset])
                                #meta_output = self.surrogate_model_list[ind](X_best_copy)
                                #indice = torch.abs(meta_output.data).cpu().numpy().reshape(-1).argsort()[
                                #         -self.update_pixels:]
                                #indice2 = indice
                                #zoo_grad, select_indice = generate_grad.run(model, input_adv_copy, labels_tgt,
                                #                                            indice)  # query for batch_size times
                                if self.rgf:
                                    ########### Use rgf to approximate the gradient
                                    l = loss_new#self.loss_func(logits_real_images, true_labels, target_labels)
                                    q = 50# No of directions used to approximate the gradient (fixed here)
                                    sigma = 1e-4
                                    pert = torch.randn(size=(
                                    q, input_adv_copy.size(-3), input_adv_copy.size(-2), input_adv_copy.size(-1)))  # q,C,H,W
                                    pert = pert.cuda()
                                    for i in range(q):
                                        pert[i] = pert[i] / torch.clamp(
                                                torch.sqrt(torch.mean(torch.mul(pert[i], pert[i]))), min=1e-12)
                                    while True:
                                        eval_points = input_adv_copy + sigma * pert  # (1,C,H,W)  pert=(q,C,H,W)
                                        logits_ = model(eval_points)
                                        target_labels_q = None
                                        #if self.targeted is not None:
                                        labels_tgt_q = labels_tgt.repeat(q)
                                        if self.targeted:
                                            losses = -self.loss_func(logits_.data, labels_tgt_q)
                                        else:
                                            losses, _, _ = self.loss_func(logits_.data, labels_tgt_q)
                                        #losses = self.loss_func()#self.xent_loss(logits_, true_labels.repeat(q),
                                                 #               target_labels_q)  # shape = (q,)
                                        nQuery += q
                                        grad = (losses - l).view(-1, 1, 1, 1) * pert  # (q,1,1,1) * (q,C,H,W)
                                        grad = torch.mean(grad, dim=0, keepdim=True)  # 1,C,H,W
                                        norm_grad = torch.sqrt(torch.mean(torch.mul(grad, grad)))
                                        if norm_grad.item() == 0:
                                            sigma *= 5
                                            log.info(
                                                "estimated grad == 0, multiply sigma by 5. Now sigma={:.4f}".format(
                                                    sigma))
                                        else:
                                            break
                                    #grad = grad / torch.clamp(torch.sqrt(torch.mean(torch.mul(grad, grad))), min=1e-12)
                                    #grad =
                                    #nQuery += #self.update_pixels
                                else:
                                    ##### Just use the gradient from the surrogate
                                    #loss_grad = (self.surrogate_model_list_temp[ind](input_adv_copy)).sum()
                                    #loss_grad.backward()
                                    #grad = X_grad.grad
                                    with torch.enable_grad():
                                        loss_grad = -self.loss_func(self.surrogate_model_list_temp[ind](input_adv_copy),
                                                               labels_tgt) if self.targeted else (
                                                self.surrogate_model_list_temp[ind](input_adv_copy) * random_direction).sum()
                                    # loss = (self.surrogate_model_list_temp[ind](X_grad) * random_direction).sum()
                                    loss_grad.backward()
                                    grad = input_adv_copy.grad
                                #print(grad.shape)
                                #print(grad)

                                zoo_gradients.append(grad.cpu().numpy())
                                zoo_gradients = np.array(zoo_gradients, np.float32)
                                zoo_gradients = torch.from_numpy(zoo_gradients).cuda()

                                #std = zoo_gradients.cpu().numpy().std(axis=(1, 2, 3))
                                #std = std.reshape((-1, 1, 1, 1)) + 1e-23
                                #zoo_gradients = zoo_gradients / torch.from_numpy(std).cuda()
                                assert not torch.isnan(zoo_gradients.sum())
                                ############## Fine-tune the surrogate
                                for i in range(20):
                                    ############################### Fine-tuning the meta model ##################################
                                    meta_optimizer.zero_grad()
                                    meta_grads = self.surrogate_model_list[ind1](input_adv_copy)
                                    meta_loss = F.mse_loss(meta_grads.reshape(-1),
                                                           zoo_gradients.reshape(-1))
                                    meta_loss.backward()
                                    meta_optimizer.step()
                                Fine_tuning_count = Fine_tuning_count + 1
                        # On optimisation success, reset the surrogate list and ensure that you go back to gradients.
                        surrogate_ind_list = torch.randperm(len(self.surrogate_model_list))
                        using_ods = False

                        break
                    # If you reach here, this attempt didn't work, so we count fail queries:
                    if self.GFCS:
                        if using_ods:
                            n_ods_fail_queries += 1
                        else:
                            n_grad_fail_queries += 1



                success = (torch.argmax(logits, dim=1) == labels_tgt) if self.targeted else (
                        torch.argmax(logits, dim=1) != labels_tgt)

                if success:
                    print('image %d: attack is successful. query = %d, dist = %.4f' % (
                        batch_idx + 1, nQuery, (X_best - images).norm()))
                    if self.GFCS:
                        print(f"grad success queries: {n_grad_succ_queries}, grad fail queries: {n_grad_fail_queries}, "
                              f"ODS success queries: {n_ods_succ_queries}, ODS fail queries: {n_ods_fail_queries}")
                    break

                if m == self.max_iters - 1:
                    print('image %d: attack is not successful (query = %d)' % (batch_idx + 1, nQuery))
                    if self.GFCS:
                        print(f"grad success queries: {n_grad_succ_queries}, grad fail queries: {n_grad_fail_queries}, "
                              f"ODS success queries: {n_ods_succ_queries}, ODS fail queries: {n_ods_fail_queries}")

            success_list.append(success.item())
            queries_list.append(nQuery)
            l2_list.append((X_best - images).norm(p=2).item())
            linf_list.append((X_best - images).norm(p=np.inf).item())
            fine_tuning_list.append(Fine_tuning_count)

            if self.GFCS:
                grad_fail_queries.append(n_grad_fail_queries)
                grad_succ_queries.append(n_grad_succ_queries)
                ods_fail_queries.append(n_ods_fail_queries)
                ods_succ_queries.append(n_ods_succ_queries)
            if self.targeted:
                target_class_list.append(labels_tgt[0].item())

        else:
            print('image %d: already adversary' % (batch_idx + 1))
            X_new = images
            success_list.append(False) ## dummy
            queries_list.append(10000) ## dumy
            l2_list.append(10000)  ## dummy
            linf_list.append(1000) ## dummy
            cosine_similarity_iter.append([-1000]) ## dummy
            fine_tuning_list.append([0]) ## dummy
        if not self.stats_grad_cosine_similarity:
            cosine_similarity_iter.append([-1000])  #dummy

        if len(cosine_similarity_iter) == 0:
            cosine_similarity_iter.append([0])
        #tmp0 = cosine_similarity_iter
        #print("#################### temps #####################")
        #print(tmp0)


        #tmp00 = np.asarray(cosine_similarity_iter).mean(axis=0)
        #print("###############################################")
        #print(tmp00)
        #tmp1 = torch.FloatTensor(np.asarray(cosine_similarity_iter).mean(axis=0))
        #print("###############################################")
        #print(tmp1)
        #tmp2 = torch.FloatTensor(fine_tuning_list)
        #print("###############################################")
        #print(tmp2)

        return X_new, torch.FloatTensor(success_list), torch.FloatTensor(queries_list)\
            , torch.FloatTensor(l2_list), torch.FloatTensor(linf_list),\
               torch.FloatTensor(np.asarray(cosine_similarity_iter).mean(axis=0)), torch.FloatTensor(fine_tuning_list)

    def normalize(self, t):
        assert len(t.shape) == 4
        norm_vec = torch.sqrt(t.pow(2).sum(dim=[1, 2, 3])).view(-1, 1, 1, 1)
        norm_vec += (norm_vec == 0).float() * 1e-8
        return norm_vec

    def attack_all_images(self, args, model, result_dump_path):

        for batch_idx, data_tuple in enumerate(self.data_loader):
            if args.dataset == "ImageNet":
                if model.input_size[-1] >= 299:
                    images, true_labels = data_tuple[1], data_tuple[2]
                else:
                    images, true_labels = data_tuple[0], data_tuple[1]
            else:
                images, true_labels = data_tuple[0], data_tuple[1]
            ########## Simba related ########
            '''
            if model.input_size[-1] == 299:
                self.freq_dims = 33
                self.stride = 7
            elif model.input_size[-1] == 331:
                self.freq_dims = 30
                self.stride = 7
            '''
            ############################# Interpolate if the sizes don't match! #######################
            if images.size(-1) != model.input_size[-1]:
                self.image_width = model.input_size[-1]
                self.image_height = model.input_size[-1]
                images = F.interpolate(images, size=model.input_size[-1], mode='bilinear',align_corners=True)

            ############################# If targeted select the target! ########################
            if self.targeted:
                if self.target_type == 'random':
                    target_labels = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                  size=true_labels.size()).long().cuda()
                    invalid_target_index = target_labels.eq(true_labels)
                    while invalid_target_index.sum().item() > 0:
                        target_labels[invalid_target_index] = torch.randint(low=0, high=CLASS_NUM[self.dataset],
                                                               size=target_labels[invalid_target_index].shape).long().cuda()
                        invalid_target_index = target_labels.eq(true_labels)
                elif args.target_type == 'least_likely':
                    target_labels = logit.argmin(dim=1)
                elif self.target_type == "increment":
                    target_labels = torch.fmod(true_labels + 1, CLASS_NUM[self.dataset])
                else:
                    raise NotImplementedError('Unknown target_type: {}'.format(self.target_type))
            else:
                target_labels = None

            ##### locate images in the .all arrays
            selected = torch.arange(batch_idx * args.batch_size,
                                    min((batch_idx + 1) * args.batch_size, self.total_images))
            images = images.cuda()
            true_labels = true_labels.cuda()
            if self.targeted:
                target_labels = target_labels.cuda()

            ######################### Only attack correctly labeled ###################
            with torch.no_grad():
                logit = model(images)
            pred = logit.argmax(dim=1)
            correct = pred.eq(true_labels).float().detach().cpu()

            ######################## Perform the attack ##################
            if self.targeted:
                adv_images, success, query, l2, linf, cosine_similarity, fine_tuning = self.attack_batch_images(model, images.cuda(), target_labels.cuda(), true_labels.cuda(), batch_idx)
            else:
                adv_images, success, query, l2, linf, cosine_similarity, fine_tuning = self.attack_batch_images(model, images.cuda(), true_labels.cuda(), batch_idx = batch_idx)
            delta = adv_images.view_as(images) - images
            # if self.norm == "l2":
            #     l2_out_bounds_mask = (self.normalize(delta) > self.l2_bound).long().view(-1).detach().cpu().numpy()  # epsilon of L2 norm attack = 4.6
            #     l2_out_bounds_indexes = np.where(l2_out_bounds_mask == 1)[0]
            #     if len(l2_out_bounds_indexes) > 0:
            #         success[l2_out_bounds_indexes] = 0

            ##################### Prune the unwanted samples like the ones the exceed query budget or labeled incorrectly from the beginning ##########
            out_of_bound_indexes = np.where(query.detach().cpu().numpy() > args.max_queries)[0]
            if len(out_of_bound_indexes) > 0:
                success[out_of_bound_indexes] = 0
            log.info("{}-th batch attack over, avg. query:{}".format(batch_idx, query.mean().item()))
            ################## Build the final result #########################
            success = success * correct
            success_query = success * query

            for key in ['query', 'correct',
                        'success', 'success_query', 'fine_tuning']:
                value_all = getattr(self, key + "_all")
                value = eval(key)
                value_all[selected] = value.detach().float().cpu()
            if self.stats_grad_cosine_similarity:
                self.cosine_similarity_all[selected.item()][int(query.detach().float().cpu())] = cosine_similarity.item()
                self.cosine_similarity_fine_tune_all[selected.item()][int(fine_tuning.detach().float().cpu())] = cosine_similarity.item()

        ##################### log the results ###########################
        #is_all_zero = np.all((self.fine_tuning_all.detach().cpu().numpy().astype(np.int32) == 0))
        log.info('Saving results to {}'.format(result_dump_path))
        meta_info_dict = {"avg_correct": self.correct_all.mean().item(),
                          "mean_query": self.success_query_all[self.success_all.byte()].mean().item(),
                          "avg_not_done": 1.0 - self.success_all[self.correct_all.byte()].mean().item(),
                          "median_query": self.success_query_all[self.success_all.byte()].median().item(),
                          "max_query": self.success_query_all[self.success_all.byte()].max().item(),
                          "not_done_all": (1 - self.success_all.detach().cpu().numpy().astype(np.int32)).tolist(),
                          "correct_all": self.correct_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "query_all": self.query_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "fine_tuning_all": self.fine_tuning_all.detach().cpu().numpy().astype(np.int32).tolist(),
                          "args": vars(args)}
        if self.stats_grad_cosine_similarity:
            meta_info_dict['grad_cosine_similarities'] = self.cosine_similarity_all#.detach().cpu().numpy().astype(np.int32).tolist()
            meta_info_dict['grad_cosine_finetuning'] = self.cosine_similarity_fine_tune_all#.detach().cpu().numpy().astype(np.int32).tolist()

            N = 0
            sum_cosine_similarity = 0.0
            for image_index, cos_dict in self.cosine_similarity_all.items():
                for q, cosine in cos_dict.items():
                    sum_cosine_similarity += abs(cosine)
                    N += 1
            avg_cosine_similarity = sum_cosine_similarity / N
            meta_info_dict["avg_cosine_similarity"] = avg_cosine_similarity#.detach().cpu().numpy().astype(np.int32).tolist()
        with open(result_dump_path, "w") as result_file_obj:
            json.dump(meta_info_dict, result_file_obj, sort_keys=True)
        log.info("done, write experimental result information to {}".format(result_dump_path))



def set_log_file(fname):
    import subprocess
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def get_exp_dir_name(dataset, GFCS, ODS, norm, norm_bound, targeted, target_type, args):
    target_str = "untargeted" if not targeted else "targeted_{}".format(target_type)
    attack_str = "SFTF"
    if ODS and not GFCS:
        attack_str = "SimBA_pixel_attack"

    if args.attack_defense:
        dirname = '{}_on_defensive_model-{}-{}-{}-{}-{}'.format(attack_str, dataset,  norm, norm_bound, target_str)
    else:
        dirname = '{}-{}-{}-{}-{}'.format(attack_str, dataset, norm, norm_bound, target_str)
    if not args.fine_tune:
        dirname += "Fine_tune-False"
    if args.cosine_grad:
        dirname += "_grad_cosine_stats4"
    return dirname

def print_args(args):
    keys = sorted(vars(args).keys())
    max_len = max([len(key) for key in keys])
    for key in keys:
        prefix = ' ' * (max_len + 1 - len(key)) + key
        log.info('{:s}: {}'.format(prefix, args.__getattribute__(key)))

def get_parse_args():
    parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for parallel runs')
    #parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')  ### num_step
    parser.add_argument('--max_queries',type=int,default=10000)
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    #parser.add_argument('--pixel_epsilon', type=float, default=0.2,  help='step size per pixel')
    parser.add_argument('--linf_bound', type=float,  help='L_inf epsilon bound for L2 norm attack, this option cannot be used with --pixel_attack together')
    parser.add_argument('--l2_bound', type=float, help='L_2 epsilon bound for L2 norm attack')
    parser.add_argument('--freq_dims', type=int, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='strided', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, help='stride for block order')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    parser.add_argument('--json-config', type=str,
                        default='/home/local/ASUAD/rmoraffa/Desktop/cikm2022/simulator/configures/SFTF_attack_conf.json',
                        help='a configures file to be passed in instead of arguments')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['CIFAR-10', 'CIFAR-100', 'ImageNet', "FashionMNIST", "MNIST", "TinyImageNet"],
                        help='which dataset to use')
    parser.add_argument('--exp-dir', default='logs', type=str,
                        help='directory to save results and logs')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--norm', type=str, required=True, help='Which lp constraint to run bandits [linf|l2]')
    parser.add_argument('--arch', default="WRN-28-10-drop", type=str, help='network architecture')
    parser.add_argument('--test_archs', action="store_true")
    parser.add_argument('--targeted', action="store_true")
    parser.add_argument('--target_type', type=str, default='increment', choices=['random', 'least_likely', "increment"])
    parser.add_argument('--attack_defense', action="store_true")
    parser.add_argument('--defense_model', type=str, default=None)
    parser.add_argument('--update_pixels', default = 125, type = int, help = 'updated pixels every iteration')


    ########################################### SFTF specific arguments #########################################
    #parser.add_argument('--device', default='cuda:0', help='Device for evaluating networks.')
    #parser.add_argument('--model_name', type=str, required=True, help='Target model to use.')
    #parser.add_argument('--smodel_name', type=str, nargs='+',
    #                    help='One or more surrogate models to use (enter all names, separated by spaces).')
    #parser.add_argument('--targeted', action='store_true', help='If true, perform targeted attack; else, untargeted.')
    parser.add_argument('--ODS', action='store_true', help='Perform ODS (original SimBA-ODS).')
    parser.add_argument('--GFCS', action='store_true', help='Activate GFCS method.')
    parser.add_argument('--num_step', type=int, default=10000,
                        help="Number of 'outer' SimBA iterations. Note that each "
                             "iteration may consume 1 or 2 queries.")
    #parser.add_argument('--num_sample', default=10, type=int, help='Number of sample images to attack.')
    parser.add_argument('--data_index_set', type=str,
                        choices=['vgg16_bn_mstr', 'vgg16_bn_batch0', 'vgg16_bn_batch1', 'vgg16_bn_batch2',
                                 'vgg16_bn_batch3', 'vgg16_bn_batch4', 'vgg16_bn_batch0_2', 'vgg16_bn_batch3_4',
                                 'resnet50_mstr', 'resnet50_batch0', 'resnet50_batch1', 'resnet50_batch2',
                                 'resnet50_batch3', 'resnet50_batch4', 'resnet50_batch0_2', 'resnet50_batch3_4',
                                 'inceptionv3_mstr', 'inceptionv3_batch0', 'inceptionv3_batch1', 'inceptionv3_batch2',
                                 'inceptionv3_batch3', 'inceptionv3_batch4', 'inceptionv3_batch0_2',
                                 'inceptionv3_batch3_4',
                                 'imagenet_val_random'],
                        default='None',
                        help='The indices from the ImageNet val set to use as inputs. Most options represent predefined '
                             'randomly sampled batches. imagenet_val_random samples from the val set randomly, and may not '
                             'necessarily give images that are correctly classified by the target net.')
    parser.add_argument('--step_size', default=2.0, type=float, help='Optimiser step size (as in SimBA).')
    parser.add_argument('--fine_tune', action="store_true")
    parser.add_argument('--rgf', action="store_true")
    parser.add_argument('--cosine_grad',action='store_true',help='record the cosine similarity of gradient')

    #parser.add_argument('--output', required=True, help='Name of the output file.')
    #parser.add_argument('--norm_bound', type=float, default=float('inf'),
    #                    help='Radius of l2 norm ball onto which solution will be maintained through PGD-type optimisation. '
    #                         'If not supplied, is effectively infinite (norm is unconstrained).')
    parser.add_argument('--net_specific_resampling', action='store_true',
                        help='If specified, resizes input images to match expectations of target net (as always), but adds '
                             'a linear interpolation step to each surrogate network to match its expected resolution. '
                             'Gradients are thus effectively computed in the native surrogate resolutions and returned to '
                             'the target net''s own resolution via the reverse interpolation.')


    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    return args

if __name__ == "__main__":

    ############################### Parse the arguments #################################
    args = get_parse_args()
    args_dict = None
    if args.json_config:
        # If a json file is given, use the JSON file as the base, and then update it with args
        defaults = json.load(open(args.json_config))[args.dataset]
        arg_vars = vars(args)
        arg_vars = {k: arg_vars[k] for k in arg_vars if arg_vars[k] is not None}
        defaults.update(arg_vars)
        args = SimpleNamespace(**defaults)
    if args.targeted:
        args.num_step = 100000
        if args.dataset == "ImageNet":
            args.max_queries = 50000
    #if args.norm == "linf":
    #    assert not args.pixel_attack, "L_inf norm attack cannot be used with --pixel_attack together"

    ########################## Make the main experiment directory ############################
    # TODO: make it work for L_infinity too
    args.exp_dir = os.path.join(args.exp_dir,
                            get_exp_dir_name(args.dataset,args.GFCS, args.ODS, args.norm, args.l2_bound, args.targeted, args.target_type, args))  # 随机产生一个目录用于实验
    os.makedirs(args.exp_dir, exist_ok=True)

    ################################ Set up logging directories #############################
    if args.test_archs:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}.log'.format(args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run.log')
    elif args.arch is not None:
        if args.attack_defense:
            log_file_path = os.path.join(args.exp_dir, 'run_defense_{}_{}.log'.format(args.arch, args.defense_model))
        else:
            log_file_path = os.path.join(args.exp_dir, 'run_{}.log'.format(args.arch))
    set_log_file(log_file_path)
    if args.attack_defense:
        assert args.defense_model is not None

    ################################ Set up cuda and seeds #######################################
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ############################## Get test architectures paths ##############################
    if args.test_archs:
        archs = []
        if args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/{}/checkpoint.pth.tar".format(PY_ROOT,
                                                                                        args.dataset,  arch)
                if os.path.exists(test_model_path):
                    archs.append(arch)
                else:
                    log.info(test_model_path + " does not exists!")
        elif args.dataset == "TinyImageNet":
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}*.pth.tar".format(
                    root=PY_ROOT, dataset=args.dataset, arch=arch)
                test_model_path = list(glob.glob(test_model_list_path))
                if test_model_path and os.path.exists(test_model_path[0]):
                    archs.append(arch)
        else:
            for arch in MODELS_TEST_STANDARD[args.dataset]:
                test_model_list_path = "{}/train_pytorch_model/real_image_model/{}-pretrained/checkpoints/{}*.pth".format(
                    PY_ROOT,
                    args.dataset, arch)
                test_model_list_path = list(glob.glob(test_model_list_path))
                if len(test_model_list_path) == 0:  # this arch does not exists in args.dataset
                    continue
                archs.append(arch)
    else:
        assert args.arch is not None
        archs = [args.arch]
    args.arch = ", ".join(archs)
    ######################### Prepare and initialize the attack #########################
    '''
    if args.order == 'rand':
        n_dims = IN_CHANNELS[args.dataset] * args.freq_dims * args.freq_dims
    else:
        n_dims = IN_CHANNELS[args.dataset] * IMAGE_SIZE[args.dataset][0] * IMAGE_SIZE[args.dataset][1]
    if args.num_iters > 0:
        max_iters = int(min(n_dims, args.num_iters))
    else:
        max_iters = int(n_dims)
    '''
    max_iters = args.num_step
    attacker = SFTF(args.dataset, args.batch_size, args.pixel_attack, args.freq_dims, args.rgf, args.order,max_iters,
                     args.targeted,args.target_type, args.norm, args.l2_bound, args.linf_bound,args.net_specific_resampling,
                    args.GFCS, args.ODS, args.step_size,  args.data_index_set, args.fine_tune, args.cosine_grad, 0.0, 1.0)
    log.info('Command line is: {}'.format(' '.join(sys.argv)))
    log.info("Log file is written in {}".format(log_file_path))
    log.info('Called with args:')
    print_args(args)

    ############################## Run the attack for every TARGET architecture ###########################
    for arch in archs:
        ############################ Set up results directory (for different experiments) ##########################
        if args.attack_defense:
            save_result_path = args.exp_dir + "/{}_{}_result.json".format(arch, args.defense_model)
        else:
            save_result_path = args.exp_dir + "/{}_result.json".format(arch)
        if os.path.exists(save_result_path):
            continue
        log.info("Begin attack {} on {}, result will be saved to {}".format(arch, args.dataset, save_result_path))

        ################## Prepare target models, could be defended or vanilla. This can be done in attack_all_images function ######
        if args.attack_defense:
            model = DefensiveModel(args.dataset, arch, no_grad=True, defense_model=args.defense_model)
        else:
            model = StandardModel(args.dataset, arch, no_grad=True)
        model.cuda()
        model.eval()
        attacker.attack_all_images(args, model, save_result_path)
        model.cpu()

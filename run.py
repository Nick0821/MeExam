# @Time   : 2021/7/14
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
session-based recommendation example
========================
Here is the sample code for running session-based recommendation benchmarks using RecBole.

args.dataset can be one of diginetica-session/tmall-session/nowplaying-session
"""
import setproctitle
from DataHandler import DataHandler
from time import time
setproctitle.setproctitle("EXP@yuh")
import pandas as pd
import argparse
from logging import getLogger
import pickle
from recbole.evaluator import Evaluator, Collector
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader, create_samplers
from recbole.utils import init_logger, init_seed, get_model, get_trainer, ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage, WandbLogger, meta_minibatch
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from Params import args
from Diffsion_model import Denoise, GaussianDiffusion
from tqdm import tqdm
from thop import profile
class Coach:
    def __init__(self, handler, Denoise, GaussianDiffusion, config, check_index):
        self.config = config
        self.handler = handler
        self.Denoise = Denoise
        self.GaussianDiffusion = GaussianDiffusion
        self.train_loss_dict = dict()
        self.gpu_available = torch.cuda.is_available() and config['use_gpu']
        self.device = config['device']
        self.check_index = torch.tensor(check_index,device=config['device'])
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        if self.device == "cpu":
            self.gpu_available = False
        self.epochs = config['epochs']
        self.start_epoch = 0
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.eval_step = min(config['eval_step'], self.epochs)
        self.valid_metric = config['valid_metric'].lower()
        self.stopping_step = config['stopping_step']
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.checkpoint_dir = config['checkpoint_dir']
        ensure_dir(self.checkpoint_dir)
        saved_model_file_path = './saved/{}/Best'.format(args.log_name.replace('.log', ''))
        cur_saved_model_file_path = './saved/{}/cur'.format(args.log_name.replace('.log', ''))

        os.makedirs(saved_model_file_path, exist_ok=True)

        os.makedirs(cur_saved_model_file_path, exist_ok=True)

        saved_model_file = saved_model_file_path + '/model.pth'
        self.saved_model_file = os.path.join(saved_model_file)
        self.best_valid_result = None
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.cur_step = 0
    def prepareModel(self):
        self.model = get_model(config['model'])(config, train_data.dataset,self.check_index).to(config['device'])
        logger.info(self.model)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                             gamma=args.gamma)

        self.diffusion_model = self.GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
        out_dims = eval(args.dims) + [args.n_item_all]
        logger.info(self.diffusion_model)
        in_dims = out_dims[::-1]
        for name, param in self.diffusion_model.named_parameters():
            if param.requires_grad:
                print(name)
        self.denoise_model = self.Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
        logger.info(self.denoise_model)
        for name, param in self.denoise_model.named_parameters():
            if param.requires_grad:
                print(name)

        self.denoise_opt = torch.optim.Adam(self.denoise_model.parameters(), lr=args.lr, weight_decay=0)

        if args.test_train_graph:
            self.diffusion_model_test = self.GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max,
                                                          args.steps).cuda()

            self.denoise_model_test = self.Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()

            self.denoise_test_opt = torch.optim.Adam(self.denoise_model_test.parameters(), lr=args.lr, weight_decay=0)


    def _add_hparam_to_tensorboard(self, best_valid_result):
        # base hparam
        hparam_dict = {
            'learner': self.config['learner'],
            'learning_rate': self.config['learning_rate'],
            'train_batch_size': self.config['train_batch_size']
        }
        # unrecorded parameter
        unrecorded_parameter = {
            parameter
            for parameters in self.config.parameters.values() for parameter in parameters
        }.union({'model', 'dataset', 'config_files', 'device'})
        # other model-specific hparam
        hparam_dict.update({
            para: val
            for para, val in self.config.final_config_dict.items() if para not in unrecorded_parameter
        })
        for k in hparam_dict:
            if hparam_dict[k] is not None and not isinstance(hparam_dict[k], (bool, str, float, int)):
                hparam_dict[k] = str(hparam_dict[k])

        self.tensorboard.add_hparams(hparam_dict, {'hparam/best_valid_result': best_valid_result})

    def _add_train_loss_to_tensorboard(self, epoch_idx, Train_losses, DiffsionLoss, tag='Loss/Train'):
        if isinstance(Train_losses, tuple):
            for idx, loss in enumerate(Train_losses):
                self.tensorboard.add_scalar(tag + str(idx), loss, epoch_idx)
        else:
            self.tensorboard.add_scalar(tag, Train_losses, epoch_idx)
            self.tensorboard.add_scalar('Loss/Diffsion', DiffsionLoss, epoch_idx)
    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses,epDiLoss):
        des = self.config['loss_decimal_place'] or 4
        train_loss_output = (set_color('epoch %d training', 'green') + ' [' + set_color('time', 'blue') +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = (set_color('train_loss%d', 'blue') + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += set_color('train loss', 'blue') + ': ' + des % losses + ', '
            train_loss_output += set_color('diffsion loss', 'blue') + ': ' + des % epDiLoss
        return train_loss_output + ']'

    def run(self):
        self.prepareModel()
        logger.info('Model Prepared')
        logger.info('Model Initialized')

        self.eval_collector.data_collect(train_data)
        valid_step = 0
        loss_his = []
        valid_score_his = []
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            reward = self.calc_reward(loss_his, args.eps)
            train_loss, epDiLoss = self.trainEpoch(train_data, epoch_idx, reward)
            logger.info(f'reward = {reward}')
            loss_his.append(train_loss)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss, epDiLoss)
            logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss,epDiLoss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'diffsion_loss': epDiLoss, 'train_step': epoch_idx},
                                         head='train')

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                # reward_test = self.calc_reward(valid_score_his, args.eps)
                reward_test = 1
                valid_score, valid_result = self._valid_epoch(test_data, reward_test)
                logger.info(f'reward_test = {reward_test}')
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                logger.info(valid_score_output)
                logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')
                valid_score_his.append(valid_score)


                cur_saved_model_file = './saved/{}/cur/model.pth'.format(args.log_name.replace('.log', ''))

                os.makedirs('./saved/{}/cur/'.format(args.log_name.replace('.log', '')), exist_ok=True)

                self._save_checkpoint(epoch_idx, cur_saved_model_file)

                torch.save(self.denoise_model.state_dict(),
                           './saved/{}/cur/diffsion_model.pth'.format(args.log_name.replace('.log', '')))

                # torch.save(self.denoise_model_test.state_dict(),
                #            './saved/{}/cur/diffsion_test_model.pth'.format(args.log_name.replace('.log', '')))

                if update_flag:
                    find_better = set_color('Find better model', 'blue')
                    logger.info(find_better)
                    self._save_checkpoint(epoch_idx,self.saved_model_file)

                    torch.save(self.denoise_model.state_dict(), './saved/{}/Best/diffsion_model.pth'.format(args.log_name.replace('.log', '')))

                    # torch.save(self.denoise_model_test.state_dict(), './saved/{}/Best/diffsion_test_model.pth'.format(args.log_name.replace('.log', '')))

                    update_output = set_color('Saving current best', 'blue') + ': %s' % self.saved_model_file
                    logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    logger.info(stop_output)
                    break

                valid_step += 1

            print(f"Epoch {epoch_idx}, Learning Rate: {self.scheduler.get_last_lr()}")

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

        self._add_hparam_to_tensorboard(self.best_valid_score)
        logger.info(set_color('test result', 'yellow') + f': {self.best_valid_result}')

    def calc_reward(self, lastLosses, eps):
        if len(lastLosses) < 3:
            return 1.0
        curDecrease = lastLosses[-2] - lastLosses[-1]
        avgDecrease = 0
        for i in range(len(lastLosses) - 2):
            avgDecrease += lastLosses[i] - lastLosses[i + 1]
        avgDecrease /= len(lastLosses) - 2
        return 1 if curDecrease > avgDecrease else eps

    def trainEpoch(self,train_data, epoch_idx, reward):

        diffusionLoader = self.handler.diffusionLoader_train
        epDiLoss = 0
        for _ in range(10):
            for i, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                self.denoise_opt.zero_grad()
                # flops, params = profile(self.denoise_model, inputs=(batch_item, batch_index))
                diff_loss = self.diffusion_model.training_losses(self.denoise_model, batch_item)

                loss = diff_loss.mean() * reward
                epDiLoss += diff_loss.mean().item()
                loss.backward()

                self.denoise_opt.step()

        logger.info('')
        logger.info('Start to re-build kg')

        with torch.no_grad():
            denoised_edges = []
            u_list = []
            i_list = []
            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                denoised_batch = self.diffusion_model.p_sample(self.denoise_model, batch_item, args.sampling_steps)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)
                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list.append(batch_index[i])
                        i_list.append(indices_[i][j])

            edge_set = set()
            for index in range(len(u_list)):
                edge_set.add((int(u_list[index].cpu().numpy()), int(i_list[index].cpu().numpy())))

            relation_dict_train = self.handler.relation_dict_train
            for index in range(len(u_list)):
                try:
                    denoised_edges.append([u_list[index], i_list[index],
                                           relation_dict_train[int(u_list[index].cpu().numpy())][int(i_list[index].cpu().numpy())]])
                except Exception:
                    continue
            graph_tensor = torch.tensor(denoised_edges)
            index_ = graph_tensor[:, :-1]
            type_ = graph_tensor[:, -1]
            denoisedKG = (index_.t().long().cuda(), type_.long().cuda())

        logger.info('Graph_train built!')

        with torch.no_grad():
            index_, type_ = denoisedKG
            mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)
            denoisedKG = (index_[:, mask], type_[mask])
            self.generatedKG = denoisedKG

        self.model.train()
        logger.info(f"Ir = {self.optimizer.param_groups[0]['lr']}")
        loss_func = self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            )
        )

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)

            self.optimizer.zero_grad()
            losses = loss_func(interaction, self.generatedKG)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self.gpu_available:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

        return total_loss, epDiLoss / 10
    def _valid_epoch(self, valid_data, reward, show_progress=False):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data, reward, load_best_model=False, show_progress=show_progress)
        valid_score = calculate_valid_score(valid_result, self.valid_metric)
        return valid_score, valid_result
    def evaluate(self, eval_data, reward, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            logger.info(message_output)

        self.model.eval()

        eval_func = self._fast_neg_batch_eval

        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = len(self.check_index)

        diffusionLoader = self.handler.diffusionLoader_test
        if args.test_train_graph:
            logger.info('Start training Graph_test')
            for _ in range(10):
                for i, batch in enumerate(diffusionLoader):
                    batch_item, batch_index = batch
                    batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

                    self.denoise_test_opt.zero_grad()

                    diff_loss = self.diffusion_model_test.training_losses(self.denoise_model_test, batch_item)
                    loss = diff_loss.mean() * reward
                    loss.backward()

                    self.denoise_test_opt.step()

            with torch.no_grad():
                denoised_edges = []
                u_list = []
                i_list = []
                for _, batch in enumerate(diffusionLoader):
                    batch_item, batch_index = batch
                    batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                    denoised_batch = self.diffusion_model_test.p_sample(self.denoise_model_test, batch_item, args.sampling_steps)
                    top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

                    for i in range(batch_index.shape[0]):
                        for j in range(indices_[i].shape[0]):
                            u_list.append(batch_index[i])
                            i_list.append(indices_[i][j])
        else:
            denoised_edges = []
            u_list = []
            i_list = []
            for _, batch in enumerate(diffusionLoader):
                batch_item, batch_index = batch
                batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
                denoised_batch = self.diffusion_model.p_sample(self.denoise_model, batch_item,
                                                                    args.sampling_steps)
                top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)

                for i in range(batch_index.shape[0]):
                    for j in range(indices_[i].shape[0]):
                        u_list.append(batch_index[i])
                        i_list.append(indices_[i][j])

        edge_set = set()
        for index in range(len(u_list)):
            edge_set.add((int(u_list[index].cpu().numpy()), int(i_list[index].cpu().numpy())))

        relation_dict_test = self.handler.relation_dict_test
        for index in range(len(u_list)):
            try:
                denoised_edges.append([u_list[index], i_list[index],
                                       relation_dict_test[int(u_list[index].cpu().numpy())][
                                           int(i_list[index].cpu().numpy())]])
            except Exception:
                continue
        graph_tensor = torch.tensor(denoised_edges)
        index_ = graph_tensor[:, :-1]
        type_ = graph_tensor[:, -1]
        denoisedKG_test = (index_.t().long().cuda(), type_.long().cuda())
        index_, type_ = denoisedKG_test
        mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)
        denoisedKG_test = (index_[:, mask], type_[mask])
        self.generatedKG_test = denoisedKG_test
        logger.info('Graph_test built!')

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            )
        )

        for batch_idx, batched_data in enumerate(iter_data):
            interaction, scores, positive_u, positive_i = eval_func(batched_data)
            if self.gpu_available:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head='eval')
        return result

    def _fast_neg_batch_eval(self, batched_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        origin_scores = self.model.fast_predict(interaction.to(self.device), self.generatedKG_test)
        origin_scores = origin_scores.view(-1, 101)
        col_idx = interaction["item_id_with_negs"]
        batch_user_num = positive_u[-1] + 1
        scores = torch.full((batch_user_num, self.tot_item_num), -np.inf, device=self.device)
        for u in positive_u:
            scores[u, col_idx[u]] = origin_scores[u]

        return interaction, scores, positive_u, positive_i

    def _save_checkpoint(self, epoch, saved_model_file):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'other_parameter': self.model.other_parameter(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, saved_model_file)

def get_check_index(dataset):
    input_dict = dataset.field2token_id['item_id_list']
    result = []
    for key, value in input_dict.items():
        try:
            if int(key) >= 1699 and int(key) <= 2400:
                result.append(value)
        except ValueError:
            continue
    return result



if __name__ == '__main__':

    # configurations initialization
    config_dict = {
        "seed": 2020,
        "reproducibility": 1,
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        'neg_sampling': {'uniform': 1},
        'neg_sampling': None,
        'benchmark_filename': ['train', 'test'],
        'alias_of_item_id': ['item_id_list'],
        'topk': [1, 5, 10, 20],
        'metrics': ['Hit', 'NDCG'],
        'valid_metric': 'Hit@1',
        'eval_args': {
            'mode': 'pop100',
            'order': 'TO'
        },
        'gpu_id': args.gpu_id,
        "MAX_ITEM_LIST_LENGTH": 200,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "stopping_step": 10,
        "fast_sample_eval": 1,
        "n_layers":args.n_layers,
        "n_heads": args.n_heads
    }


    config = Config(model=args.model, dataset=f'{args.dataset}', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config, args.log, logfilename=args.log_name)
    logger = getLogger()

    logger.info(args)
    logger.info(config)


    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    train_sampler, _, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio([1 - args.valid_portion, args.valid_portion])
        train_data = get_dataloader(config, 'train')(config, new_train_dataset, None, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, new_test_dataset, None, shuffle=False)
    else:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
        test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)


    logger.info('Start')
    handler = DataHandler(train_data, test_data)
    handler.LoadData()

    check_index = get_check_index(dataset)
    coach = Coach(handler, Denoise, GaussianDiffusion, config, check_index)
    coach.run()

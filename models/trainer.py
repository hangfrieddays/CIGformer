import os
import sys

import torch
import mmcv

pth = os.getcwd()
utils_pth = os.path.join(pth, 'utils')
sys.path.append(utils_pth)

import random
import yaml
from models.utils.logger import Logger
from models.utils.registry import DATASET_REGISTRY
from models.model import *
import datasets
from datasets.utils import *
from Eval import *
from torch.utils.tensorboard import SummaryWriter
from .losses import *
from models.utils.utils import *
from .common import metrics as mtc


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BaseTrainer:
    def __init__(self, config_file, log_name, log_level, log_pth, use_pretrained=False):
        self.config = self.load_config(config_file)
        self.summary_dir = self.config['summary_dir']
        self.model_name = self.config['model']['name']
        self.pretrained_pth = self.config['model']['pretrained_pth']

        self.use_pretrained = use_pretrained
        self.logger = self.init_logger(log_name=log_name, log_level=log_level, log_pth=log_pth)
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.MSELoss().to(self.device)
        self.save_dir = self.config['model']['save_dir']
        self.save_interval = int(self.config['train']['save_interval'])
        self.metric_interval = int(self.config['train']['metric_interval'])
        self.shuffle = self.config['dataset']['train']['use_shuffle']
        self.precision = 6
        self.train_dataset_root = self.config['dataset']['train']['dataset_pth']
        self.test_dataset_root = self.config['dataset']['test']['dataset_pth']

    def init_logger(self, log_pth, log_level, log_name):
        logger_ = Logger(log_pth=log_pth, log_level=log_level, log_name=log_name).get_log()
        return logger_

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config

    def add_writer(self, writer_pth):
        if not os.path.exists(writer_pth):
            os.makedirs(writer_pth)

        return SummaryWriter(writer_pth)

    def init_model(self, model_name, use_pretrained, pretrained_pth):
        model_config = self.config['model']
        if use_pretrained:
            self.logger.info('using the pretrained model!')
            model = MODEL_REGISTRY.get(model_name)(model_config)
            model.load_state_dict(torch.load(pretrained_pth))

        else:
            self.logger.info('using random init model!')
            model = MODEL_REGISTRY.get(model_name)(model_config)
            model.initialize()
        model = model.to(self.device)
        return model

    def init_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def save_training_state(self, epoch):
        save_path = f'{self.config["save_path"]}/model_epoch_{epoch}.pth'
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"saved model at {save_path}")

    def print_network(self):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        self.net.eval()
        assert self.logger, "logger should be defined first!"
        net_cls_str = f'{self.net.__class__.__name__}'

        net_str = str(self.net)
        net_params = sum(map(lambda x: x.numel(), self.net.parameters()))

        self.logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        self.logger.info(net_str)


class TransferTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(TransferTrainer, self).__init__(*args, **kwargs)

        self.n_epochs = self.config['train']['n_epochs']
        self.batch_size = self.config['dataset']['train']['batch_size']
        self.writer = self.add_writer(os.path.join(self.summary_dir, 'transfer_writer')) 
        self.lr = float(self.config['train']['optim']['lr'])
        self.weight_decay = float(self.config['train']['optim']['weight_decay'])
        self.net = self.init_model(model_name=self.model_name, use_pretrained=self.use_pretrained,
                                   pretrained_pth=self.pretrained_pth)

        self.optimizer = self.init_optimizer(lr=self.lr, weight_decay=self.weight_decay)
        self.ratio = self.config['train']['loss']['ratio']
        self.bit_depth = self.config['dataset']['bit_depth']

    def train(self):
        # 创建自定义的转换
        if 'seed' in self.config:
            self.config.info('===> Setting Random Seed')
            set_random_seed(self.congfig['seed'], True) 
        self.logger.info('===> Loading Datasets')
        train_dataset = DATASET_REGISTRY.get('PanSharpeningDataset')(root=self.train_dataset_root,
                                                                     norm_input=True, bit_depth=self.bit_depth)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                                   shuffle=self.shuffle)

        self.logger.info(f'nums of train-set figures{len(train_dataset)}')
        self.logger.info(f'batch size is:{self.batch_size}')


        self.logger.info("------------------Start Training !----------------")
        for epoch in range(self.n_epochs):
            # Train
            self.logger.info(f'epoch:{epoch}')
            self.net.train()  # 单独训练transfer
            for i, item in enumerate(train_loader): 
                image_pan = item['image_pan'].to(self.device)
                image_ms_label = item['image_ms_label'].to(self.device)

                self.optimizer.zero_grad()
                image_transfer = self.net(image_ms_label.to(self.device)).squeeze(
                    1) 
                loss = self.loss_function(image_transfer.to(self.device),
                                          image_pan.to(self.device))  
                loss.backward()
                self.optimizer.step()
                self.logger.info(
                    f'Epoch : {epoch}/{self.n_epochs}   Batch : {i}/{len(train_loader)}  Loss : {loss.item() * self.ratio:{self.precision}f}')
            if epoch % self.save_interval == 0:
                torch.save(self.net.state_dict(), os.path.join(self.save_dir, f'{epoch}.pth'))

class PromoteFusionTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(PromoteFusionTrainer, self).__init__(*args, **kwargs)

        self.n_epochs = self.config['train']['n_epochs']
        self.train_batch_size = self.config['dataset']['train']['batch_size']
        self.test_batch_size = self.config['dataset']['test']['batch_size']
        self.bit_depth = self.config['dataset']['bit_depth']

        self.transfer_cfg = self.config['transfer']

        self.writer = self.add_writer(os.path.join(self.summary_dir, 'PromteFusion_Writer'))
        self.losswriter = {
            'train': self.add_writer(os.path.join(self.summary_dir, 'train_ref_loss_writer')),
        }

        self.mtcwriter = {
            'ERGAS':   self.add_writer(os.path.join(self.summary_dir, 'EARGS_writer')),
            'SSIM':    self.add_writer(os.path.join(self.summary_dir, 'SSIM_writer')),
            'MPSNR':    self.add_writer(os.path.join(self.summary_dir, 'PSNR_writer')),
            'SCC':     self.add_writer(os.path.join(self.summary_dir, 'SCC_writer')),
            'SAM':  self.add_writer(os.path.join(self.summary_dir, 'SAM_writer')),
            'Q4':   self.add_writer(os.path.join(self.summary_dir, 'Q4_writer')),  
        }

        self.lr = float(self.config['train']['optim']['lr'])
        self.weight_decay = float(self.config['train']['optim']['weight_decay'])
        self.net = self.init_model(model_name=self.model_name, use_pretrained=self.use_pretrained,
                                   pretrained_pth=self.pretrained_pth)
        self.test_out = self.config['dataset']['test']['test_out']
        self.optimizer = self.init_optimizer(lr=self.lr, weight_decay=self.weight_decay)

        self.choose_save = self.config['dataset']['test']['choose_save']
        self.scale = self.config['model']['scale']
        self.eval_results = {}
        self.getLossFunction()
        self.SetLossRatio()

    def getLossFunction(self):
        self.REFLoss = REFLoss()
        self.PCLoss = SpatialConsistentLoss()
        self.MCLoss = SpectralConsistentLoss()

    def SetLossRatio(self):
        # set loss ratio
        self.loss_alpha = self.config['loss_ratio']['fusion']['loss_alpha']

    def train(self):
        if 'seed' in self.config:
            self.config.info('===> Setting Random Seed')
            set_random_seed(self.congfig['seed'], True)

        self.logger.info('===> Loading model')
        self.print_network()
        self.logger.info('===> Loading Datasets')
        train_dataset = DATASET_REGISTRY.get('PanSharpeningDataset')(root=self.train_dataset_root,
                                                                     norm_input=True, bit_depth=self.bit_depth)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.train_batch_size,
                                                   shuffle=self.shuffle)
        self.logger.info(f'nums of train-set figures{len(train_dataset)}')
        self.logger.info(f'batch size is:{self.train_batch_size}')

        test_dataset = DATASET_REGISTRY.get('PanSharpeningDataset')(root=self.test_dataset_root,
                                                                    norm_input=True, bit_depth=self.bit_depth, mode = 'test')
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.test_batch_size,
                                                       shuffle=self.shuffle)

        self.logger.info("===> start training")
        transfer_pan = TransferNetwork(cfg=self.transfer_cfg)
        transfer_pth = self.config['model']['transfer_pth']
        if os.path.exists(transfer_pth):
            self.logger.info('TransferNet is loading parameters')
            transfer_pan.load_state_dict(torch.load(transfer_pth, map_location=self.device))
        else:
            self.logger.info('Transfer self.net loading Error')
            exit(0)

        transfer_pan.to(self.device)
        loss_eval = 9999999
        train_loss_0 = 99999999
        # training loop
        count = 0

        for epoch in range(self.n_epochs):
            self.logger.info(f'epoch:{epoch}')
            # train
            self.net.train()
            transfer_pan.eval()  
            train_loss = 0
            bct = 0
            for i, item in enumerate(train_loader):
                count += 1
                bct+=1
                image_pan = item['image_pan'].to(self.device)
                image_ms = item['image_ms'].to(self.device)
                image_ms_label = item['image_ms_label'].to(self.device)

                # Reduced-Resolution Fusion Stage  
                self.optimizer.zero_grad()
                HMS = self.net(image_pan, image_ms, transfer_pan) 
                Intensity_generate = transfer_pan(HMS)
                Intensity_truth = transfer_pan(image_ms_label)

                loss_ref = self.REFLoss(HMS, image_ms_label)

                loss_PC = self.PCLoss(Intensity_generate, Intensity_truth)
                loss_MC = self.MCLoss(HMS, image_ms_label, Intensity_generate, Intensity_truth)
                loss_sum = self.loss_alpha * loss_ref + (1 - self.loss_alpha) * (loss_MC + loss_PC)
                loss_sum = loss_ref
                loss_sum.backward()
                train_loss += loss_ref

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.logger.info(f'Epoch : {epoch}/{self.n_epochs}   Batch : {i}/{len(train_loader)}  '
                                 f'ref Loss : [{loss_ref.item():.{self.precision}f}]]'
                                 f'MC loss : [{loss_MC.item():.{self.precision}f}]'
                                 f'PC loss : [{loss_PC.item():.{self.precision}f}]')
            
                if (count + 1) % self.metric_interval == 0:
                    self.test(count, save=self.choose_save, transferpan=transfer_pan)

            self.losswriter['train'].add_scalar('ref loss  ', train_loss * self.ratio / bct, global_step=epoch)

            # Save
            if (epoch + 1) % self.save_interval == 0:
                torch.save(self.net.state_dict(), os.path.join(self.save_dir, f'{epoch}.pth'))
            if train_loss < train_loss_0:
                train_loss_0 = train_loss
                torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'PCGNet.pth'))


    def test(self, iter_id, save, transferpan):
        r""" test and evaluate the model

        Args:
            iter_id (int): current iteration num
            save (bool): whether to save the output of test images
            ref (bool): True for low-res testing, False for full-res testing
        """
        use_sewar = self.config['train']['use_sewar']
        self.logger.info(f'resolution testing {"with sewar" if use_sewar else ""}...')
        self.net.eval()
        transferpan.eval()
        ref = True
        test_path = os.path.join(self.test_out, f'{iter_id}')
        if save:
            if not os.path.exists(test_path):
                os.makedirs(test_path, exist_ok=True)

        tot_time = 0
        tot_count = 0
        tmp_results = {}
        eval_metrics = ['SAM', 'ERGAS', 'Q4', 'SCC', 'SSIM', 'MPSNR'] if ref \
            else ['D_lambda', 'D_s', 'QNR', 'FCC', 'SF', 'SD', 'SAM_nrf']

        eval_metrics = ['SAM', 'ERGAS', 'Q4', 'SCC', 'SSIM', 'MPSNR']
        for metric in eval_metrics:
            tmp_results.setdefault(metric, [])

        for _, input_batch in enumerate(self.test_loader):
            image_pan = input_batch['image_pan'].to(self.device)
            image_ms = input_batch['image_ms'].to(self.device)
            image_ms_label = input_batch['image_ms_label'].to(self.device)

            image_ids = input_batch['image_id']
            n = len(image_ids)
            tot_count += n
            timer = mmcv.Timer()
            with torch.no_grad():
                output = self.net(image_pan, image_ms, transferpan)
            tot_time += timer.since_start()

            image_pan = torch2np(image_pan)
            image_ms = torch2np(image_ms)
            if ref:
                target = torch2np(image_ms_label)
            output_np = torch2np(output)

            if self.config['model']['norm_input']:
                image_pan = data_denormalize(image_pan, self.bit_depth)
                image_ms = data_denormalize(image_ms, self.bit_depth)
                if ref:
                    target = data_denormalize(target, self.bit_depth)
                    output_np = data_denormalize(output_np, self.bit_depth)
                output = data_denormalize(output, self.bit_depth)

            for i in range(n):
                if ref:
                    tmp_results['SAM'].append(mtc.SAM_numpy(target[i], output_np[i], sewar=use_sewar))
                    tmp_results['ERGAS'].append(mtc.ERGAS_numpy(target[i], output_np[i], sewar=use_sewar))
                    tmp_results['Q4'].append(mtc.Q4_numpy(target[i], output_np[i]))
                    tmp_results['SCC'].append(mtc.SCC_numpy(target[i], output_np[i], sewar=use_sewar))
                    tmp_results['SSIM'].append(mtc.SSIM_numpy(target[i], output_np[i], 2 ** self.bit_depth - 1,
                                                              sewar=use_sewar))
                    tmp_results['MPSNR'].append(mtc.MPSNR_numpy(target[i], output_np[i], 2 ** self.bit_depth - 1))
                else:
                    tmp_results['D_lambda'].append(mtc.D_lambda_numpy(image_ms[i], output_np[i], sewar=use_sewar))
                    tmp_results['D_s'].append(mtc.D_s_numpy(image_ms[i], image_pan[i], output_np[i], sewar=use_sewar))
                    tmp_results['QNR'].append((1 - tmp_results['D_lambda'][-1]) * (1 - tmp_results['D_s'][-1]))
                    tmp_results['FCC'].append(mtc.FCC_numpy(image_pan[i], output_np[i]))
                    tmp_results['SF'].append(mtc.SF_numpy(output_np[i]))
                    tmp_results['SD'].append(mtc.SD_numpy(output_np[i]))
                    tmp_results['SAM_nrf'].append(
                        mtc.SAM_numpy(image_ms[i], cv2.resize(output_np[i], (100, 100)), sewar=use_sewar))

                if save:
                    save_image(os.path.join(test_path, f'{image_ids[i]}_mul_hat.tif'), output[i].cpu().detach().numpy())

        for metric in eval_metrics:
            self.eval_results.setdefault(f'{metric}_mean', [])
            self.eval_results.setdefault(f'{metric}_std', [])
            mean = np.mean(tmp_results[metric])
            std = np.std(tmp_results[metric])
            self.eval_results[f'{metric}_mean'].append(round(mean, 4))
            self.eval_results[f'{metric}_std'].append(round(std, 4))
            self.logger.info(f'{metric} metric value: {mean:.4f} +- {std:.4f}')
            self.mtcwriter[metric].add_scalar(f'{metric}_mean', round(mean, 4), global_step=iter_id)

        if iter_id == self.n_epochs:  # final testing
            for metric in eval_metrics:
                mean_array = self.eval_results[f'{metric}_mean']
                self.logger.info(f'{metric} metric curve: {mean_array}')
        self.logger.info(f'Avg time cost per img: {tot_time / tot_count:.5f}s')
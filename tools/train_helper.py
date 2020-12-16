# -*- coding: utf-8 -*-
# @Time    : 2020/12/04 11:15
# @Author  : liyujun
import os
import time
import torch
from tqdm import tqdm
from config.config import base_dir
import shutil
from utils.util import setup_logger, cal_text_score
from utils.schedulers import WarmupPolyLR
from utils.metrics import runningScore


class Train_Helper(object):
    def __init__(self, train_config, model, criterion, train_loader, validate_loader, metric_cls=None, post_process=None):
        super(Train_Helper, self).__init__()
        save_output = os.path.abspath(os.path.join(base_dir, train_config.train_output_dir))
        save_name = "DBNet" + "_" + model.name
        self.save_dir = os.path.abspath(os.path.join(save_output, save_name))             # 训练过程保存文件夹
        self.checkpoint_dir = os.path.abspath(os.path.join(self.save_dir, "checkpoint"))    # checkpoint保存目录

        if train_config.train_resume_checkpoint == "" and train_config.train_finetune_checkpoint == "":
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 0
        self.config = train_config
        self.model = model                          # DBNet网络模型
        self.criterion = criterion                  # 损失函数定义

        self.epochs = train_config.train_epochs
        self.log_iter = train_config.train_log_iter

        self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))

        # device
        torch.manual_seed(self.config.trainer_seed)  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True           # 保证每次方向传播参数结果一样
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config.trainer_seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config.trainer_seed)  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")

        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))
        # metrics
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_config.opti_lr, weight_decay=train_config.opti_weight_decay, amsgrad=train_config.opti_amsgrad)

        # resume or finetune
        if self.config.train_resume_checkpoint != '':
            self._load_checkpoint(self.config.train_resume_checkpoint, resume=True)
        elif self.config.train_finetune_checkpoint != '':
            self._load_checkpoint(self.config.train_finetune_checkpoint, resume=False)

        self.model.to(self.device)
        # make inverse Normalize
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]
        self.UN_Normalize = True

        self.show_images_iter = self.config.train_show_images_iter
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)

        if self.config.lr_scheduler_type == 'WarmupPolyLR':
            warmup_iters = self.config.lr_scheduler_warmup_epoch * self.train_loader_len
            lr_dict = {}
            if self.start_epoch > 1:
                self.config.lr_scheduler_last_epoch = (self.start_epoch - 1) * self.train_loader_len
                lr_dict["last_epoch"] = self.config.lr_scheduler_last_epoch
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **lr_dict)

        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), self.train_loader_len))

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            # backward
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if self.config.lr_scheduler_type == 'WarmupPolyLR':
                self.scheduler.step()
            # acc iou
            score_shrink_map = cal_text_score(preds[:, 0, :, :], batch['shrink_map'], batch['shrink_mask'], running_metric_text,
                                              thred=self.config.post_processing_thresh)

            # loss 和 acc 记录到日志
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, acc: {:.4f}, iou_shrink_map: {:.4f}, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, acc,
                        iou_shrink_map, loss_str, lr, batch_time))
                batch_start = time.time()

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                # 数据进行转换和丢到gpu
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(batch, preds,is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        net_model_save_path = '{}/model_latest_model.pth'.format(self.checkpoint_dir)
        net_model_save_path_best = '{}/model_best_model.pth'.format(self.checkpoint_dir)


        self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
        save_best = False
        if self.validate_loader is not None and self.metric_cls is not None:  # 使用f1作为最优模型指标
            recall, precision, hmean = self._eval(self.epoch_result['epoch'])

            self.logger_info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean))

            if hmean >= self.metrics['hmean']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['hmean'] = hmean
                self.metrics['precision'] = precision
                self.metrics['recall'] = recall
                self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        else:
            if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.logger_info(best_str)
        if save_best:
            import shutil
            shutil.copy(net_save_path, net_save_path_best)
            shutil.copy(net_model_save_path, net_model_save_path_best)
            self.logger_info("Saving current best: {}".format(net_save_path_best))
        else:
            self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
            'models': self.model
        }
        model_name = file_name.split(".")[0] + "_model.pth"
        filename = os.path.join(self.checkpoint_dir, file_name)
        model_name = os.path.join(self.checkpoint_dir, model_name)
        torch.save(state, filename)
        torch.save(self.model, model_name)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config.lr_scheduler_last_epoch = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger_info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def logger_info(self, s):
        # 输出信息至logger文件中
        self.logger.info(s)






import os
import random

import numpy as np
import torch

# Per-epoch history; the time columns make TrainModel record the time per epoch (BP and EP)
HISTORY_KEYS = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'clean_train_loss', 'clean_train_acc', 'lr',
                'generalization_gap', 'train_time_s', 'epoch_time_s']


# Best / latest checkpoints, random states and exact resume for one run
class CheckpointManager:
    def __init__(self, save_dir, model, optimizer, scheduler, config, device, generators):
        self.save_dir, self.model, self.optimizer, self.scheduler = save_dir, model, optimizer, scheduler
        self.config, self.device, self.generators = config, device, generators
        self.best_path = save_dir / 'best_model.pth'
        self.latest_path = save_dir / 'latest_checkpoint.pth'
        self.history = None

    def rng_state(self):
        state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     mps=torch.mps.get_rng_state() if self.device.type == 'mps' else None)  # dropout on MPS
        state.update({name: g.get_state() for name, g in self.generators.items()})
        return state

    def restore_rng(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if torch.cuda.is_available() and state.get('cuda') is not None:
            torch.cuda.set_rng_state_all(state['cuda'])
        if self.device.type == 'mps' and state.get('mps') is not None:
            torch.mps.set_rng_state(state['mps'])
        for name, g in self.generators.items():
            g.set_state(state[name])

    # Fresh run or resume; returns (history, best_acc, best_epoch, start_epoch)
    def start(self, resume, history_keys=HISTORY_KEYS, overwrite=False):
        if not resume:
            if self.latest_path.exists() and not overwrite:  # never overwrite a run by accident
                raise FileExistsError(f'{self.latest_path} already exists. Set RESUME = True to continue that run, '
                                      f'or pass overwrite=True to ckpt.start to start it again from epoch 1.')
            self.history = {k: [] for k in history_keys}
            print('Starting fresh training from epoch 1.')
            return self.history, -1.0, 0, 1

        ck = torch.load(self.latest_path, map_location='cpu', weights_only=False)
        saved = ck['config']
        if saved != self.config:
            diff = sorted(k for k in set(saved) | set(self.config) if saved.get(k) != self.config.get(k))
            raise ValueError(f'Resume configuration does not match this experiment: {diff}')

        self.model.load_state_dict(ck['model'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.scheduler.load_state_dict(ck['scheduler'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        self.restore_rng(ck['rng'])
        self.history = ck['history']
        start_epoch = ck['epoch'] + 1
        print(f'Resuming from epoch {start_epoch} | LR = {self.scheduler.get_last_lr()[0]:.10f}')
        return self.history, ck['best_acc'], ck['best_epoch'], start_epoch

    # Load best_model.pth into the model
    def load_best(self):
        best = torch.load(self.best_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(best['model'])
        return best

    def save_latest(self, epoch, best_acc, best_epoch):
        payload = dict(epoch=epoch, model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                       scheduler=self.scheduler.state_dict(), history=self.history, best_acc=best_acc,
                       best_epoch=best_epoch, rng=self.rng_state(), config=self.config)
        temp = self.latest_path.with_suffix('.tmp')
        torch.save(payload, temp)
        os.replace(temp, self.latest_path)

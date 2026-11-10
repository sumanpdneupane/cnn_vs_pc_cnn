import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.utils.metrics_csv import save_metrics_csv


# Training loop for every model. Only the gradient differs: 'bp' = loss.backward(), 'ep' = model.ep_gradients().
class TrainModel:
    def __init__(self, model, optimizer, scheduler, criterion, device, method=None, clip=1.0):
        method = method or model.training_method
        if method not in ('bp', 'ep'):
            raise ValueError(f"method must be 'bp' or 'ep', got {method!r}")
        if method == 'ep' and not hasattr(model, 'ep_gradients'):
            raise ValueError(f"{type(model).__name__} cannot be trained with EP; use PCCNN for method='ep'")
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.criterion, self.device, self.method, self.clip = criterion, device, method, clip
        self.per_image_loss = nn.CrossEntropyLoss(label_smoothing=criterion.label_smoothing, reduction='none')

    # Train, evaluate, save checkpoints, write metrics.csv
    def fit(self, train_loader, val_loader, clean_train_loader, epochs, history, save_dir, save_latest,
            start_epoch=1, best_acc=-1.0, best_epoch=0, clean_eval_interval=5, best_info=None):
        best_path = save_dir / 'best_model.pth'
        timed = 'train_time_s' in history
        for epoch in tqdm(range(start_epoch, epochs + 1)):
            epoch_start = time.time()
            train_loss, train_acc = self.run_epoch(train_loader, training=True)
            train_time = time.time() - epoch_start

            if epoch == 1 or epoch % clean_eval_interval == 0:
                clean_train_loss, clean_train_acc = self.run_epoch(clean_train_loader)
            else:
                clean_train_loss, clean_train_acc = None, None

            val_loss, val_acc = self.run_epoch(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            epoch_metrics = {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss,
                             'val_acc': val_acc, 'clean_train_loss': clean_train_loss,
                             'clean_train_acc': clean_train_acc, 'lr': current_lr,
                             'generalization_gap': clean_train_acc - val_acc if clean_train_acc is not None else None}
            if timed:
                epoch_metrics.update(train_time_s=train_time, epoch_time_s=epoch_time)
            for name, value in epoch_metrics.items():
                history[name].append(value)

            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                torch.save({'model': self.model.state_dict(), 'epoch': epoch, 'val_acc': val_acc,
                            **(best_info or {})}, best_path)

            save_latest(epoch, best_acc, best_epoch)

            clean_text = f'{clean_train_acc:.2f}% / {clean_train_loss:.4f}' if clean_train_acc is not None else 'Skipped'
            time_text = f' | Time: {train_time / 60:.1f} min train, {epoch_time / 60:.1f} min total' if timed else ''
            print(f'Epoch {epoch:03d}/{epochs} | Train: {train_acc:.2f}% / {train_loss:.4f} | '
                  f'Val: {val_acc:.2f}% / {val_loss:.4f} | Clean Train: {clean_text} | LR: {current_lr:.7f} | '
                  f'Best: {best_acc:.2f}% (epoch {best_epoch})' + time_text)

        print(f'Training finished at epoch {epochs}. Best validation accuracy: {best_acc:.2f}% (epoch {best_epoch})')
        print('Metrics CSV saved:', save_metrics_csv(history, save_dir / 'metrics.csv'))
        return best_acc, best_epoch

    # One pass over a loader. training=True: (images, labels_a, labels_b, lam) batches, gradient step per batch.
    # training=False: evaluation on (images, labels). Returns (mean loss, accuracy %).
    def run_epoch(self, loader, training=False):
        self.model.train(training)
        total_loss = torch.zeros((), device=self.device)
        total_correct = torch.zeros((), device=self.device)
        total_images = 0
        if not training:
            context = torch.inference_mode()
        else:
            if self.method == 'bp':
                context = torch.enable_grad()  # BP needs autograd
            else:
                context = torch.no_grad()  # EP needs no autograd

        with context:
            for batch in tqdm(loader, leave=False, desc='Train' if training else 'Eval'):
                images = batch[0].to(self.device, non_blocking=True)
                if training:
                    labels_a, labels_b, lam = (item.to(self.device, non_blocking=True) for item in batch[1:])
                    loss, correct = self.train_step(images, labels_a, labels_b, lam)
                else:
                    labels = batch[1].to(self.device, non_blocking=True)
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    correct = (logits.detach().argmax(dim=1) == labels).sum()

                batch_size = images.size(0)
                total_images += batch_size
                total_loss += loss.detach() * batch_size
                total_correct += correct

        return (total_loss / total_images).item(), (total_correct * (100.0 / total_images)).item()

    # One training batch: gradient (BP or EP), per-tensor clip, optimizer and LR scheduler step.
    # Returns the mixed loss and the (lam-weighted) number of correct predictions.
    def train_step(self, images, labels_a, labels_b, lam):
        self.optimizer.zero_grad(set_to_none=True)

        if self.method == 'bp':
            logits = self.model(images)
            loss = self.mixed_loss(logits, labels_a, labels_b, lam)
            loss.backward()
        else:
            logits = self.model.ep_gradients(images, labels_a, labels_b, lam)  # writes param.grad, no backprop
            loss = self.mixed_loss(logits, labels_a, labels_b, lam)

        predictions = logits.detach().argmax(dim=1)
        correct = (lam * (predictions == labels_a) + (1.0 - lam) * (predictions == labels_b)).sum()

        for param in self.model.parameters():
            torch.nn.utils.clip_grad_norm_(param, self.clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss, correct

    # lam * CE(label_a) + (1 - lam) * CE(label_b), with label smoothing, averaged over the batch
    def mixed_loss(self, logits, labels_a, labels_b, lam):
        return (lam * self.per_image_loss(logits, labels_a)
                + (1.0 - lam) * self.per_image_loss(logits, labels_b)).mean()

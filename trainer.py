import os

import torch
import numpy as np


def save(model, ckpt_num, dir_name):
    os.makedirs(dir_name, exist_ok=True)
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    else:
        torch.save(model.state_dict(), os.path.join(dir_name, 'model_%s' % ckpt_num))
    print('model saved!')


def fit(train_loader, model, loss_fn, optimizer, scheduler, nb_epoch,
        device, log_interval, start_epoch=0, save_model_to='/tmp/save_model_to'):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    # Save pre-trained model
    save(model, 0, save_model_to)

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, nb_epoch):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval)

        log_dict = {'epoch': epoch + 1,
                    'epoch_total': nb_epoch,
                    'loss': float(train_loss),
                    }

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, nb_epoch, train_loss)
        for metric in metrics:
            log_dict[metric.name()] = metric.value()
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        print(log_dict)
        if (epoch + 1) % 5 == 0:
            save(model, epoch + 1, save_model_to)


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval):

    for metric in loss_fn.metrics:
        metric.reset()

    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)

        data = tuple(d.to(device) for d in data)
        if target is not None:
            target = target.to(device)

        optimizer.zero_grad()
        if loss_fn.cross_entropy_flag:
            output_embedding, output_cross_entropy = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding, output_cross_entropy)
        else:
            output_embedding = model(*data)
            blended_loss, losses = loss_fn.calculate_loss(target, output_embedding)
        total_loss += blended_loss.item()
        blended_loss.backward()

        optimizer.step()

        # Print log
        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(data[0]), len(train_loader.dataset), 100. * batch_idx / len(train_loader))
            for name, value in losses.items():
                message += '\t{}: {:.6f}'.format(name, np.mean(value))
            for metric in loss_fn.metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)

    total_loss /= (batch_idx + 1)
    return total_loss, loss_fn.metrics


""" 
Models for different kinds of training.
"""

import os
import torch
import torch.nn as nn 
import transformers
import data_utils
import train_utils
import pandas as pd
import numpy as np
import common
import losses
import wandb


class AdverseEventClassification:
    ''' Adverse event classification task '''

    def __init__(self, args):

        # Initialize and data loaders
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)
        self.train_loader, self.val_loader = data_utils.get_dataloaders(
            root=args['data_root'], val_size=self.config['val_size'], batch_size=self.config['batch_size'])
        self.done_epochs = 0

        # Models, optimizer and scheduler
        self.encoder = transformers.AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').to(self.device)
        self.clf_head = nn.Linear(self.config['encoder']['model_dim'], 2).to(self.device)
        
        self.optim = train_utils.get_optimizer(
            config = self.config['optim'], 
            params = list(self.encoder.parameters()) + list(self.clf_head.parameters())
        )
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']}, 
            optimizer = self.optim
        )

        # Param count
        total_params = common.count_parameters([self.encoder, self.clf_head, self.embeds])
        if total_params // 1e06 > 0:
            self.logger.record(f'Total trainable parameters: {round(total_params/1e06, 2)} M', mode='info')
        else:
            self.logger.record(f'Total trainable parameters: {total_params}', mode='info')
        
        # Losses and performance monitoring
        self.criterion = losses.ClassificationLoss(self.device)
        self.best_val_acc = 0
        run = wandb.init("medical-transformer-aec")
        self.logger.write(run.get_url(), mode='info')

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr'] / self.warmup_epochs

        # Save state handling
        if os.path.exists(os.path.join(self.output_dir, 'last_state.ckpt')):
            self.done_epochs = self.load_state()
            self.logger.print("Successfully loaded last saved state", mode='info')
        else:
            self.logger.print("No saved state found, starting fresh", mode='info')

        # Load best model if specified in args
        if args['load'] is not None:
            if os.path.exists(os.path.join(args['load'], 'best_model.ckpt')):
                self.load_model(args['load'])
                self.logger.print(f"Succesfully loaded model from {args['load']}", mode='info')
            else:
                raise NotImplementedError(f"No saved model found at {args['load']}")

    def train_on_batch(self, batch):
        inp, trg = batch
        inp = inp.to(self.device)
        out = self.clf_head(self.encoder(inp)['last_hidden_state'][:, 0, :])
        
        loss, acc = self.criterion(out, trg)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return {'Loss': loss.item(), 'Accuracy': acc}

    def validate_on_batch(self, batch):
        inp, trg = batch
        inp = inp.to(self.device)
        with torch.no_grad():
            out = self.clf_head(self.encoder(inp)['last_hidden_state'][:, 0, :])
        
        loss, acc = self.criterion(out, trg)
        return {'Loss': loss.item(), 'Accuracy': acc}

    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()

    def save_state(self, epoch):
        state = {
            'encoder': self.encoder.state_dict(),
            'clf': self.clf_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'epoch': epoch
        }
        if epoch % int(0.1 * self.config['epochs']) == 0:
            torch.save(state, os.path.join(self.output_dir, f'state_epoch_{epoch}.ckpt'))
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))

    def save_model(self):
        state = {
            'encoder': self.encoder.state_dict(),
            'clf': self.clf_head.state_dict(),
        }
        torch.save(state, os.path.join(self.output_dir, 'best_model.ckpt'))

    def load_state(self):
        state = torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))
        self.encoder.load_state_dict(state['encoder'])
        self.clf_head.load_state_dict(state['clf'])
        self.optim.load_state_dict(state['optim'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        return state['epoch']

    def load_model(self, output_dir):
        state = torch.load(os.path.join(output_dir, 'best_model.ckpt'))
        self.encoder.load_state_dict(state['encoder'])
        self.clf_head.load_state_dict(state['clf'])

    def save_embeds_for_projection(self):
        embed_vals = self.encoder.embeddings.word_embeddings.weight.data.detach().cpu().numpy()
        word_index = self.train_loader.word2idx
        labels = np.array(sorted(list(word_index.keys()), key=lambda i: word_index[i])).reshape(-1, 1)
        pd.DataFrame(embed_vals).to_csv(os.path.join(self.output_dir, 'embeds.tsv'), sep='\t', index=None, header=None)
        pd.DataFrame(labels).to_csv(os.path.join(self.output_dir, 'labels.tsv'), sep='\t', index=None, header=None)
        self.logger.record(f'Saved embeddings and metadata to {self.output_dir}', mode='info')

    def test_model(self):
        lines = data_utils.load_data(self.args['test_root'], name='test_data')
        data, labels = [], []
        for l in lines:
            if len(l) > 0:
                data.append(l.split('\t')[0])
                labels.append(int(l.split('\t')[1]))
        data, labels = np.asarray(data), np.asarray(labels)

        tokens = self.train_loader.pad_and_tokenize_(data)
        out = self.clf_head(self.encoder(tokens)['last_hidden_state'][:, 0, :]).argmax(dim=-1).detach().cpu().numpy()
        acc = (out == labels).mean()
        
        for i in range(len(data)):
            print("Entry:", data[i])
            print(f"Target: {labels[i]} \t Prediction: {out[i]}")
            print()
        
        self.logger.print(f"Model accuracy: {acc}", mode='val')


    def train(self):
        print()
        for epoch in range(self.done_epochs, self.config['epochs']+1):
            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()
            self.logger.record('Epoch [{:3d}/{}]'.format(epoch, self.config['epochs']), mode='train')
            self.adjust_learning_rate(epoch+1)

            for idx in range(len(self.train_loader)):
                batch = self.train_loader.flow()
                train_metrics = self.train_on_batch(batch)
                wandb.log({'Loss': train_metrics['Loss'], 'Epoch': epoch})
                train_meter.add(train_metrics)
                common.progress_bar(progress=idx/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            wandb.log({'Train accuracy': train_meter.return_metrics()['Accuracy'], 'Epoch': epoch})
            wandb.log({'Learning rate': self.optim.param_groups[0]['lr'], 'Epoch': epoch})

            # Save state
            self.save_state(epoch)

            # Validation
            if epoch % self.config['eval_every'] == 0:
                self.logger.record('Epoch [{:3d}/{}]'.format(epoch, self.config['epochs']), mode='val')
                
                for idx in range(len(self.val_loader)):
                    batch = self.val_loader.flow()
                    val_metrics = self.validate_on_batch(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=idx/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                val_metrics = val_meter.return_metrics()
                wandb.log({'Validation loss': val_metrics['Loss'], 'Validation accuracy': val_metrics['Accuracy'], 'Epoch': epoch})

                if val_metrics['Accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['Accuracy']
                    self.save_model()

        print('\n\n[INFO] Training complete!')
        self.save_embeds_for_projection()
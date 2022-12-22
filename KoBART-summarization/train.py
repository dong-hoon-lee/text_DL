import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KobartSummaryModule
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from rouge import Rouge

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

parser.add_argument('--mode',
                    default='train',
                    type=str,
                    help='set [train / test] mode')

parser.add_argument('--hparams_file',
                    type=str,
                    help='input hparams.yaml file')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train.tsv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default='data/test.tsv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=14,
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class Base(pl.LightningModule):
    def __init__(self, hparams, trainer, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)
        self.trainer = trainer

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=14,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=3e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser
    
    def setup_steps(self, stage=None):
        # NOTE There is a problem that len(train_loader) does not work.
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers

        data_len = self.setup_steps(self)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, trainer=None, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, trainer, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
        
    ######## ADDITIONAL CODES ##########
    def test_step(self, batch, batch_idx):
        # rouge score calculation
        score = {'rouge-1':{'r':0, 'p':0, 'f':0}, 
                 'rouge-2':{'r':0, 'p':0, 'f':0}, 
                 'rouge-l':{'r':0, 'p':0, 'f':0}}
        rouge = Rouge()
        
        x = batch['input_ids']
        y = batch['labels']
        
        output = self.model.generate(x, eos_token_id=1, max_length=512, num_beams=5)
        # print(f'\n\n{len(output)}\n\n')
        # print(f'{output[1]}\n\n')
        # print(f'{y[13]}\n\n')
        
        
        for i in range(len(output)):
            predict = self.tokenizer.decode(output[i], skip_special_tokens=True)
            # pred = output[i]
            l = y[i].tolist()
            idx = l.index(1)
            l = l[:idx+1]
            # print(l)
            label = self.tokenizer.decode(l, skip_special_tokens=True)
            # if len(pred) != len(label):
            #     label = label[:len(pred)]
            
            # print(predict)
            # print(label)
            
            s = rouge.get_scores(predict, label)[0]
            
            score['rouge-1']['r'] += s['rouge-1']['r']
            score['rouge-1']['p'] += s['rouge-1']['p']
            score['rouge-1']['f'] += s['rouge-1']['f']
            
            score['rouge-2']['r'] += s['rouge-2']['r']
            score['rouge-2']['p'] += s['rouge-2']['p']
            score['rouge-2']['f'] += s['rouge-2']['f']
            
            score['rouge-l']['r'] += s['rouge-l']['r']
            score['rouge-l']['p'] += s['rouge-l']['p']
            score['rouge-l']['f'] += s['rouge-l']['f']
            
            
        score['rouge-1']['r'] /= len(output)
        score['rouge-1']['p'] /= len(output)
        score['rouge-1']['f'] /= len(output)
            
        score['rouge-2']['r'] /= len(output)
        score['rouge-2']['p'] /= len(output)
        score['rouge-2']['f'] /= len(output)
            
        score['rouge-l']['r'] /= len(output)
        score['rouge-l']['p'] /= len(output)
        score['rouge-l']['f'] /= len(output)
        
        return score
    
    def test_epoch_end(self, outputs):
        score = {'rouge-1':{'r':0, 'p':0, 'f':0}, 'rouge-2':{'r':0, 'p':0, 'f':0}, 'rouge-l':{'r':0, 'p':0, 'f':0}}
        
        for s in outputs:
            score['rouge-1']['r'] += s['rouge-1']['r']
            score['rouge-1']['p'] += s['rouge-1']['p']
            score['rouge-1']['f'] += s['rouge-1']['f']
            
            score['rouge-2']['r'] += s['rouge-2']['r']
            score['rouge-2']['p'] += s['rouge-2']['p']
            score['rouge-2']['f'] += s['rouge-2']['f']
            
            score['rouge-l']['r'] += s['rouge-l']['r']
            score['rouge-l']['p'] += s['rouge-l']['p']
            score['rouge-l']['f'] += s['rouge-l']['f']
            
        
        score['rouge-1']['r'] /= len(outputs)
        score['rouge-1']['p'] /= len(outputs)
        score['rouge-1']['f'] /= len(outputs)
            
        score['rouge-2']['r'] /= len(outputs)
        score['rouge-2']['p'] /= len(outputs)
        score['rouge-2']['f'] /= len(outputs)
            
        score['rouge-l']['r'] /= len(outputs)
        score['rouge-l']['p'] /= len(outputs)
        score['rouge-l']['f'] /= len(outputs)
        
        
        df = pd.DataFrame(score)
        df.to_csv(os.path.join(args.default_root_dir, 'rouge_score.csv'))
        ##########################################

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
    args = parser.parse_args()
    logging.info(args)

    dm = KobartSummaryModule(args.train_file,
                        args.test_file,
                        tokenizer,
                        batch_size=args.batch_size,
                        max_len=args.max_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=3)
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,   # training stops if val_loss doesn't improve for 10 epochs
        verbose=True,
        mode='min'
    )

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])

    # model = KoBARTConditionalGeneration(args, trainer)
    # trainer.fit(model, dm)
    
    if args.mode == 'train':
        if not args.checkpoint_path and args.hparams_file:
            model = KoBARTConditionalGeneration(args, trainer)
        else:
            with open(args.hparams_file) as f:
                hparams = yaml.load(f, Loader=yaml.loader)
            model = KoBARTConditionalGeneration.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams_file=args.hparams_file)
            
        trainer.fit(model, dm)
        
        
    elif args.mode == 'test':
        model = KoBARTConditionalGeneration.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams_file=args.hparams_file)
        trainer.test(model, dm)

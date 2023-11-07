import torch
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from utils.tools import get_lr


class Trainer:

    def __init__(self,
                 args = None,
                 model=None,
                 feature_extractor=None,
                 tokenizer=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None,
                 ):
        self.args = args
        if self.args is None:
            raise ValueError("args is None!")
            
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        
        if self.accelerator is not None:
            # if use accelerate, need to remove model.to(device).
            self.model = model
        else:
            self.model = model
            self.model.to(self.args.device)

        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")
        
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.scheduler = scheduler

    def train(self, train_data_loader=None, test_data_loader=None):
        if self.accelerator is not None:
            train_data_loader, test_data_loader, self.model, self.optimizer = self.accelerator.prepare(train_data_loader, 
                                                                                                       test_data_loader, 
                                                                                                       self.model, 
                                                                                                       self.optimizer)
        
        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0
            self.model.train()
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Epoch: {epoch}/{self.args.epochs}',
                      postfix=dict) as train_pbar:
                for step, batch in train_pbar:
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}

                    # backward, calculate gradient
                    if self.accelerator is not None:
                        with self.accelerator.autocast():
                            # forward
                            outputs = self.model(**batch)
                            loss = outputs.loss
                            self.accelerator.backward(loss)
                            if self.accelerator.sync_gradients:
                                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    else:
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()  # zero the gradient
                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    if self.accelerator is not None:
                        train_total_loss += self.accelerator.gather(loss).item()
                    else:
                        train_total_loss += loss.item()

                    train_pbar.set_postfix(
                        **{'train average loss': train_total_loss / (step + 1), 'train loss': loss.item(),
                           "lr": get_lr(self.optimizer)})
            # test
            if test_data_loader is not None:
                test_total_loss = 0
                with tqdm(enumerate(test_data_loader), total=len(test_data_loader),
                          desc=f'Epoch: {epoch}/{self.args.epochs}',
                          postfix=dict) as test_pbar:
                    self.model.eval()
                    for step, batch in test_pbar:
                        batch = {k: v.to(self.args.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss

                        # tqdm
                        test_total_loss += loss.item()
                        test_pbar.set_postfix(
                            **{'test average loss': test_total_loss / (step + 1), 'test loss': loss.item()})

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            self.model = self.accelerator.unwrap_model(self.model)
            
        self.model.save_pretrained(out_dir, torch_dtype=torch.float16)
        self.feature_extractor.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)

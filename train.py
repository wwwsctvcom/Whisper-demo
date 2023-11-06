import torch
from utils.trainer import Trainer
from utils.dataset import DataCollator, WhisperDataset
from torch.utils.data import DataLoader
from utils.tools import seed_everything
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration


class Arguments:

    def __init__(self):
        # model name or path
        self.model_name_or_path = "openai/whisper-small"

        # training arguments
        self.epochs = 1
        self.batch_size = 64
        self.lr = 2e-5
        self.lr_backbone = 1e-5
        self.weight_decay = 1e-4

        # dataset
        self.wav_data_path = "./data/chinese-single-speaker-speech-dataset"
        self.wav_transcript_path = "./data/chinese-single-speaker-speech-dataset/transcript.txt"
        self.num_workers = 12

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # args
    args = Arguments()

    # seed
    seed_everything()

    # loading model and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path,
                                                                language="chinese",
                                                                task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path,
                                                 language="chinese",
                                                 task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # loading dataset
    lines = open(args.wav_transcript_path, "r", encoding="utf-8").readlines()

    train_dataset = WhisperDataset(wav_data_path=args.wav_data_path,
                                   lines=lines,
                                   feature_extractor=feature_extractor,
                                   tokenizer=tokenizer,
                                   sampling_rate=16000)

    data_collator = DataCollator(feature_extractor, tokenizer)

    train_data_loader = DataLoader(train_dataset,
                                   collate_fn=data_collator,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           args.epochs * len(train_data_loader),
                                                           eta_min=0,
                                                           last_epoch=-1,
                                                           verbose=False)

    # start train
    trainer = Trainer(args=args,
                      model=model,
                      feature_extractor=feature_extractor,
                      tokenizer=tokenizer,
                      optimizer=optimizer,
                      scheduler=scheduler)
    trainer.train(train_data_loader=train_data_loader)

    # save model
    trainer.save_model("./whisper-small-finetuned")

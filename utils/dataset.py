import soundfile
from torch.utils.data import Dataset
from pathlib import Path


class WhisperDataset(Dataset):

    def __init__(self,
                 wav_data_path=None,
                 lines=None,
                 feature_extractor=None,
                 tokenizer=None,
                 sampling_rate=16000):
        self.wav_data_path = wav_data_path
        self.lines = lines
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        wav_path, wav_text = Path(self.wav_data_path) / line.split("|")[0], line.split("|")[1]

        # process wav text
        input_ids = self.tokenizer(wav_text, return_tensors="pt", add_special_tokens=True).input_ids

        # process wav data
        wav_array, sample_rate = soundfile.read(wav_path, dtype='float32')
        input_features = self.feature_extractor(wav_array, sampling_rate=self.sampling_rate,
                                                return_tensors="pt").input_features
        return {"input_features": input_features.squeeze(0), "labels": input_ids.squeeze(0)}


class DataCollator:

    def __init__(self, feature_extractor=None, tokenizer=None):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")  # pad the labels to max length

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

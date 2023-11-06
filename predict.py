import soundfile
from transformers import AutoProcessor, WhisperForConditionalGeneration


if __name__ == "__main__":
    # loading model and processor
    processor = AutoProcessor.from_pretrained("./whisper-small-finetuned")
    model = WhisperForConditionalGeneration.from_pretrained("./whisper-small-finetuned")


    # process wav data
    wav_path = "./data/chinese-single-speaker-speech-dataset/call_to_arms/call_to_arms_0001.wav"
    wav_array, sample_rate = soundfile.read(wav_path, dtype='float32')

    # feature extract
    inputs = processor.feature_extractor(wav_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)
    transcription = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("predict result: " + transcription)
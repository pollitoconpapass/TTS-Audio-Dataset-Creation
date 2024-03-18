import os
import torch  
import tempfile
import torchaudio
import demucs.api
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
from datasets import Dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor


DEMUCS_MODEL_NAME = "htdemucs_ft"
BATCH_SIZE = 8
FILE_LIMIT_MB = 10000
YT_LENGTH_LIMIT_S = 36000
HG_TOKEN = "<KEY>"

separator = demucs.api.Separator(model = DEMUCS_MODEL_NAME)


def separate_vocal(path):
    _, separated = separator.separate_audio_file(path)
    demucs.api.save_audio(separated["vocals"], path, samplerate=separator.samplerate)
    return path


def naive_postprocess_chunks(chunks, audio_array, sampling_rate,  stop_chars = ".,!:;?", min_duration = 5):
    # merge chunks as long as merged audio duration is lower than min_duration and that a stop character is not met
    # return list of dictionnaries (text, audio)
    # min duration is in seconds
    min_duration = int(min_duration * sampling_rate)


    new_chunks = []
    while chunks:
        current_chunk = chunks.pop(0)

        begin, end = current_chunk["timestamp"]
        begin, end = int(begin*sampling_rate), int(end*sampling_rate)

        current_dur = end-begin

        text = current_chunk["text"]


        chunk_to_concat = [audio_array[begin:end]]
        while chunks and (text[-1] not in stop_chars or (current_dur<min_duration)):
            ch = chunks.pop(0)
            begin, end = ch["timestamp"]
            begin, end = int(begin*sampling_rate), int(end*sampling_rate)
            current_dur += end-begin

            text = "".join([text, ch["text"]])
            chunk_to_concat.append(audio_array[begin:end])


        new_chunks.append({
            "text": text.strip(),
            "audio": np.concatenate(chunk_to_concat),
        })
        print(f"LENGTH CHUNK #{len(new_chunks)}: {current_dur/sampling_rate}s")

    return new_chunks


def stt_general(wav_file_path):
    try:
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        audio_data, original_sampling_rate = torchaudio.load(wav_file_path)
        resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)

        processor.tokenizer.set_target_lang("quz")  # -> you can change this depending on the language you want...
        model.load_adapter("quz")

        inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}
    

def split_audio_chunks(audio_file, chunk_duration=30):
    audio = AudioSegment.from_file(audio_file)
    chunk_length_ms = chunk_duration * 1000  # -> to miliseconds

    num_chunks = len(audio) // chunk_length_ms  # -> total chunks from the whole audio when cut every 30 s
    total_duration_seconds = len(audio) / 1000  # -> len(audio) will be in ms

    whole_transcription = ""
    chunks = []

    for i in range(num_chunks):
        # === TIMESTAMPS ===
        start_time = i * chunk_duration  # -> for going to 0-30, 30-60, etc
        end_time = min((i + 1) * chunk_duration, total_duration_seconds)
        chunk = audio[start_time * 1000: end_time * 1000]

        # === TRANSCRIPTIONS ===
        file = f"tmp/chunk_{i}.wav"
        chunk.export(file, format="wav")

        transcription_result = stt_general(file)
        os.remove(file)

        # === RESULTS ===
        whole_transcription += transcription_result["transcription"] + ", "
        chunks.append({"timestamp": (start_time, end_time),
                       "text": transcription_result["transcription"]})

    return {"whole_transcription": whole_transcription.strip(),
            "chunks": chunks}


# === IMPLEMENTING A NEW TRANSCRIBE METHOD FOR QUECHUA ===
def transcribe_definitivo(inputs_path, dataset_name):
    # Part of the original transcribe function:
    total_step = 4
    current_step = 0

    current_step += 1
    print(current_step, total_step)

    sampling_rate, inputs = wavfile.read(inputs_path)

    # GET THE TRANSCRIPTION:
    # out = pipe(inputs_path, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)
    # text = out["text"]
    # out = requests.post(endpoint_url, json={"file_path": inputs_path})
    result = split_audio_chunks(inputs_path)
    text = result["whole_transcription"]

    current_step += 1
    print(current_step, total_step)
    chunks = naive_postprocess_chunks(result["chunks"], inputs, sampling_rate)  # Q VR@ : out["chunks"]    30??

    current_step += 1
    print(current_step, total_step)

    transcripts = []
    audios = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, chunk in enumerate(chunks):
            arr = chunk["audio"]
            path = os.path.join(tmpdirname, f"{i}.wav")
            wavfile.write(path, sampling_rate,  arr)

            print(f"Separating vocals #{i}")
            path = separate_vocal(path)

            audios.append(path)
            transcripts.append(chunk["text"])

        dataset = Dataset.from_dict({"audio": audios, "text": transcripts}).cast_column("audio", Audio())

        dataset.push_to_hub(dataset_name, HG_TOKEN)

        local_filename = f"{dataset_name}.parquet"
        dataset.to_parquet(local_filename)
        print(f"Dataset saved as: {local_filename}")

    return [[transcript] for transcript in transcripts], text


# === MAIN ===
audio_file = "prueba.wav"
dataset_name = "dataset_quz_QC2X"

transcribe_definitivo(audio_file, dataset_name)

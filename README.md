# Text-To-Speech Dataset Creation

### Definition
This comes up for an alternative of the same TTS Dataset Creation made with Whisper v3, but for languages that the mentioned tool doesn't support. 

All of this is based on the mms-1b-all Facebook Speech-to-Text model: https://huggingface.co/facebook/mms-1b-all 


### Usage
1. Install all the dependencies

        pip install -r requirements.txt
2. Separately install the urllib package

        pip install 'urllib3<2'
3. Create a Hugging Face token and place it inside the HG_TOKEN part (ln. 17)

        HG_TOKEN = "<KEY>"
4. Inside the `stt_general` function you can change the language code for the one you like (ln. 75-76)

        processor.tokenizer.set_target_lang("quz") # -> cuzco quechua code
        model.load_adapter("quz")
5. Place your data in the calling function parameters
    - `audio_file`: path of your audio wav file (only works with .wav files)
    - `dataset_name`: Name you want for your dataset
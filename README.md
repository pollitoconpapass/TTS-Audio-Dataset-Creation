# Text-To-Speech Dataset Creation

### Definition
This comes up for an alternative of the [Create your own TTS dataset](https://huggingface.co/spaces/ylacombe/create-your-own-TTS-dataset) space from HuggingFace, but for languages that Whisper AI doesn't support. (Accepts only wav audio inputs NOT Youtube links)

All of this is based on the [facebook/mms-1b-all Speech-to-Text](https://huggingface.co/facebook/mms-1b-all )


### Usage
1. Install all the dependencies

        pip install -r requirements.txt

2. Separately install the urllib package

        pip install 'urllib3<2'

3. Create a `tmp` folder in the root of the project.

4. Create a Hugging Face token and paste it inside the console after running the commands:

        git config --global credential.helper store
        huggingface-cli login

5. Inside the `stt_general` function you can change the language code for the one you like (ln. 73-74). Check the supported languages code [here](https://huggingface.co/facebook/mms-1b-all#supported-languages)

        processor.tokenizer.set_target_lang("quz") # -> cuzco quechua code
        model.load_adapter("quz")

6. Place your data in the calling function parameters of ln.169-170 in `app.py`
    - `audio_file`: path of your audio wav file (only works with .wav files)
    - `dataset_name`: Name you want for your dataset

7. Run the project

        python app.py
# UniErase

1. Set up environment according to closer-look-LLM-Unlearning and EasyEdit
2. Download model into local directory using download_models.py
2. Perform unlearning token by running train_UNL.py (change model path)
3. Perform unlearning editing by running run_edit.py (make sure "test" is set to False)
make sure you have datasets==4.0.0
use this command:
NO_PROXY="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" no_proxy="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" HTTP_PROXY= http_proxy= HTTPS_PROXY= https_proxy= python run_edit.py
4. Evaluate by running xxx_eval.py
5. Run general ability evaluation via evaluate_utility.py

### We will provide detailed documentaion soon.

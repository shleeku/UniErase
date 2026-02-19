# UniErase

1. Set up environment according to closer-look-LLM-Unlearning and EasyEdit
2. Download model into local directory using download_models.py
3. Perform unlearning token by running train_UNL.py for FINAL stage only (change model path)
4. (If starting again after running run_edit_seq.py, delete the files in data/P_loc first)
5. Perform unlearning editing by running run_edit_seq.py for EACH stage (make sure "test" is set to False)
make sure you have datasets==4.0.0
use this command:
NO_PROXY="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" no_proxy="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" HTTP_PROXY= http_proxy= HTTPS_PROXY= https_proxy= python run_edit.py
6. Evaluate by running eval.py
7. Run general ability evaluation via evaluate_utility.py?

OR use run.sh to run each stage in order

### We will provide detailed documentaion soon.

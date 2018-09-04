`classification.py` script provides a finetuned Xception model, which is able to distinguish a bird on the RGB image among 50 classes, 
and helper functions so as to check model performance.
- Accuracy on train dataset is 0.9108
- Accuracy on test dataset is 0.8160

## Usage:
1. Download public data (train dataset) [here](https://cloud.mail.ru/public/Ft4T/vqvxvmkYQ) and unzip it.
2. Execute `sudo python3 run_tests.py public_data`
3. Wait until model is evaluated and get information about accuracy. 
This may require up to 120 minutes based on performance of your machine. External GPU is recommended.

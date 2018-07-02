Improved gan on Imagenet in tensorflow eager

Download imagenet data using the script imagenet_to_gcs.py
Edit the filepaths and run bash train.sh

The model is unstable and is sensitive to initialization and hyper-parameters. 

Work to be done:

Virtual batch normalization
Historical averaging of gradients
Multi-gpu run
Incepttion score and FID

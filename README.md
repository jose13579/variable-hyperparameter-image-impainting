# Variable Hyperparameter Efficient Visual Transformer for Image Inpainting

![VHII](images/proposal.png?raw=true)

<!-- ---------------------------------------------- -->
## Installation  

Clone this repo.

```
git clone https://github.com/jose13579/variable-hyperparameter-image-impainting.git
cd variable-hyperparameter-image-impainting/
```

We build our project based on Pytorch and Python libraries. To train and test our project, we suggest create a Conda environment from the provided YAML, e.g.

```
conda env create -f environment.yml 
conda activate vhii
```

Or use the Dockerfile to install conda and prerequisites, e.g.

```
Docker build -t VHII-image .
```

<!-- ---------------------------------------------- -->
## Dataset Preparation

We prepare the real and masks datasets. 

**Preparing CelebA Dataset.** The dataset can be downloaded from [here](https://competitions.codalab.org/competitions/19544#participate-get-data). This dataset contains more than 200,000 large-scale facial celebrity images. We adopt 162,770 for training and 19,961 for testing. The dataset should be arranged in the same directory structure as 

```
datasets
    ｜- celeba
        |- JPEGImages
           |- <video_id>.zip
           |- <video_id>.zip
        |- test.json 
        |- train.json 
``` 

**Preparing Places365 Dataset.** The dataset can be downloaded from [here](https://davischallenge.org/davis2017/code.html). The training set has approximately 1.8 million images from 365 scene categories, where there are at most 5,000 images per category. For testing, we adopt the
original validation set. The dataset should be arranged in the same directory structure as

```
datasets
    ｜- places
        |- JPEGImages
          |- cows.zip
          |- goat.zip
        |- Annoatations
          |- cows.zip
          |- goat.zip
        |- test.json 
        |- train.json 
``` 

**Preparing Paris Street View (PSV) Dataset.** The dataset can be downloaded from [here](https://davischallenge.org/davis2017/code.html). The training and test set includes 14,900 and 100 images, respectively. This dataset was collected from the street views of Paris, taking a large
number of buildings, and structure information, such as windows and doors. The dataset should be arranged in the same directory structure as

```
datasets
    ｜- psv
        |- JPEGImages
          |- cows.zip
          |- goat.zip
        |- Annoatations
          |- cows.zip
          |- goat.zip
        |- test.json 
        |- train.json 
``` 


<!-- ---------------------------------------------- -->
## Training New Models
Once the dataset is prepared, new models can be trained with the following commands:  

```
bash run_train.sh --train_config_file
```
For example:

```
bash run_train.sh configs/psv_proposal_efficient_128_64_32_16_channels.json
```
<!-- ---------------------------------------------- -->

<!-- ---------------------------------------------- -->
## Testing
To test the models 

1. Download the trained models, and save they in ```trained_models/```. 

2. Run the test bash file to evaluate/test the trained model. 

```
bash run_test_dataset.sh --model_name --model_path --seed --real_dataset_path --mask_dataset_path --output_dataset_path 
```

For example:
```
bash run_test_dataset.sh "VHII" "trained_models/celeba/celeba_VHII_efficient/gen_00050.pth" 0 "/data/celeba/celeba_dataset/test/" "/data/pconv/test_mask/20-30/" "test_output_datasets/trained_celeba_VHII_efficient_seed_0"
```
The outputs inpainted images are saved at ```test_output_datasets/```.  

<!-- ---------------------------------------------- -->

## Image Demo

To inference a single image like this:
```
bash run_test_image.sh --model_name --model_path --input_path --mask_path --output_path --output_name
```

For example:

```
bash run_test_image.sh "VHII" trained_models/celeba/celeba_VHII_efficient/gen_00050.pth "examples/img/100_000100_gt.png" "examples/mask/100_000100_mask.png" "examples/output" "100_000100_output"
```

<!-- ---------------------------------------------- -->
## Contact
If you have any questions or suggestions about this paper, feel free to contact me (j209820@dac.unicamp.br).

# All Data Are Ext -- the Winning Model of the SIIM/ISIC 2020 Grand Challenge
This repository contains the code necessary to run the inference part of the
ADAE algorithm. It was created by Qishen Ha, Bo Liu and colleagues from NVIDIA
as their submission for the SIIM/ISIC 2020 Grand Challenge hosted by Kaggle.

## Original Repository
The original repository of the submission can be found here:
https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution

## Hardware
The code was edited to allow running the prediction on NVIDIA (CUDA)
GPUs, as well as CPU.

To run the GPU version, please ensure that all necessary drivers for
Torch are installed (those are not part of the requirements!).

## Data and model setup (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
By default, the prediction code (in predict.py) uses the ./data folder as
input and the ./weights folder for the model files.

Before using the code, please issue a call of

``predict.py --download-models``

to download the necessary PTH model files. For this to work, you must already
have an account at Kaggle, and have generated an API token for the python API
to be able to download files!

## Predicting
To run inference, use the ``predict.py`` script. If no CSV file is given as
argument, the code continually scans the folder given in --data-dir for new
CSV files, which will then be processed, and then renamed (with an extension
of "*.done"). If a CSV file is given, the file will not be renamed, and the
program will exit after processing the referenced images.

Should the 3 resized versions of an image not exist, they will be created
(takes additional time).

### Usage
```
usage: predict.py [-h] [-b BATCH_SIZE] [-c CSV_FILE] [-C CUDA_VISIBLE_DEVICES]
                  [-d DATA_DIR] [-D DEVICE] [--download-models] [-e ENSEMBLE]
                  [-f FOLDS] [-g GAUSS_KERNEL] [-H HEATMAP_DIR] [-l]
                  [-L LOG_ODDS_DIR] [-m MODELS] [-M MODEL_DIR] [-n N_TEST]
                  [-o OUTPUTS] [-O OVERLAYS] [-p] [-P PROBMAP_DIR] [-s]
                  [-S SUB_DIR] [-t TARGET] [-T LOG_ODDS_TERMS]
                  [--thresh-heat-mean THRESH_HEAT_MEAN]
                  [--thresh-heat-t THRESH_HEAT_T]
                  [--thresh-logodds-mean THRESH_LOGODDS_MEAN]
                  [--thresh-logodds-t THRESH_LOGODDS_T]
                  [--thresh-prob-mean THRESH_PROB_MEAN]
                  [--thresh-prob-t THRESH_PROB_T] [-w NUM_WORKERS]
                  [-z MAP_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Set batch size (images run at once, default=16)
  -c CSV_FILE, --csv-file CSV_FILE
                        CSV file (default: scans for *.csv, then renames)
  -C CUDA_VISIBLE_DEVICES, --CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
                        CUDA_VISIBLE_DEVICES setting (default="0")
  -d DATA_DIR, --data-dir DATA_DIR
                        Directory for images (and CSVs, default=./data)
  -D DEVICE, --device DEVICE
                        Torch device to use (default=auto, use GPU if present)
  --download-models     Download models from kaggle (requires user account !)
  -e ENSEMBLE, --ensemble ENSEMBLE
                        Average created CSV files (default=90, 0 to disable)
  -f FOLDS, --folds FOLDS
                        Folds to run (default=0,1,2,3,4 ; -f list to show)
  -g GAUSS_KERNEL, --gauss-kernel GAUSS_KERNEL
                        Gaussian smoothing kernel for maps (default=1.2)
  -H HEATMAP_DIR, --heatmap-dir HEATMAP_DIR
                        Stores heatmaps in given folder (default:None)
  -l, --log-avg         Average n-test passes as log-probs (default=True)
  -L LOG_ODDS_DIR, --log-odds-dir LOG_ODDS_DIR
                        Stores log-odds maps in given folder (default:None)
  -m MODELS, --models MODELS
                        Models to evaluate (default:all ; -m list to show)
  -M MODEL_DIR, --model-dir MODEL_DIR
                        Folder with model (*.pth) files
                        (default=/home/jochenw/test/weights)
  -n N_TEST, --n-test N_TEST
                        Evaluate flipped/rotated versions (default=8)
  -o OUTPUTS, --outputs OUTPUTS
                        Targets to output (default=melanoma, -o list to show)
  -O OVERLAYS, --overlays OVERLAYS
                        Generate overlays (default=None, -O list to show)
  -p, --preload-models  Preload models (only in scanning mode, default=False)
  -P PROBMAP_DIR, --probmap-dir PROBMAP_DIR
                        Stores probmaps in given folder (default:None)
  -s, --silent          Silent mode (do not print anything, default=False)
  -S SUB_DIR, --sub-dir SUB_DIR
                        Folder to store submission CSVs in (default=./subs)
  -t TARGET, --target TARGET
                        Output that is the target (default=melanoma)
  -T LOG_ODDS_TERMS, --log-odds-terms LOG_ODDS_TERMS
                        Log-odds map terms (default:outputs, -o list to show)
  --thresh-heat-mean THRESH_HEAT_MEAN
                        Threshold for heat (mean) maps (default=5.0)
  --thresh-heat-t THRESH_HEAT_T
                        Threshold for heat (mean) maps (default=3.0)
  --thresh-logodds-mean THRESH_LOGODDS_MEAN
                        Threshold for heat (mean) maps (default=2.0)
  --thresh-logodds-t THRESH_LOGODDS_T
                        Threshold for heat (mean) maps (default=3.0)
  --thresh-prob-mean THRESH_PROB_MEAN
                        Threshold for heat (mean) maps (default=0.2)
  --thresh-prob-t THRESH_PROB_T
                        Threshold for heat (mean) maps (default=3.0)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of worker threads (default=16)
  -z MAP_SIZE, --map-size MAP_SIZE
                        Output map size, X-by-X pixels (default=84 [56..512])
```

## Example
This repository contains one example image, ``ISIC_0074542``, together with the
metadata. It is the first image of the challenge dataset coming from Memorial Sloan
Kettering Cancer Center.

The resulting files were generated with this command line

``./predict.py -d ./data -H ./heatmaps -L ./logodds -P ./probmaps -S ./subs -c ./data/ADAE_example.csv -w 0``

## Troubleshooting
If you are doing inference using the CPU as the device, you may have to set the
number of workers to 0, to prevent a Torch issue from stopping program execution:

https://github.com/pytorch/pytorch/issues/46409


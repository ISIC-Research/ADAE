#!/usr/bin/env python

# global imports
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

# helper wrapper to suppress tqdm output if silent
def silent_tqdm(genvar, *genargs, **genkwargs):
    for o in genvar:
        yield o


# configuration settings
_base_folder = os.path.dirname(os.path.abspath(__file__))

# - columns to check for input CSVs (and some renaming settings)
_check_cols = [
    'image_name',
    'patient_id',
    'sex',
    'age_approx',
    'anatom_site_general_challenge',
]
# NB: width and height will be extracted from original images if not given!
_rename_cols = {
    'anatom_site_general': 'anatom_site_general_challenge',
    'age': 'age_approx',
}

# - list of models with options
_used_models = { # 'name': [data_folder, image_size, enet_type, out_dim, use_meta, n_meta_dim],
    '4c_b5ns_1.5e_640_ext_15ep':        [ 768, 640, 'tf_efficientnet_b5_ns', 4, False, '512,128', 2048],
    '9c_b4ns_2e_896_ext_15ep':          [1024, 896, 'tf_efficientnet_b4_ns', 9, False, '512,128', 1792],
    '9c_b4ns_448_ext_15ep-newfold':     [ 512, 448, 'tf_efficientnet_b4_ns', 9, False, '512,128', 1792],
    '9c_b4ns_768_640_ext_15ep':         [ 768, 640, 'tf_efficientnet_b4_ns', 9, False, '512,128', 1792],
    '9c_b4ns_768_768_ext_15ep':         [ 768, 768, 'tf_efficientnet_b4_ns', 9, False, '512,128', 1792],
    '9c_b5ns_1.5e_640_ext_15ep':        [ 768, 640, 'tf_efficientnet_b5_ns', 9, False, '512,128', 2048],
    '9c_b5ns_448_ext_15ep-newfold':     [ 512, 448, 'tf_efficientnet_b5_ns', 9, False, '512,128', 2048],
    '9c_b6ns_448_ext_15ep-newfold':     [ 512, 448, 'tf_efficientnet_b6_ns', 9, False, '512,128', 2304],
    '9c_b6ns_576_ext_15ep_oldfold':     [ 768, 576, 'tf_efficientnet_b6_ns', 9, False, '512,128', 2304],
    '9c_b6ns_640_ext_15ep':             [ 768, 640, 'tf_efficientnet_b6_ns', 9, False, '512,128', 2304],
    '9c_b7ns_1e_576_ext_15ep_oldfold':  [ 768, 576, 'tf_efficientnet_b7_ns', 9, False, '512,128', 2560],
    '9c_b7ns_1e_640_ext_15ep':          [ 768, 640, 'tf_efficientnet_b7_ns', 9, False, '512,128', 2560],
    '9c_meta128_32_b5ns_384_ext_15ep':  [ 512, 384, 'tf_efficientnet_b5_ns', 9, True , '128,32' , 2048],
    '9c_meta_1.5e-5_b7ns_384_ext_15ep': [ 512, 384, 'tf_efficientnet_b7_ns', 9, True , '512,128', 2560],
    '9c_meta_b3_768_512_ext_18ep':      [ 768, 512, 'efficientnet_b3',       9, True , '512,128', 1536],
    '9c_meta_b4ns_640_ext_15ep':        [ 768, 640, 'tf_efficientnet_b4_ns', 9, True , '512,128', 1792],
    '9c_nest101_2e_640_ext_15ep':       [ 768, 640, 'resnest101',            9, False, '512,128', 2048],
    '9c_se_x101_640_ext_15ep':          [ 768, 640, 'seresnext101',          9, False, '512,128', 2048],
}
_used_data_folders = set(sorted([m[0] for m in _used_models.values()]))
_used_image_sizes = set(sorted([m[1] for m in _used_models.values()]))
_full_model_list = []
for m in _used_models.keys():
    for f in range(5):
        _full_model_list.append(f'{m}_best_fold{f}')

# - hard-code meta features as needed
_used_meta_features = [
    'sex',
    'age_approx',
    'n_images',
    'image_size',
    'site_anterior torso',
    'site_head/neck',
    'site_lateral torso',
    'site_lower extremity',
    'site_oral/genital',
    'site_palms/soles',
    'site_posterior torso',
    'site_torso',
    'site_upper extremity',
    'site_nan',
]

# hard-coded melanoma diagnosis output index
_diags_full = ['ak', 'bcc', 'bkl', 'df', 'scc', 'vasc', 'melanoma', 'nevus', 'unknown']
_diags_part = ['bkl', 'melanoma', 'nevus', 'unknown']
_outidx_full = {dc: di for di, dc in enumerate(_diags_full)}
_outidx_part = {dc: di for di, dc in enumerate(_diags_part)}

# preset options
opts = {
    'csv_file': '*.csv',
    'data_dir': './data',
    'device_name': 'cpu',
    'embeddings': False,
    'embeddings_dir': './embeddings',
    'ensemble': 90,
    'folds': [0,1,2,3,4],
    'gauss_kernel': 1.2,
    'heatmaps': False,
    'heatmap_dir': './heatmaps',
    'keep_ensembled': False,
    'log_avg': False,
    'log_odds': [],
    'log_odds_dir': './logodds',
    'map_size': 84,
    'min_p': 1.0e-9,
    'model_dir': './weights',
    'n_test': 8,
    'outputs': ['melanoma'],
    'preload': False,
    'probmaps': False,
    'probmap_dir': './probmaps',
    'select_radius': 0.0,
    'silent': False,
    'store_raw': False,
    'sub_dir': './subs',
    'target': 'melanoma',
    'use_models': list(_used_models.keys()),
    'weight_radius': 0.0,
}


# parse arguments (raw)
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=16,
        help='Set batch size (images run at once, default=16)')
    parser.add_argument('-c', '--csv-file', type=str, default='*.csv',
        help='CSV file (default: scans for *.csv, then renames)')
    parser.add_argument('-C', '--CUDA_VISIBLE_DEVICES', type=str, default='0',
        help='CUDA_VISIBLE_DEVICES setting (default="0")')
    parser.add_argument('-d', '--data-dir', type=str, default='./data',
        help='Directory for images (and CSVs, default=./data)')
    parser.add_argument('-D', '--device', type=str, default='auto',
        help='Torch device to use (default=auto, use GPU if present)')
    parser.add_argument('--download-models', action='store_const', const=True,
        help='Download models from kaggle (requires user account !)')
    parser.add_argument('-e', '--ensemble', type=int, default=90,
        help='Average created files (default=90, 0 to disable)')
    parser.add_argument('-E', '--embeddings-dir', type=str,
        help='Stores embeddings in given folder (default:None)')
    parser.add_argument('-f', '--folds', type=str, default='0,1,2,3,4',
        help='Folds to run (default=0,1,2,3,4 ; -f list to show)')
    parser.add_argument('-g', '--gauss-kernel', type=float, default=1.2,
        help='Gaussian smoothing kernel for maps (default=1.2)')
    parser.add_argument('-H', '--heatmap-dir', type=str,
        help='Stores heatmaps in given folder (default:None)')
    parser.add_argument('-k', '--keep-ensembled', action='store_const', const=True,
        help='Keep files after ensembling (default=False)')
    parser.add_argument('-l', '--log-avg', action='store_const', const=True,
        help='Average n-test passes as log-probs (default=True)')
    parser.add_argument('-L', '--log-odds-dir', type=str,
        help='Stores log-odds maps in given folder (default:None)')
    parser.add_argument('-m', '--models', type=str, default='all',
        help='Models to evaluate (default:all ; -m list to show)')
    parser.add_argument('-M', '--model-dir', type=str, default='./weights',
        help='Folder with model (*.pth) files (default=' + _base_folder + '/weights)')
    parser.add_argument('-n', '--n-test', type=int, default=8,
        help='Evaluate flipped/rotated versions (default=8)')
    parser.add_argument('-o', '--outputs', type=str, default='melanoma',
        help='Targets to output (default=melanoma, -o list to show)')
    parser.add_argument('-p', '--preload-models', action='store_const', const=True,
        help='Preload models (only in scanning mode, default=False)')
    parser.add_argument('-P', '--probmap-dir', type=str,
        help='Stores probmaps in given folder (default:None)')
    parser.add_argument('-r', '--store-raw', action='store_const', const=True,
        help='Stores the raw output predictions (default:False)')
    parser.add_argument('-R', '--select_radius', type=float, default=0.0,
        help='Select circular part of the feature maps with radius R (default:0.0, no selection)')
    parser.add_argument('-s', '--silent', action='store_const', const=True,
        help='Silent mode (do not print anything, default=False)')
    parser.add_argument('-S', '--sub-dir', type=str, default='./subs',
        help='Folder to store submission CSVs in (default=./subs)')
    parser.add_argument('-t', '--target', type=str, default='melanoma',
        help='Output that is the target (default=melanoma)')
    parser.add_argument('-T', '--log-odds-terms', type=str,
        help='Log-odds map terms (default:outputs, -o list to show)')
    parser.add_argument('-w', '--num-workers', type=int, default=16,
        help='Number of worker threads (default=16)')
    parser.add_argument('-W', '--weight_radius', type=float, default=0.0,
        help='Weight smoothing kernel for circular selection part (default:0.0, no smoothing)')
    parser.add_argument('-z', '--map-size', type=int, default=84,
        help='Output map size, X-by-X pixels (default=84 [56..512])')

    args, _ = parser.parse_known_args()
    return args

# process args (set into opts, and check, etc.)
def process_args(args, download_models):
    opts['model_dir'] = args.model_dir
    if opts['model_dir'] == './weights':
        opts['model_dir'] = _base_folder + '/weights'
    if not os.path.exists(opts['model_dir']):
        if not download_models:
            if not opts['silent']:
                download_models = input('Download models (y/n)? ').lower() == 'y'
        if download_models:
            if not opts['silent']:
                print(f'Creating {opts["model_dir"]} for model weights.')
            os.makedirs(opts['model_dir'])
        else:
            print('Directory in --model-dir not found.')
            return 1
    elif not os.path.isdir(opts['model_dir']):
        print(f'{opts["model_dir"]} must be a directory (in --model-dir).')
        return 1
    
    # download models?
    found_models = list(sorted(glob.glob(opts['model_dir'] + os.path.sep + '*.pth')))
    if download_models and len(found_models) < 90:
        if not kaggle_download_models():
            return 1
        found_models = list(sorted(glob.glob(opts['model_dir'] + os.path.sep + '*.pth')))

    # all models available?
    if len(found_models) < 90:
        print('Directory in --model-dir does not contain the required (90) files.')
        return 1
    
    # check for models first
    opts['use_models'] = args.models.split(',')
    if opts['use_models'][0] == 'all':
        opts['use_models'] = list(_used_models.keys())
    elif opts['use_models'][0] == 'none':
        opts['use_models'] = []
    elif opts['use_models'][0] == 'list':
        if not opts['silent']:
            print('Available models (comma separated):')
        for m in _used_models.keys():
            print(f' - {m}')
        return 1
    else:
        for m in opts['use_models']:
            if not m in _used_models:
                print(f'Unknown model {m} in --models argument.')
                return 1
    
    # then folds
    opts['folds'] = args.folds
    if opts['folds'] == 'list':
        if not opts['silent']:
            print('Available folds: 0,1,2,3,4 (comma separated)')
        else:
            for m in range(5):
                print(m)
        return 1
    try:
        opts['folds'] = [int(f) for f in opts['folds'].split(',')]
    except:
        raise RuntimeError('Invalid --folds argument.')
    for f in opts['folds']:
        if not f in [0, 1, 2, 3, 4]:
            print(f'Invalid fold {f} in --folds argument.')
            return 1
    
    # then outputs
    opts['outputs'] = args.outputs.lower()
    if opts['outputs'] == 'list':
        if not opts['silent']:
            print('Available outputs (comma separated):')
        for o in _diags_full:
            print(f' - {o}')
        return 1
    opts['outputs'] = opts['outputs'].split(',')
    for o in opts['outputs']:
        if not o in _diags_full:
            print(f'Output {o} not in list of diagnoses.')
            return 1
    
    # then log-odds
    opts['log_odds'] = args.log_odds_terms.lower() if args.log_odds_terms else []
    opts['log_odds_dir'] = args.log_odds_dir
    if opts['log_odds'] == 'list':
        if not opts['silent']:
            print('Available --log-odds-terms options (comma separated):')
        print(' - melanoma~nevus = log(P(melanoma) / P(nevus))')
        print(' - melanoma~other = log(P(melanoma) / P(not(melanoma)))')
        print(' - etc.')
        if not opts['silent']:
            print(' - available diagnosis terms:')
            for dt in sorted(_diags_full):
                print(f'   - {dt}')
        return 1
    if opts['log_odds_dir']:
        os.makedirs(opts['log_odds_dir'], exist_ok=True)
        
        # auto-set log-odds terms
        if not opts['log_odds']:
            opts['log_odds'] = [f'{loterm}~other' for loterm in opts['outputs']]
            opts['log_odds'] = ','.join(opts['log_odds'])
            if not opts['silent']:
                print(f'Auto-selected log-odds terms: {opts["log_odds"]}')
    if opts['log_odds']:
        if not args.log_odds_dir:
            print('Log-odds terms require --log-odds-dir to be set')
            return 1
        opts['log_odds'] = opts['log_odds'].split(',')
    for li, lo in enumerate(opts['log_odds']):
        lop = lo.split('~')
        if len(lop) != 2:
            print(f'Invalid number of particles in --log-odds expression {lo}.')
            return 1
        if lop[0] != 'other' and lop[0] not in _diags_full:
            print(f'Invalid first particle in --log-odds expression {lop[0]}.')
            return 1
        if (lop[1] != 'other' and lop[1] not in _diags_full) or lop[0] == lop[1]:
            print(f'Invalid second particle in --log-odds expression {lop[1]}.')
            return 1
        opts['log_odds'][li] = lop
    
    opts['csv_file'] = args.csv_file
    opts['data_dir'] = args.data_dir
    if opts['data_dir'][-1] == os.path.sep:
        opts['data_dir'] = opts['data_dir'][:-1]
    if not os.path.exists(opts['data_dir']):
        if opts['data_dir'] == './data' or opts['data_dir'] == 'data':
            opts['data_dir'] = os.path.join(_base_folder, 'data')
        if not opts['silent']:
            if input('Create ./data as directory for --data-dir (y/n): ').lower() == 'y':
                os.makedirs(opts['data_dir'])
        if not os.path.exists(opts['data_dir']):
            print('Directory in --data-dir not found.')
            return 1
    elif not os.path.isdir(opts['data_dir']):
        print(f'{opts["data_dir"]} must be a directory (in --data-dir)')
    opts['device_name'] = args.device if args.device in ['auto', 'cpu', 'cuda'] else 'cpu'
    opts['embeddings_dir'] = args.embeddings_dir
    if opts['embeddings_dir']:
        os.makedirs(opts['embeddings_dir'], exist_ok=True)
        opts['embeddings'] = True
    opts['ensemble'] = args.ensemble
    if opts['ensemble'] < 0:
        opts['ensemble'] = 0
    opts['gauss_kernel'] = args.gauss_kernel
    if opts['gauss_kernel'] < 1.0:
        if opts['gauss_kernel'] != 0.0:
            opts['gauss_kernel'] = 1.0
    elif opts['gauss_kernel'] > 6.0:
        opts['gauss_kernel'] = 6.0
    opts['heatmap_dir'] = args.heatmap_dir
    if opts['heatmap_dir']:
        os.makedirs(opts['heatmap_dir'], exist_ok=True)
        opts['heatmaps'] = True
    opts['keep_ensembled'] = True if args.keep_ensembled else False
    opts['log_avg'] = True if args.log_avg else False
    opts['map_size'] = args.map_size
    if opts['map_size'] < 56:
        if opts['map_size'] != 0:
            opts['map_size'] = 56
    elif opts['map_size'] > 512:
        opts['map_size'] = 512
    opts['n_test'] = args.n_test
    if opts['n_test'] < 1:
        opts['n_test'] = 1
    opts['preload'] = True if args.preload_models else False
    opts['probmap_dir'] = args.probmap_dir
    if opts['probmap_dir']:
        os.makedirs(opts['probmap_dir'], exist_ok=True)
        opts['probmaps'] = True
    opts['store_raw'] = True if args.store_raw else False
    opts['sub_dir'] = args.sub_dir
    os.makedirs(opts['sub_dir'], exist_ok=True)
    opts['target'] = args.target.lower()
    if not opts['target'] in opts['outputs']:
        if not opts['target'] in _diags_full:
            print(f'Target {opts["target"]} not in list of diagnoses.')
            return 1
        opts['outputs'].append(opts['target'])
    opts['select_radius'] = args.select_radius
    opts['weight_radius'] = args.weight_radius

# delete files
def delete_files(fs):
    for f in fs:
        try:
            os.remove(f)
        except:
            pass

# selection radius
def selection_radius(image_size, device):

    # import torch late
    import torch

    # create tensor map and pass through network to gauge
    tssize = int(image_size / 32)
    tsmap = torch.zeros(tssize, tssize)
    tsctr = float(tssize) / 2.0 - 0.5
    tsrad = opts['select_radius'] * tsctr * tsctr + 0.25
    for tsx in range(tssize):
        tspx = tsctr - float(tsx)
        for tsy in range(tssize):
            tspy = tsctr - float(tsy)
            tspxy = tspx * tspx + tspy * tspy
            if tspxy < tsrad:
                tsmap[tsx, tsy] = 1.0
    # scale tensor map
    tsmap *= (float(torch.numel(tsmap)) / torch.sum(tsmap))
    tsmap = tsmap.to(device)
    return tsmap

# mean + std of maps
def mean_std_maps(m, dolog=False):
    p = [None] * len(m)
    for mi, mn in enumerate(m):
        p[mi] = np.load(mn)['arr_0']
        if dolog:
            p[mi] = np.log(p[mi])
        if mi == 0:
            pm = p[mi]
            pv = p[mi] * p[mi]
        else:
            pm += p[mi]
            pv += p[mi] * p[mi]
        if dolog:
            p[mi] = np.exp(p[mi])
    pm /= len(m)
    pv = (pv - len(m) * pm * pm) / (len(m) - 1)
    ps = np.sqrt(pv)
    if dolog:
        pl = np.exp(pm-ps)
        pu = np.exp(pm+ps)
        pm = np.exp(pm)
        ps = 0.5 * (pu - pl)
    return pm, ps

# ensemple CSVs
def ensemble_csvs(df_name):
    import pandas as pd
    if opts['silent']:
        utqdm = silent_tqdm
    else:
        utqdm = tqdm
    
    # look for CSV files
    to_ensemble = list(sorted(glob.glob(f'{opts["sub_dir"]}{os.path.sep}{df_name}{os.path.sep}sub_*fold*.csv')))
    if len(to_ensemble) == opts['ensemble']:
        if not opts['silent']:
            print(f' - averaging {opts["ensemble"]} CSV files in {opts["sub_dir"]}/{df_name}...')
        target_avg = {}
        target_num = {}
        output_dfs = {}
        for o in opts['outputs']:
            output_dfs[o] = None
        for lidx, load_csv in utqdm(enumerate(to_ensemble), desc='Averaging CSV files'):
            model_name = os.path.splitext(os.path.basename(load_csv))[0][4:]
            df_part = pd.read_csv(load_csv).set_index('image_name')
            for o in opts['outputs']:
                if output_dfs[o] is None:
                    output_dfs[o] = df_part.copy()
                    output_dfs[o].drop(list(output_dfs[o].columns), axis=1)
                oo = o
                if o == opts['target']:
                    o = 'target'
                if not o in target_num:
                    target_num[o] = 0
                    target_avg[o] = np.zeros((df_part.shape[0]), dtype=np.float64)
                if not o in df_part.columns:
                    continue
                target_var = df_part[o].values
                output_dfs[oo][model_name] = target_var
                if opts['log_avg']:
                    target_var = np.log(target_var)
                target_avg[o] += target_var
                target_num[o] += 1
        for tkey, tnum in target_num.items():
            target_avg[tkey] /= tnum
            if opts['log_avg']:
                target_avg[tkey] = np.exp(target_avg[tkey])
            df_part[tkey] = target_avg[tkey]
            if tkey == 'target':
                okey = opts['target']
            else:
                okey = tkey
            output_dfs[okey]['target'] = df_part[tkey].values
            if opts['log_avg']:
                output_dfs[okey].to_csv(os.path.join(opts['sub_dir'], df_name, f'sub_output_{okey}_logavg.csv'))
            else:
                output_dfs[okey].to_csv(os.path.join(opts['sub_dir'], df_name, f'sub_output_{okey}_average.csv'))
        if opts['log_avg']:
            df_part[list(target_num.keys())].to_csv(os.path.join(opts['sub_dir'], df_name, 'sub_allfolds_logavg.csv'))
        else:
            df_part[list(target_num.keys())].to_csv(os.path.join(opts['sub_dir'], df_name, 'sub_allfolds_average.csv'))
        if not opts['keep_ensembled']:
            delete_files(to_ensemble)

# ensemble maps
def ensemble_maps(image_name):
    if opts['heatmaps']:
        for o in opts['outputs']:
            mapfiles = glob.glob(f'{opts["heatmap_dir"]}{os.path.sep}{image_name}{os.path.sep}hmap_{o}_*_fold*.npz')
            if len(mapfiles) != opts['ensemble']:
                if not o in _diags_part:
                    if opts['ensemble'] == 90 and len(mapfiles) < 85:
                        return False
                else:
                    return False
            mmap, smap = mean_std_maps(mapfiles)
            np.savez(os.path.join(opts['heatmap_dir'], image_name, f'hmap_{o}_mean.npz'), mmap, smap)
            if not opts['keep_ensembled']:
                delete_files(mapfiles)
    if opts['probmaps']:
        for o in opts['outputs']:
            mapfiles = glob.glob(f'{opts["probmap_dir"]}{os.path.sep}{image_name}{os.path.sep}pmap_{o}_*_fold*.npz')
            if len(mapfiles) != opts['ensemble']:
                if not o in _diags_part:
                    if opts['ensemble'] == 90 and len(mapfiles) < 85:
                        return False
                else:
                    return False
            mmap, smap = mean_std_maps(mapfiles, True)
            np.savez(os.path.join(opts['probmap_dir'], image_name, f'pmap_{o}_mean.npz'), mmap, smap)
            if not opts['keep_ensembled']:
                delete_files(mapfiles)
    if opts['log_odds']:
        for lo in opts['log_odds']:
            lo_name = f'{lo[0]}~{lo[1]}'
            mapfiles = glob.glob(f'{opts["log_odds_dir"]}{os.path.sep}{image_name}{os.path.sep}lomap_{lo_name}_*_fold*.npz')
            if len(mapfiles) != opts['ensemble']:
                if not lo[0] in _diags_part:
                    if opts['ensemble'] == 90 and len(mapfiles) < 85:
                        return False
                else:
                    return False
            mmap, smap = mean_std_maps(mapfiles)
            np.savez(os.path.join(opts['log_odds_dir'], image_name, f'lomap_{lo_name}_mean.npz'), mmap, smap)
            if not opts['keep_ensembled']:
                delete_files(mapfiles)
    return True

# download models form kaggle
# see https://stackoverflow.com/questions/55934733/documentation-for-kaggle-api-within-python
def kaggle_download_models():
        
    # already downloaded?
    downloaded_files = glob.glob(os.path.join(opts['model_dir'], '*.pth'))
    if len(downloaded_files) == 90:
        return True
    
    # does the user have a ~/.kaggle/kaggle.json file?
    try:
        home_folder = os.path.expanduser('~')
        kaggle_json = os.path.join(home_folder, '.kaggle', 'kaggle.json')
        if not os.path.exists(kaggle_json):
            if opts['silent']:
                print('Cannot request username/API key in silent mode.')
                return False
            username = input('Kaggle username: ')
            if username == '':
                print('No username given.')
                return False
            apikey = input('Kaggle API key: ')
            if len(apikey) != 32:
                print('Invalid API key length (32 characters).')
                return False
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = apikey
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f'Could not log into kaggle: {str(e)}')
        return False
    
    # download models dataset
    try:
        api.dataset_download_files('boliu0/melanoma-winning-models',
                                   path=opts['model_dir'], quiet=False, unzip=True)
    except Exception as e:
        print(f'Could not download data from kaggle: {str(e)}')
        return False
    return True

# process embeddings
def process_emb(e, mname, imnames, si):
    if opts['n_test'] > 1:
        e /= opts['n_test']
    e = e.detach().cpu().numpy()
    esh = e.shape
    for iidx in range(esh[0]):
        edir = os.path.join(opts['embeddings_dir'], f'{imnames[si+iidx]}')
        os.makedirs(edir, exist_ok=True)
        np.savez(os.path.join(edir, f'emb_{mname[:-4]}.npz'), e[iidx,:].squeeze())

# process maps and outputs
def process_maps(m, mname, oimap, imnames, si):
    from scipy import ndimage
    import torch
    if opts['n_test'] > 1:
        m /= opts['n_test']
    m = m.detach().cpu()
    msh = m.shape
    if opts['map_size'] > 0:
        zfac = opts['map_size'] / msh[2]
    else:
        zfac = 1.0
    sfac = opts['gauss_kernel'] * zfac
    if opts['probmaps'] or opts['log_odds']:
        pm = torch.nn.functional.softmax(m, 1).numpy().astype(np.float32)
        pm[pm < opts['min_p']] = opts['min_p']
        if zfac != 1.0:
            pm = np.exp(ndimage.zoom(np.log(pm), (1.0, 1.0, zfac, zfac), order=3, mode='nearest'))
        if sfac > 0.0:
            pm = ndimage.gaussian_filter(pm, (0.0, 0.0, sfac, sfac), mode='nearest')
        if opts['probmaps']:
            for iidx in range(msh[0]):
                sdir = os.path.join(opts['probmap_dir'], f'{imnames[si+iidx]}')
                os.makedirs(sdir, exist_ok=True)
                for o in opts['outputs']:
                    if o in oimap:
                        np.savez(os.path.join(sdir, f'pmap_{o}_{mname[:-4]}.npz'),
                                 pm[iidx,oimap[o],:,:].squeeze())
    m = m.numpy().astype(np.float32)
    if opts['heatmaps']:
        if zfac != 1.0:
            m = ndimage.zoom(m, (1.0, 1.0, zfac, zfac), order=3, mode='nearest')
        if sfac > 0.0:
            m = ndimage.gaussian_filter(m, (0.0, 0.0, sfac, sfac), mode='nearest')
        for iidx in range(msh[0]):
            sdir = os.path.join(opts['heatmap_dir'], f'{imnames[si+iidx]}')
            os.makedirs(sdir, exist_ok=True)
            for o in opts['outputs']:
                if o in oimap:
                    np.savez(os.path.join(sdir, f'hmap_{o}_{mname[:-4]}.npz',),
                             m[iidx,oimap[o],:,:].squeeze())
    if opts['log_odds']:
        for iidx in range(msh[0]):
            sdir = os.path.join(opts['log_odds_dir'], f'{imnames[si+iidx]}')
            os.makedirs(sdir, exist_ok=True)
    for lo in opts['log_odds']:
        lop0 = lo[0]
        lop1 = lo[1]
        if lop0 != 'other' and lop0 not in oimap:
            continue
        if lop1 != 'other' and lop1 not in oimap:
            continue
        if lop0 == 'other':
            lop0 = list(set(range(len(oimap))) - set([oimap[lop1]]))
            lop1 = [oimap[lop1]]
        elif lop1 == 'other':
            lop1 = list(set(range(len(oimap))) - set([oimap[lop0]]))
            lop0 = [oimap[lop0]]
        else:
            lop0 = [oimap[lop0]]
            lop1 = [oimap[lop1]]
        m0 = np.sum(pm[:,lop0,:,:], axis=1)
        m1 = np.sum(pm[:,lop1,:,:], axis=1)
        lm = np.log(m0 / m1)
        for iidx in range(msh[0]):
            np.savez(os.path.join(opts['log_odds_dir'], f'{imnames[si+iidx]}',
                     f'lomap_{lo[0]}~{lo[1]}_{mname[:-4]}.npz'), lm[iidx,:,:])

# flip/rotate transforms (n_test)
def get_trans(img, I, ft=True):
    if I >= 4 and ft:
        img = img.transpose(2, 3)
    if I % 4 == 1:
        img = img.flip(2)
    elif I % 4 == 2:
        img = img.flip(3)
    elif I % 4 == 3:
        img = img.flip(2).flip(3)
    if I >= 4 and not ft:
        img = img.transpose(2, 3)
    return img


# main()
def predict(**kwargs):

    # override args (no checks!)
    for kw, kv in kwargs.items():
        if kw in opts:
            opts[kw] = kv
    
    # imports to avoid wait for argparse --help
    from time import sleep

    import pandas as pd
    from PIL import Image as pil_image
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import torch
    from torch.utils.data import DataLoader
    if opts['device_name'] == 'auto':
        opts['device_name'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if opts['silent']:
        utqdm = silent_tqdm
    else:
        utqdm = tqdm

    from melanoma_util import (
        get_transforms,
        MelanomaDataset,
        Effnet_Melanoma,
        Resnest_Melanoma,
        Seresnext_Melanoma,
    )

    # device
    device = torch.device(opts['device_name'])
    
    # pre-load transformations
    transforms_val = {}
    for image_size in _used_image_sizes:
        transforms_val[image_size] = get_transforms(image_size)

    # default: no selection map
    tsmap = None
   
    # pre-load models
    if opts['preload']:
        
        # iterate over used models
        models = {}
        msradm = {}
        for mshort in opts['use_models']:
            
            # select class to instantiate
            mparam = _used_models[mshort]
            if mparam[2] == 'resnest101':
                ModelClass = Resnest_Melanoma
            elif mparam[2] == 'seresnext101':
                ModelClass = Seresnext_Melanoma
            else:
                ModelClass = Effnet_Melanoma
            
            # and meta features used (and number)
            if mparam[4]:
                meta_features = _used_meta_features
                n_meta_features = len(meta_features)
            else:
                meta_features = None
                n_meta_features = 0
            
            # iterate over selected folds
            for fold in opts['folds']:
                mname = f'{mshort}_best_fold{fold}.pth'
                
                # instantiate model
                model = ModelClass(
                    mparam[2],
                    n_meta_features=n_meta_features,
                    n_meta_dim=[int(nd) for nd in mparam[5].split(',')],
                    out_dim=mparam[3]
                )
                model = model.to(device)
                
                # load model (PTH) file
                model_file = os.path.join(opts['model_dir'], mname)
                if not opts['silent']:
                    print(f'Loading {model_file}...')
                try:  # single GPU model_file
                    model.load_state_dict(torch.load(model_file, map_location=device), strict=True)
                except:  # multi GPU model_file
                    state_dict = torch.load(model_file, map_location=device)
                    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
                    model.load_state_dict(state_dict, strict=True)
                if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
                    model = torch.nn.DataParallel(model)
                model.eval()
                models[mname] = model

            # create model-specific radius map
            if opts['select_radius'] > 0.0:
                msradm[mshort] = selection_radius(mparam[1], device)
    
    # setup potential loop
    lc = 0
    while True:
        
        # keep scanning data dir for CSV files
        if '*' in opts['csv_file']:
            lc += 1
            if lc == 1:
                if not opts['silent']:
                    if os.path.sep in opts['csv_file']:
                        print(f'Scanning for CSV files ({opts["csv_file"]})...', end='', flush=True)
                    else:
                        print(f'Scanning {opts["data_dir"]} for CSV files ({opts["csv_file"]})...', end='', flush=True)
            elif lc < 600:
                if (lc % 10) == 0 and not opts['silent']:
                    print('.', end='', flush=True)
            else:
                if not opts['silent']:
                    print('.', flush=True)
                lc = 0
            if os.path.sep in opts['csv_file']:
                test_csv = glob.glob(opts['csv_file'])
            else:
                test_csv = glob.glob(os.path.join(opts['data_dir'], opts['csv_file']))
            
            # nothing found loop
            if len(test_csv) == 0:
                sleep(0.1)
                continue
            
            # process (leave loop)
            if not opts['silent']:
                print('.', flush=True)
            lc = 0
        
        # specific file name
        else:
            
            # prepend data-dir if file not found
            load_file = opts['csv_file']
            if not os.path.sep in load_file:
                if not os.path.exists(load_file):
                    load_file = os.path.join(opts['data_dir'], load_file)
            
            # if file not found, exit
            if not os.path.exists(load_file):
                print('CSV file from --csv-file not found.')
                return 1
            test_csv = [load_file]
    
        # process data
        if not opts['silent']:
            print(f'Processing data in {test_csv[0]}...')
        try:
            
            # load CSV and check columns
            df_name = os.path.splitext(os.path.basename(test_csv[0]))[0]
            df_test = pd.read_csv(test_csv[0]).rename(columns=_rename_cols)
            for c in _check_cols:
                try:
                    col = df_test[c]
                except:
                    raise RuntimeError(f'No {c} column in CSV.')
        except Exception as e:
            if not opts['silent']:
                print(e)
            os.rename(test_csv[0], os.path.splitext(test_csv[0])[0] + '.error')
            continue
        
        # get image names and test/retrieve width/height
        image_names = df_test['image_name'].values
        if not ('width' in df_test.columns and 'height' in df_test.columns):
            import imagesize
            if not opts['silent']:
                print(f'Retrieving image sizes for {df_test.shape[0]} images...')
            filepath = opts['data_dir']
            widths = [0] * len(image_names)
            heights = [0] * len(image_names)
            for idx, image in enumerate(image_names):
                try:
                    (h, w) = imagesize.get(f'{filepath}/{image}.jpg')
                    widths[idx] = w
                    heights[idx] = h
                except:
                    pass
            df_test['width'] = widths
            df_test['height'] = heights
        
        # show DF (head/tail) for diagnostic purposes
        if not opts['silent']:
            print(df_test)

        # minimal mangling (scale age, numeric sex, and dummy site coding)
        df_test['age_approx'] = df_test['age_approx'].fillna(0)
        df_test['age_approx'] = df_test['age_approx'] / 90.0
        df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
        df_test['sex'] = df_test['sex'].fillna(-1)
        site_nan = np.ones(df_test.shape[0])
        for s in _used_meta_features:
            if not 'site_' in s:
               continue
            st = s[5:]
            sv = [1 if v == st else 0 for v in df_test['anatom_site_general_challenge'].values]
            df_test[s] = sv
            site_nan[df_test[s] > 0] = 0
        df_test['site_nan'] = site_nan
        df_org = df_test
        
        # iterate over used data folders (base sizes)
        df_tests = {}
        dataset_tests_meta = {}
        dataset_tests_nometa = {}
        test_loaders_meta = {}
        test_loaders_nometa = {}
        skip_images = {}
        for data_folder in _used_data_folders:
            df_test = df_org.copy()
            size_folder = os.path.join(opts['data_dir'],
                          f'jpeg-melanoma-{data_folder}x{data_folder}', 'test')
            if not os.path.exists(size_folder):
                os.makedirs(size_folder, exist_ok=True)
            df_test['filepath'] = df_test['image_name'].apply(
                lambda x: os.path.join(size_folder, f'{x}.jpg'))
            
            # test for resized image availability and rescale if necessary
            test_images = df_test['filepath'].values
            for iidx, test_image in enumerate(utqdm(test_images, desc=f'Checking images in {data_folder}x{data_folder}')):
                if not os.path.exists(test_image):
                    source_image = os.path.join(opts['data_dir'], f'{image_names[iidx]}.jpg')
                    if not os.path.exists(source_image):
                        skip_images[image_names[iidx]] = True
                        continue
                    try:
                        image_data = pil_image.open(source_image)
                        image_w, image_h = image_data.size
                        if image_w > image_h:
                            to_crop = (image_w - image_h) // 2
                            image_data = image_data.crop((to_crop, 0, to_crop + image_h, image_h))
                        elif image_h > image_w:
                            to_crop = (image_h - image_w) // 2
                            image_data = image_data.crop((0, to_crop, image_w, to_crop + image_w))
                        image_data = image_data.resize((data_folder, data_folder), resample=pil_image.LANCZOS)
                        image_data.save(test_image, quality=95)
                    except:
                        if not opts['silent']:
                            print(f'Error resizing {image_names[iidx]}.jpg to {data_folder}x{data_folder}')
                        skip_images[image_names[iidx]] = True
            df_tests[data_folder] = df_test
        
        # if any images were missing (and couldn't be resized)
        did_break = False
        if skip_images:
            if not opts['silent']:
                print(f'{len(skip_images)} images not found in all resolutions and skipped.')
            for data_folder in _used_data_folders:
                df_tests[data_folder] = df_tests[data_folder][np.logical_not(df_tests[data_folder]['image_name'].isin(skip_images))]
                if df_tests[data_folder].shape[0] == 0:
                    print('No images remain after skipping.')
                    did_break = True
                    break
        if did_break:
            if '*' in opts['csv_file']:
                os.rename(test_csv[0], os.path.splitext(test_csv[0])[0] + '.error')
                continue
            else: 
                return 1
        
        # set number of images per patient and image file sizes
        for data_folder in _used_data_folders:
            df_test = df_tests[data_folder]
            test_images = df_test['filepath'].values
            test_sizes = np.zeros(test_images.shape[0])
            for i, img_path in enumerate(test_images):
                test_sizes[i] = os.path.getsize(img_path)
            df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
            df_test['n_images'] = np.log1p(df_test['n_images'].values)
            df_test['image_size'] = np.log(test_sizes)
            df_tests[data_folder] = df_test
            
            # create appropriate DataLoader (with and without metadata) objects
            for image_size in _used_image_sizes:
                df_is = f'{data_folder}_{image_size}'
                dataset_tests_meta[df_is] = MelanomaDataset(
                    df_test, 'test', _used_meta_features, transform=transforms_val[image_size])
                dataset_tests_nometa[df_is] = MelanomaDataset(
                    df_test, 'test', None, transform=transforms_val[image_size])
                test_loaders_meta[df_is] = torch.utils.data.DataLoader(
                    dataset_tests_meta[df_is], batch_size=args.batch_size, num_workers=args.num_workers)
                test_loaders_nometa[df_is] = torch.utils.data.DataLoader(
                    dataset_tests_nometa[df_is], batch_size=args.batch_size, num_workers=args.num_workers)
        
        # re-set image_names (for later use)
        image_names = df_test['image_name'].values
        
        # rename CSV file (handled at this point)
        if '*' in opts['csv_file']:
            os.rename(test_csv[0], os.path.splitext(test_csv[0])[0] + '.processing')
        
        # for preloaded models
        if opts['preload']:
            
            # iterate over (already loaded) models
            for mname, model in utqdm(models.items()):
                mshort = mname[:-15]
                mparam = _used_models[mshort]
                data_folder = mparam[0]
                image_size = mparam[1]
                df_is = f'{data_folder}_{image_size}'
                if opts['select_radius'] > 0.0:
                    tsmap = msradm[mshort]
                
                # get appropriate DataLoader object
                if mparam[4]:
                    test_loader = test_loaders_meta[df_is]
                else:
                    test_loader = test_loaders_nometa[df_is]
                
                # preset outputs
                raw_outs = torch.zeros((df_test.shape[0], mparam[3], opts['n_test']), dtype=torch.float64).to(device)
                raw_outi = 0
                if mparam[3] > 4:
                    oimap = _outidx_full
                else:
                    oimap = _outidx_part
                
                # iterate over data
                with torch.no_grad():
                    for (data) in test_loader:
                        
                        # load data (and metadata)
                        if mparam[4]:
                            data, meta = data
                            data, meta = data.to(device), meta.to(device)
                        else:
                            data = data.to(device)
                            meta = None
                        
                        # keep track of index
                        ds0 = data.shape[0]
                        raw_oute = raw_outi + ds0
                        
                        # iterate over flipped/rotated images
                        for I in range(opts['n_test']):
                            l, pemb, pmaps = model.forward_and_maps(get_trans(data, I), meta, tsmap)
                            
                            # store outputs
                            raw_outs[raw_outi:raw_oute, :, I] = l
                            if I == 0:
                                emb = pemb
                                maps = pmaps
                            else:
                                emb += pemb
                                maps += get_trans(pmaps, I, False)
                        
                        # process embeddings and maps
                        if opts['embeddings']:
                            process_emb(emb, mname, image_names, raw_outi)
                        if opts['heatmaps'] or opts['probmaps'] or opts['log_odds']:
                            process_maps(maps, mname, oimap, image_names, raw_outi)
                        
                        # keep track of index
                        raw_outi = raw_oute
                
                # store raw predictions (before softmax!)
                raw_outs = raw_outs.detach().cpu()
                if opts['store_raw']:
                    raw_outs_np = raw_outs.numpy()
                    np.savez(os.path.join(opts['sub_dir'], f'raw_{df_name}_{mname[:-4]}.npz'), raw_outs_np)
                
                # appropriate averaging
                if opts['log_avg'] and opts['n_test'] > 1:
                    PROBS = torch.nn.functional.log_softmax(raw_outs, 1).numpy()
                else:
                    PROBS = torch.nn.functional.softmax(raw_outs, 1).numpy()
                if opts['n_test'] > 1:
                    PROBS = np.mean(PROBS, axis=2)
                else:
                    PROBS = PROBS[:, :, 0]
                if opts['log_avg'] and opts['n_test'] > 1:
                    PROBS = np.exp(PROBS)
                
                # store requested columns
                store_cols = ['image_name']
                for o in opts['outputs']:
                    if o in oimap:
                        df_test[o] = PROBS[:, oimap[o]]
                        store_cols.append(o)
                os.makedirs(os.path.join(opts['sub_dir'], df_name), exist_ok=True)
                df_test[store_cols].rename(columns={opts['target']: 'target'}).to_csv(
                    os.path.join(opts['sub_dir'], df_name, f'sub_{mname[:-4]}.csv'), index=False)
                for scoli in range(1, len(store_cols)):
                    if store_cols[scoli] in df_test.columns:
                        df_test = df_test.drop(store_cols[scoli], axis=1)
                
        # load models one at a time (more time, less memory, same logic as above)
        else:
            
            # iterate over models
            for mshort in opts['use_models']:
                mparam = _used_models[mshort]
                if mparam[2] == 'resnest101':
                    ModelClass = Resnest_Melanoma
                elif mparam[2] == 'seresnext101':
                    ModelClass = Seresnext_Melanoma
                else:
                    ModelClass = Effnet_Melanoma
                if mparam[4]:
                    meta_features = _used_meta_features
                    n_meta_features = len(meta_features)
                else:
                    meta_features = None
                    n_meta_features = 0
                model = ModelClass(
                    mparam[2],
                    n_meta_features=n_meta_features,
                    n_meta_dim=[int(nd) for nd in mparam[5].split(',')],
                    out_dim=mparam[3]
                )
                model = model.to(device)
                if mparam[3] > 4:
                    oimap = _outidx_full
                else:
                    oimap = _outidx_part
                
                # get correct DataLoader object
                data_folder = mparam[0]
                image_size = mparam[1]
                df_is = f'{data_folder}_{image_size}'
                if mparam[4]:
                    test_loader = test_loaders_meta[df_is]
                else:
                    test_loader = test_loaders_nometa[df_is]
                
                # selection radius
                if opts['select_radius'] > 0.0:
                    tsmap = selection_radius(mparam[1], device)
                
                # iterate over folder
                for fold in opts['folds']:
                    
                    # load model
                    mname = f'{mshort}_best_fold{fold}.pth'
                    model_file = os.path.join(opts['model_dir'], mname)
                    try:  # single GPU model_file
                        model.load_state_dict(torch.load(model_file, map_location=device), strict=True)
                    except:  # multi GPU model_file
                        state_dict = torch.load(model_file, map_location=device)
                        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
                        model.load_state_dict(state_dict, strict=True)
                    if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
                        model = torch.nn.DataParallel(model)
                    model.eval()
                    
                    # run through data
                    raw_outs = torch.zeros((df_test.shape[0], mparam[3], opts['n_test']), dtype=torch.float64).to(device)
                    raw_outi = 0
                    with torch.no_grad():
                        for (data) in utqdm(test_loader, desc=f'Evaluating {mname}'):
                            
                            if mparam[4]:
                                data, meta = data
                                data, meta = data.to(device), meta.to(device)
                            else:
                                data = data.to(device)
                                meta = None
                            ds0 = data.shape[0]
                            raw_oute = raw_outi + ds0
                            for I in range(opts['n_test']):
                                l, pemb, pmaps = model.forward_and_maps(get_trans(data, I), meta, tsmap)
                                raw_outs[raw_outi:raw_oute, :, I] = l
                                if I == 0:
                                    emb = pemb
                                    maps = pmaps
                                else:
                                    emb += pemb
                                    maps += get_trans(pmaps, I, False)
                            if opts['embeddings']:
                                process_emb(emb, mname, image_names, raw_outi)
                            if opts['heatmaps'] or opts['probmaps'] or opts['log_odds']:
                                process_maps(maps, mname, oimap, image_names, raw_outi)
                            raw_outi = raw_oute
                            
                    raw_outs = raw_outs.detach().cpu()
                    if opts['store_raw']:
                        raw_outs_np = raw_outs.numpy()
                        np.savez(os.path.join(opts['sub_dir'], f'raw_{df_name}_{mname[:-4]}.npz'), raw_outs_np)
                    if opts['log_avg'] and opts['n_test'] > 1:
                        PROBS = torch.nn.functional.log_softmax(raw_outs, 1).numpy()
                    else:
                        PROBS = torch.nn.functional.softmax(raw_outs, 1).numpy()
                    if opts['n_test'] > 1:
                        PROBS = np.mean(PROBS, axis=2)
                    else:
                        PROBS = PROBS[:, :, 0]
                    if opts['log_avg'] and opts['n_test'] > 1:
                        PROBS = np.exp(PROBS)
                    store_cols = ['image_name']
                    for o in opts['outputs']:
                        if o in oimap:
                            df_test[o] = PROBS[:, oimap[o]]
                            store_cols.append(o)
                    os.makedirs(os.path.join(opts['sub_dir'], df_name), exist_ok=True)
                    df_test[store_cols].rename(columns={opts['target']: 'target'}).to_csv(
                        os.path.join(opts['sub_dir'], df_name, f'sub_{mname[:-4]}.csv'), index=False)
                    for scoli in range(1, len(store_cols)):
                        if store_cols[scoli] in df_test.columns:
                            df_test = df_test.drop(store_cols[scoli], axis=1)
        
        # ensemble CSVs
        if opts['ensemble']:
            
            # pass on
            ensemble_csvs(df_name)
            for image_name in utqdm(image_names, desc='Ensembling maps'):
                if not ensemble_maps(image_name):
                    break
        
        # rename CSV file (handled at this point)
        if '*' in opts['csv_file']:
            os.rename(os.path.splitext(test_csv[0])[0] + '.processing',
                      os.path.splitext(test_csv[0])[0] + '.done')
            
        # single pass for named CSV file
        else:
            break
        
    # return (if broken out of glob search loop)
    return 0


if __name__ == '__main__':
    
    # parse arguments (and throw errors on basics)
    args = parse_args()
    download_models = True if args.download_models else False
    opts['silent'] = True if args.silent else False

    # mini banner
    if not opts['silent']:
        print('SIIM/ISIC 2020 Melanoma Detection challenge winning models evaluation')
    
    # check model file directory
    ap = process_args(args, download_models)
    if ap:
        sys.exit(ap)

    # environment setup (only done as __main__ !)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
    
    # some information, then run predict()
    if not opts['silent']:
        if not os.path.sep in opts['csv_file']:
            print(f'Processing {len(opts["folds"])} folds of {len(opts["use_models"])} models on {opts["data_dir"]}/{opts["csv_file"]}...')
        else:
            print(f'Processing {len(opts["folds"])} folds of {len(opts["use_models"])} models on {opts["csv_file"]}...')
        print(f' - storing submission CSVs in {opts["sub_dir"]}...')
        print(f' - extracting values/maps for: {", ".join(opts["outputs"])}')
        print(f'   - with target {opts["target"]}')
        if opts['heatmaps'] or opts['probmaps']:
            if opts['gauss_kernel'] > 0.0:
                print(f' - smoothing all maps with {opts["gauss_kernel"]}-unit sigma...')
            else:
                print(' - keeping maps unsmoothed...')
        if opts['map_size'] > 0:
            map_shape = f'{opts["map_size"]}x{opts["map_size"]}'
        else:
            map_shape = 'original sized'
        if opts['heatmaps']:
            print(f' - storing {map_shape} heat maps in {opts["heatmap_dir"]}...')
        if opts['probmaps']:
            print(f' - storing {map_shape} probability maps in {opts["probmap_dir"]}...')
        if opts['log_odds']:
            print(f' - storing {map_shape} log-odds maps in {opts["log_odds_dir"]}...')
    sys.exit(predict())

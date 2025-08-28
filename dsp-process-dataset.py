import numpy as np
from dsp import generate_features
import sys, os, json, signal, time, math
import argparse

parser = argparse.ArgumentParser(description='Generate features')
parser.add_argument('--in-file-x', type=str, required=True)
parser.add_argument('--out-file-x', type=str, required=True)
parser.add_argument('--implementation-version', type=int, required=True)
parser.add_argument('--input-block-type', type=str, required=True)
parser.add_argument('--axes', type=str, required=True)
parser.add_argument('--uses-state', action="store_true")
parser.add_argument('--metadata-file', type=str, required=True)

args, unknown = parser.parse_known_args()

def exit_gracefully(signum, frame):
    sys.exit(1)

signal.signal(signal.SIGINT, exit_gracefully)
signal.signal(signal.SIGTERM, exit_gracefully)

curr_dir = os.path.dirname(os.path.realpath(__file__))

param_mapping = {}
with open(os.path.join(curr_dir, 'parameters.json'), 'r') as f:
    param_file = json.loads(f.read())
    for group in param_file['parameters']:
        for item in group['items']:
            param_mapping[item['param']] = item

extra_kwargs = {}

for i in np.arange(0, len(unknown), step=2):
    key = unknown[i].replace('--', '')
    value = unknown[i + 1]
    if not key in param_mapping.keys():
        print('Cannot find key "' + key + '" in parameters.json')
        sys.exit(1)

    param = param_mapping[key]
    if value == 'None':
        extra_kwargs[key] = None
    elif (param['type'] == 'int'):
        extra_kwargs[key] = int(value)
    elif (param['type'] == 'float'):
        extra_kwargs[key] = float(value)
    else:
        extra_kwargs[key] = value

in_file_x = args.in_file_x
outFileX = args.out_file_x
axes = args.axes.split(',')
metadataFile = args.metadata_file
labels = []
output_config = None
fft_used = None
total = 0
last_message = 0
freq_hz = 0

X_train = np.load(in_file_x, mmap_mode='r')

rows = X_train.shape[0]
input_els = np.prod(X_train.shape[1:])

print('Creating features...')
sys.stdout.flush()

features_file = None

state = None
for example in X_train:
    # time-series examples have the interval in the first column
    raw_data = example if args.input_block_type == 'image' else example[1:]
    freq_hz = 0 if args.input_block_type == 'image' else example[0]

    kwargs = {
        "implementation_version": args.implementation_version,
        "draw_graphs": False,
        "raw_data": raw_data,
        "axes": axes,
        "sampling_freq": freq_hz,
    }
    if args.uses_state:
        kwargs['state'] = state
    for k in extra_kwargs.keys():
        kwargs[k] = extra_kwargs[k]

    f = generate_features(**kwargs)

    # first row? then we look at the number of features generated...
    if (features_file is None):
        features_file = np.lib.format.open_memmap(outFileX, mode=('r+' if os.path.exists(outFileX) else 'w+'), dtype='float32', shape=(rows, len(f['features'])))

        if ('labels' in f):
            labels = f['labels']
        if ('output_config' in f):
            output_config = f['output_config']
        if ('fft_used' in f):
            fft_used = f['fft_used']

    features_file[total] = f['features']

    total += 1

    if (int(round(time.time() * 1000)) - last_message >= 3000):
        print('[%s/%d] Creating features...' % (str(total).rjust(len(str(rows)), ' '), rows))
        sys.stdout.flush()
        last_message = int(round(time.time() * 1000))

if (features_file is None):
    features_file = np.lib.format.open_memmap(outFileX, mode=('r+' if os.path.exists(outFileX) else 'w+'), dtype='float32', shape=(0, 0))

print('[%s/%d] Creating features...' % (str(total).rjust(len(str(rows)), ' '), rows))
sys.stdout.flush()

# make sure to flush by calling del
del features_file

if metadataFile:
    with open(metadataFile, 'w') as f:
        metadata = { 'labels': labels, 'input_shape': [ int(input_els) ], 'output_config': output_config, 'fft_used': fft_used }
        if freq_hz:
            metadata['frequency'] = freq_hz
        f.write(json.dumps(metadata, indent=4))

print('Created features')

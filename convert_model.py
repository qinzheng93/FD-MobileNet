from __future__ import print_function

import argparse
import os
import json

import torch
import torch.nn.parallel

import pyvision.models as models

parser = argparse.ArgumentParser(description='Convert Multi-GPU PyTorch Model to Single-GPU Model')
parser.add_argument('--data', dest='data_config', required=True, metavar='DATA_CONFIG', help='Dataset config file')
parser.add_argument('--model', dest='model_config', required=True, metavar='MODEL_CONFIG', help='Model config file')
parser.add_argument('--label', dest='label', required=True, metavar='MODEL_LABEL', help='Model label')
parser.add_argument('--input', dest='input', required=True, metavar='INPUT_FILE', help='Checkpoint file to be converted')
parser.add_argument('--output', dest='output', required=True, metavar='OUTPUT_FILE', help='Output filename')

def main():
    args = parser.parse_args()
    with open(args.data_config, 'r') as json_file:
        data_config = json.load(json_file)
    with open(args.model_config, 'r') as json_file:
        model_config = json.load(json_file)

    model = models.get_model(data_config['name'], model_config)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.input)
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint = {'name': args.label, 'state_dict': model.module.state_dict()}
    torch.save(checkpoint, args.output)

if __name__ == '__main__':
    main()

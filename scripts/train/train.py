# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import sys

from llmfoundry.command_utils import train_from_yaml
from omegaconf import OmegaConf as om

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train_from_yaml(om, yaml_path, args_list)

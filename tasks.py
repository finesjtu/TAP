# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
import numpy as np
import ast

import log
from pet import task_helpers
from pet.utils import InputExample

logger = log.get_logger('root')


TARGET = {
    'AT': 'Atheism',
    'CC': 'Climate Change is a real Concern',
    'FM': 'Feminist Movement',
    'HC': 'Hillary Clinton',
    'LA': 'Legalization of Abortion'
}
TARGET_ARGMIN = {
    'AB': 'abortion',
    'CL': 'cloning',
    'DP': 'death penalty',
    'GC': 'gun control',
    'ML': 'marijuana legalization',
    'MW': 'minimum wage',
    'NE': 'nuclear energy',
    'SU': 'school uniforms',

}

def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    Shuffle a list of examples and restrict it to a given maximum size.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples


class LimitedExampleList:
    def __init__(self, labels: List[str], max_examples=-1):
        """
        Implementation of a list that stores only a limited amount of examples per label.

        :param labels: the set of all possible labels
        :param max_examples: the maximum number of examples per label. This can either be a fixed number,
               in which case `max_examples` examples are loaded for every label, or a list with the same size as
               `labels`, in which case at most `max_examples[i]` examples are loaded for label `labels[i]`.
        """
        self._labels = labels
        self._examples = []
        self._examples_per_label = defaultdict(int)

        if isinstance(max_examples, list):
            self._max_examples = dict(zip(self._labels, max_examples))
        else:
            self._max_examples = {label: max_examples for label in self._labels}

    def is_full(self):
        """Return `true` iff no more examples can be added to this list"""
        for label in self._labels:
            if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
                return False
        return True

    def add(self, example: InputExample) -> bool:
        """
        Add a new input example to this list.

        :param example: the example to add
        :returns: `true` iff the example was actually added to the list
        """
        label = example.label
        if self._examples_per_label[label] < self._max_examples[label] or self._max_examples[label] < 0:
            self._examples_per_label[label] += 1
            self._examples.append(example)
            return True
        return False

    def to_list(self):
        return self._examples


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass


class Semeval2016t6Processor(DataProcessor):
    """Processor for the 2016t6 data set."""

    def get_train_examples(self, data_dir):

        return self._create_examples(Semeval2016t6Processor._read_tsv(os.path.join(data_dir, "trainingdata-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(Semeval2016t6Processor._read_dev_tsv([os.path.join(data_dir, "trialdata-all-annotations.txt"),
                                                                           os.path.join(data_dir, "trainingdata-all-annotations.txt")],
                                                                      os.path.join(data_dir, "split/semeval2016t6_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(Semeval2016t6Processor._read_tsv(os.path.join(data_dir, "testdata-taskA-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_test.csv")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["AGAINST", "FAVOR", "NONE"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        # for i in lines:
        #     print(i)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1][1]
            text_b = line[1][0]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, split_file, quotechar=None):
        with open(input_file, "r", encoding="ISO-8859-1") as f, \
                open(split_file, "r") as split_file:
            data = list(csv.reader(f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                line.append(line_number)
                # print(data[line_number])
                line.append((data[line_number][1], data[line_number][2]))
                line.append(data[line_number][3])
                lines.append(line)


            return lines

    @staticmethod
    def _read_dev_tsv(input_file, split_file, quotechar=None):
        with open(input_file[0], "r", encoding="ISO-8859-1") as dev_f, \
                open(input_file[1], "r", encoding="ISO-8859-1") as train_f, \
                open(split_file, "r") as split_file:
            dev_data = list(csv.reader(dev_f, delimiter='\t', quotechar='"', ))
            train_data = list(csv.reader(train_f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                if data_file == 'train':
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((train_data[line_number][1], train_data[line_number][2]))
                    line.append(train_data[line_number][3])
                else:
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((dev_data[line_number][1], dev_data[line_number][2]))
                    line.append(dev_data[line_number][3])
                lines.append(line)

            return lines

class TargetSemeval2016t6Processor(DataProcessor):
    """Processor for the 2016t6 data set."""
    def __init__(self, target):

        self.target = TARGET[target]

    def get_train_examples(self, data_dir):

        return self._create_examples(TargetSemeval2016t6Processor._read_tsv(os.path.join(data_dir, "trainingdata-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(TargetSemeval2016t6Processor._read_dev_tsv([os.path.join(data_dir, "trialdata-all-annotations.txt"),
                                                                           os.path.join(data_dir, "trainingdata-all-annotations.txt")],
                                                                      os.path.join(data_dir, "split/semeval2016t6_dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(TargetSemeval2016t6Processor._read_tsv(os.path.join(data_dir, "testdata-taskA-all-annotations.txt"),
                                                                      os.path.join(data_dir, "split/semeval2016t6_test.csv")), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["AGAINST", "FAVOR", "NONE"]


    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []
        # for i in lines:
        #     print(i)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1][1]
            text_b = line[1][0]
            label = line[-1]
            # print(self.target)
            # print(text_b)
            # print(text_a)
            if text_b.lower() == self.target.lower():
                # print(text_b)
                # print(self.target)
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
            else:
                continue

        return examples

    @staticmethod
    def _read_tsv(input_file, split_file, quotechar=None):
        with open(input_file, "r", encoding="ISO-8859-1") as f, \
                open(split_file, "r") as split_file:
            data = list(csv.reader(f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                line.append(line_number)
                # print(data[line_number])
                line.append((data[line_number][1], data[line_number][2]))
                line.append(data[line_number][3])
                lines.append(line)


            return lines

    @staticmethod
    def _read_dev_tsv(input_file, split_file, quotechar=None):
        with open(input_file[0], "r", encoding="ISO-8859-1") as dev_f, \
                open(input_file[1], "r", encoding="ISO-8859-1") as train_f, \
                open(split_file, "r") as split_file:
            dev_data = list(csv.reader(dev_f, delimiter='\t', quotechar='"', ))
            train_data = list(csv.reader(train_f, delimiter='\t', quotechar='"', ))
            lines = []
            for row in split_file.readlines():
                data_file, line_number = row.rstrip().split("_")
                line_number = int(line_number)
                line = []
                if data_file == 'train':
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((train_data[line_number][1], train_data[line_number][2]))
                    line.append(train_data[line_number][3])
                else:
                    line.append(line_number)
                    # print(data[line_number])
                    line.append((dev_data[line_number][1], dev_data[line_number][2]))
                    line.append(dev_data[line_number][3])
                lines.append(line)

            return lines

class TargetArgminProcessor(DataProcessor):
    """Processor for the Argmin data set."""
    def __init__(self, target):
        self.target = TARGET_ARGMIN[target]

    def get_train_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessor._read_tsv(os.path.join(data_dir), "train")[0], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessor._read_tsv(os.path.join(data_dir), "dev")[1], "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessor._read_tsv(os.path.join(data_dir), "test")[2], "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["Argument_against", "Argument_for", "NoArgument"]

    # @staticmethod
    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1]
            text_b = line[0][0]
            label = line[-1]
            # print(line)
            if text_b.lower() == self.target.lower():
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)
            else:
                continue

        return examples

    @staticmethod
    def _read_tsv(input_file, set, quotechar=None):
        gold_files = os.listdir(input_file)
        train_lines = []
        dev_lines = []
        test_lines = []
        count = 0
        for data_file in gold_files:
            if data_file.endswith(".tsv"):
                topic = data_file.replace(".tsv", "")
                with open(input_file + data_file, 'r') as f_in:
                    reader = csv.reader(f_in, delimiter="\t", quoting=3)
                    next(reader, None)
                    for row in reader:
                        line = []
                        line.append((row[0], row[4]))
                        line.append(row[5])
                        set = row[6]

                        if set == 'train':
                            train_lines.append(line)
                        elif set == 'test':
                            test_lines.append(line)
                        else:
                            dev_lines.append(line)

        return [train_lines, dev_lines, test_lines]

class TargetArgminProcessorall(DataProcessor):
    """Processor for the Argmin data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessorall._read_tsv(os.path.join(data_dir), "train")[0], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessorall._read_tsv(os.path.join(data_dir), "dev")[1], "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(TargetArgminProcessorall._read_tsv(os.path.join(data_dir), "test")[2], "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["Argument_against", "Argument_for", "NoArgument"]

    # @staticmethod
    def _create_examples(self, lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1]
            text_b = line[0][0]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)


        return examples

    @staticmethod
    def _read_tsv(input_file, set, quotechar=None):
        gold_files = os.listdir(input_file)
        train_lines = []
        dev_lines = []
        test_lines = []
        count = 0
        for data_file in gold_files:
            if data_file.endswith(".tsv"):
                topic = data_file.replace(".tsv", "")
                with open(input_file + data_file, 'r') as f_in:
                    reader = csv.reader(f_in, delimiter="\t", quoting=3)
                    next(reader, None)
                    for row in reader:
                        line = []
                        line.append((row[0], row[4]))
                        line.append(row[5])
                        set = row[6]

                        if set == 'train':
                            train_lines.append(line)
                        elif set == 'test':
                            test_lines.append(line)
                        else:
                            dev_lines.append(line)

        return [train_lines, dev_lines, test_lines]

class ArgminProcessor(DataProcessor):
    """Processor for the Argmin data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "train")[0], "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "dev")[1], "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(ArgminProcessor._read_tsv(os.path.join(data_dir), "test")[2], "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["Argument_against", "Argument_for"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[0][1]
            text_b = line[0][0]
            label = line[-1]
            # print(line)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, set, quotechar=None):
        gold_files = os.listdir(input_file)
        train_lines = []
        dev_lines = []
        test_lines = []
        count = 0
        for data_file in gold_files:
            if data_file.endswith(".tsv"):
                topic = data_file.replace(".tsv", "")
                with open(input_file + data_file, 'r') as f_in:
                    reader = csv.reader(f_in, delimiter="\t", quoting=3)
                    next(reader, None)
                    for row in reader:
                        line = []
                        if row[5] != 'NoArgument':
                            if topic == "death_penalty":
                                # print(set)
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                # if len(line) != 0:
                                dev_lines.append(line)

                            elif topic == "school_uniforms" or topic == "gun_control":

                                # print(set)
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                test_lines.append(line)
                            else:
                                line.append((row[0], row[4]))
                                line.append(row[5])
                                # if len(line) != 0:
                                train_lines.append(line)
                                # print(topic)

        # print(len(lines))
        # print(count)
        return [train_lines, dev_lines, test_lines]



import re

import zipfile



PROCESSORS = {

    "2016t6": Semeval2016t6Processor,
    "AT2016t6": lambda: TargetSemeval2016t6Processor('AT'),
    "CC2016t6": lambda: TargetSemeval2016t6Processor('CC'),
    "FM2016t6": lambda: TargetSemeval2016t6Processor('FM'),
    "HC2016t6": lambda: TargetSemeval2016t6Processor('HC'),
    "LA2016t6": lambda: TargetSemeval2016t6Processor('LA'),
    # "argmin": ArgminProcessor,
    "argmin": TargetArgminProcessorall,
    "ABargmin": lambda: TargetArgminProcessor('AB'),
    "CLargmin": lambda: TargetArgminProcessor('CL'),
    "DPargmin": lambda: TargetArgminProcessor('DP'),
    "GCargmin": lambda: TargetArgminProcessor('GC'),
    "MLargmin": lambda: TargetArgminProcessor('ML'),
    "MWargmin": lambda: TargetArgminProcessor('MW'),
    "NEargmin": lambda: TargetArgminProcessor('NE'),
    "SUargmin": lambda: TargetArgminProcessor('SU'),

    }  # type: Dict[str,Callable[[],DataProcessor]]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}

METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"]
}

DEFAULT_METRICS = ["acc", "f1-macro", "all"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]


def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,
                  num_examples_per_label: int = None, seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) ^ (num_examples_per_label is not None), \
        "Exactly one of 'num_examples' and 'num_examples_per_label' must be set."
    assert (not set_type == UNLABELED_SET) or (num_examples is not None), \
        "For unlabeled data, 'num_examples_per_label' is not allowed"

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}" if num_examples is not None \
        else f"num_examples_per_label={num_examples_per_label}"
    logger.info(
        f"Creating features from dataset file at {data_dir} ({ex_str}, set_type={set_type})"
    )

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
        # print(examples)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    elif set_type == UNLABELED_SET:
        examples = processor.get_unlabeled_examples(data_dir)
        for example in examples:
            example.label = processor.get_labels()[0]
    else:
        raise ValueError(f"'set_type' must be one of {SET_TYPES}, got '{set_type}' instead")
    # for i in examples:
    #     print(i)
    if num_examples is not None:
        # print(examples)
        # for i in examples:
        #     print(i)
        # print(examples)
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    elif num_examples_per_label is not None:
        limited_examples = LimitedExampleList(processor.get_labels(), num_examples_per_label)
        for example in examples:
            limited_examples.add(example)
        examples = limited_examples.to_list()

    label_distribution = Counter(example.label for example in examples)
    logger.info(f"Returning {len(examples)} {set_type} examples with label dist.: {list(label_distribution.items())}")

    return examples

if __name__ == '__main__':
    train_data = load_examples(
        '2016t6', '../GLUE_data/SemEval2016Task6/', TRAIN_SET, num_examples=-1)
    for i in train_data:
        print(i)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:23:22 2017

@author: kohr
"""

import argparse
import numpy as np
import os
from xml.etree import ElementTree


# Initialize command line argument parser
parser = argparse.ArgumentParser(
    description='Find timing outliers in pytest runs.')
parser.add_argument('path', type=str,
                    help='path of the directory containing the XML reports')

args = parser.parse_args()

# Get parameters from arguments
path = args.path

# Collect all report files into a list
reports = []
commits = []
dates = []
for p, dirs, files in os.walk(path):
    for fname in files:
        if fname.startswith('report') and fname.endswith('.xml'):
            reports.append(os.path.join(p, fname))
            _, commit, date = fname.rstrip('.xml').split('__')
            commits.append(commit[:7])
            dates.append(date)

# Create dictionary of test run times per test name
test_times = {}
for report in reports:
    tree = ElementTree.parse(report)
    root = tree.getroot()
    for testcase in root.iter('testcase'):
        name = testcase.attrib['name']
        times = test_times.get(name, None)
        if times is not None:
            test_times[name].append(testcase.attrib['time'])
        else:
            test_times[name] = [testcase.attrib['time']]

# Compute statistics and report stuff
# TODO: choose what to report
for name, times in test_times.items():
    times = np.array(times)
    mean_time = np.mean(times)
    median_time = np.median(times)
    time_std = np.std(times)
    time_mdev = np.abs(times - median_time)



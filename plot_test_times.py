#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 16:23:22 2017

@author: kohr
"""

import argparse
import matplotlib.pyplot as plt
import os
from xml.etree import ElementTree


# Initialize command line argument parser
parser = argparse.ArgumentParser(
    description='Plot test result times from XML reports.')
parser.add_argument('path', type=str,
                    help='path of the directory containing the XML reports')
parser.add_argument('test_name', type=str,
                    help='name of the test to plot')
parser.add_argument('-x', '--xlabel', type=str, choices=['date', 'commit'],
                    default='date',
                    help='label the x axis by this')
parser.add_argument('-o', '--outfile', type=str,
                    help='save the plot in this file using matplotlib')

args = parser.parse_args()

# Get parameters from arguments
path = args.path
test_name = args.test_name
outfile = args.outfile
xlabel = args.xlabel

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

# Find run times for the specified test name
filename, testname = test_name.split(': ', maxsplit=1)
times = []
for report in reports:
    tree = ElementTree.parse(report)
    root = tree.getroot()
    for testcase in root.iter('testcase'):
        if (testcase.attrib['file'] == filename and
                testcase.attrib['name'] == testname):
            times.append(testcase.attrib['time'])
            break

# Plot stuff
fig, ax = plt.subplots()
ax.plot(times, label=test_name)
ax.legend()
ax.set_xticks(list(range(len(times))))

if xlabel == 'date':
    ax.set_xticklabels(dates, rotation='vertical')
    # Make some space on the bottom for the labels
    fig.subplots_adjust(bottom=0.5)
elif xlabel == 'commit':
    ax.set_xticklabels(commits, rotation='vertical')
    fig.subplots_adjust(bottom=0.25)
else:
    assert False

if outfile is None:
    fig.show(warn=True)
else:
    fig.savefig(outfile)

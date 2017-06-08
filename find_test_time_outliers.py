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
parser.add_argument('-t', '--threshold', type=float, default=0.0,
                    help="don't report tests with median runtime below this")
parser.add_argument('-c', '--cluster', action='store_false',
                    help='perform cluster analysis (good for finding jumps)')


args = parser.parse_args()

# Get parameters from arguments
path = args.path
threshold = args.threshold
do_clustering = args.cluster

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
        key = testcase.attrib['file'] + ': ' + testcase.attrib['name']
        times = test_times.get(key, None)
        if times is not None:
            test_times[key].append(testcase.attrib['time'])
        else:
            test_times[key] = [testcase.attrib['time']]

# Compute statistics and report stuff
# TODO: choose what to report
for name, times in test_times.items():
    times = np.array(times, dtype=float)
    median_time = np.median(times)

    # Don't report anything below the threshold
    if median_time <= threshold:
        continue

    mean_time = np.mean(times)
    time_std = np.std(times)
    time_mdev = np.abs(times - median_time)
    time_mad = np.median(time_mdev)

    if do_clustering:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3)
        features = np.hstack([np.zeros(times.size)[:, None], times[:, None]])
        clusters = kmeans.fit(features)
        cluster_centers = clusters.cluster_centers_[:, 1]
        center_median = np.median(cluster_centers)
        mdev = np.abs(cluster_centers - center_median)
        mad = np.median(mdev)
        if np.any(mdev > 5 * mad):
            fmt = ('In test {name}: found significant clustering of times '
                   'with centers {centers}. Use `plot_teset_times.py` to '
                   'analyze the test further.')
            print(fmt.format(name=name, centers=cluster_centers))
            print('')

    else:
        # Find and report outliers
        outlier_idcs = np.where(time_mdev > 10 * time_mad)[0]
        outliers = times[outlier_idcs]
        if outliers.size > 0:
            outlier_commits = np.take(commits, outlier_idcs)
            outlier_dates = np.take(dates, outlier_idcs)
            fmt = ('Outliers in test {name!r} at indices {idcs}: '
                   'median time = {median}, outliers = {outliers}, '
                   'commits = {commits}, dates = {dates}')
            print(fmt.format(name=name, idcs=outlier_idcs, median=median_time,
                             outliers=outliers, commits=outlier_commits,
                             dates=outlier_dates))
            print('')

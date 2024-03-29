import numpy as np
import pandas as pd
import pickle
# import matplotlib.pyplot as plt
from scipy.stats import binom, norm
# import matplotlib.dates as mdates

from TOAST import tape_width, tape_spacing


def col2int(arr, li):
    """

    Parameters
    ----------
    arr : list of colors (in this chunk)
    li : list of color definitions

    Returns
    -------
    list of integer representations of colors
    
    """
    # if not isinstance(arr, list):
    #     arr = [arr]
    return [li.index(v) for v in arr]  # Only for lists/arrays, not for scalars


def score_chunk(chunk, orig):
    """

    Parameters
    ----------
    chunk : list of colors in this chunk
    orig : (sub)list of colors on the stake

    Returns
    -------
    number of color matches

    """
    score = np.sum(np.array(orig) == np.array(chunk))
    return score  # /sum(x!=0 for x in chunk) # 0 is 'na'


colList = ['na', 'blu', 'wht', 'red', 'gry', 'grn', 'blk', 'yel', 'stp']
markerList = ['blu', 'wht', 'red', 'gry', 'blu', 'wht', 'red', 'grn', 'blu',
              'wht', 'red', 'yel', 'blu', 'wht', 'red', 'stp', 'blu', 'wht',
              'red', 'blk', 'blu', 'wht', 'gry', 'grn', 'blu', 'wht', 'gry',
              'yel', 'blu', 'wht', 'gry', 'stp', 'blu', 'wht', 'gry', 'blk',
              'blu', 'wht', 'grn', 'yel', 'blu', 'wht', 'grn', 'stp', 'blu',
              'wht', 'grn', 'blk', 'blu', 'wht', 'yel', 'stp', 'blu', 'wht',
              'yel', 'blk', 'blu', 'wht', 'stp', 'blk', 'blu', 'red', 'gry',
              'grn', 'blu', 'red', 'gry', 'yel', 'blu', 'red', 'gry', 'stp',
              'blu', 'red', 'gry', 'blk', 'blu', 'red', 'grn', 'yel', 'blu',
              'red', 'grn', 'stp', 'blu', 'red', 'grn', 'blk', 'blu', 'red',
              'yel', 'stp', 'blu', 'red', 'yel', 'blk', 'blu', 'red', 'stp',
              'blk', 'blu', 'gry', 'grn', 'yel', 'blu', 'gry', 'grn', 'stp',
              'blu', 'gry', 'grn', 'blk', 'blu', 'gry', 'yel', 'stp', 'blu',
              'gry', 'yel', 'blk', 'blu', 'gry', 'stp', 'blk', 'blu', 'grn',
              'yel', 'stp', 'blu', 'grn', 'yel', 'blk', 'blu', 'grn', 'stp',
              'blk', 'blu', 'yel', 'stp', 'blk', 'wht', 'red', 'gry', 'grn',
              'wht', 'red', 'gry', 'yel', 'wht', 'red', 'gry', 'stp', 'wht',
              'red', 'gry', 'blk', 'wht', 'red', 'grn', 'yel', 'wht', 'red',
              'grn', 'stp', 'wht', 'red', 'grn', 'blk', 'wht', 'red', 'yel',
              'stp', 'wht', 'red', 'yel', 'blk', 'wht', 'red', 'stp', 'blk',
              'wht', 'gry', 'grn', 'yel', 'wht', 'gry', 'grn', 'stp', 'wht',
              'gry', 'grn', 'blk', 'wht', 'gry', 'yel', 'stp', 'wht', 'gry',
              'yel', 'blk', 'wht', 'gry', 'stp', 'blk', 'wht', 'grn', 'yel',
              'stp', 'wht', 'grn', 'yel', 'blk', 'wht', 'grn', 'stp', 'blk',
              'wht', 'yel', 'stp', 'blk', 'red', 'gry', 'grn', 'yel', 'red',
              'gry', 'grn', 'stp', 'red', 'gry', 'grn', 'blk', 'red', 'gry',
              'yel', 'stp', 'red', 'gry', 'yel', 'blk', 'red', 'gry', 'stp',
              'blk', 'red', 'grn', 'yel', 'stp', 'red', 'grn', 'yel', 'blk',
              'red', 'grn', 'stp', 'blk', 'red', 'yel', 'stp', 'blk', 'gry',
              'grn', 'yel', 'stp', 'gry', 'grn', 'yel', 'blk', 'gry', 'grn',
              'stp', 'blk', 'gry', 'yel', 'stp', 'blk', 'grn', 'yel', 'stp',
              'blk']

marker_list_int = col2int(markerList, colList)

# Getting back the data:
with open('results.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    results = pickle.load(f)

# drop empty
for key, value in dict(results).items():
    if value is None:
        del results[key]
        print("No Data: " + str(key))

# step through timestamps/images, step through each chunk and put all "good"
# abstich values into dataframe
ts_list = list(results)
meas = pd.DataFrame(index=ts_list, columns=['abstiche', 'match_qual', 'marker_errors', 'scale_errors',
                                            'final', 'marker_error', 'scale_error'])
# Dangerous: results dict must be sorted by key (i.e. timestamp)
for ts in sorted(results):
    # print(ts)
    res_this = results[ts]
    res_chunks = res_this.groupby('chunk')
    abst = []
    match_qual = []
    marker_err = []
    scale_err = []
    for name, group in res_chunks:
        # print(chunk, group)
        pat = col2int(group['color'], colList)
        N = len(pat)
        match_arr = np.array(
            pd.Series(marker_list_int).rolling(window=N, min_periods=N).apply(
                lambda x: score_chunk(pat, x)))
        match_arr[0:N - 1] = 0
        match_arr = np.int8(match_arr)

        # keep 10 best with val ge 3
        ii = np.ravel(np.argwhere(match_arr >= 3))  # np.squeeze(np.argwhere(match_arr >= 3)).tolist()
        n = min(len(ii), 10)
        ii = np.array(ii)[np.argpartition(match_arr[ii], -n)[-n:]]

        if len(ii) == 0:
            continue  # jump to next chunk

        if np.isscalar(ii):
            ii = np.array([ii])  # if unique match found make list

        for i in ii:
            # i is position number on stake (from top) of lowermost marker of the best fit/s of this chunk
            # Calculate distance of lower end of pattern from stake top PLUS last mmheight
            abst.append(np.double(i) * (tape_width + tape_spacing) +
                        group.iloc[-1]['mmheight'])
            scale_err.append(group.iloc[-1]['mmheight_scale_error'])

            # number of matched colors and number of segments
            match_qual.append([match_arr[i], len(group)])

            if len(group) >= 2:
                # difference of regression-based segment width from theoretic marker spacing + width

                # # only for the bottom marker
                # marker_err.append(group.iloc[-2]['mmheight']-group.iloc[-1]['mmheight']-(tape_width + tape_spacing))

                # standard error of all markers in this group
                marker_err.append(np.sqrt(np.sum((np.diff(-group['mmheight'])
                                                  - (tape_width + tape_spacing)) ** 2) / len(group)))
            else:
                marker_err.append(np.nan)

    meas.loc[ts, 'abstiche'] = abst
    meas.loc[ts, 'match_qual'] = match_qual
    meas.loc[ts, 'marker_errors'] = marker_err
    meas.loc[ts, 'scale_errors'] = scale_err

# drop if nothing found
ii = [len(a) >= 1 for a in meas['abstiche']]
print('Dropping ' + str(len(ii) - np.count_nonzero(ii)) + ' with no abstich values defined:')
print(meas.index[~np.array(ii)])
meas = meas[ii]

# melt down to straight time series
# pick best choice in abstich
prev = None
for index, row in meas.iterrows():
    this = np.array(row['abstiche'])

    # https://stats.stackexchange.com/questions/85676/ratio-that-accounts-for-different-sample-sizes
    # n / np.sqrt(N) # the larger the better

    # Probability to match less than this number of markers
    P_colmatch = 1 - binom.cdf(np.array(row['match_qual'])[:, 0],
                               np.array(row['match_qual'])[:, 1],
                               0.8)  # binomial distribution, smaller ist better
    if prev is not None:

        # Probability for change in Abstich
        unit_mu, unit_sigma = 100., 100.  # daily values (estimates)
        delta_t = (index - prev_index).total_seconds() / 60 / 60 / 24  # days
        P_abstmatch = np.abs((norm.cdf(+(this - prev - unit_mu * delta_t),
                                       0,
                                       np.sqrt(unit_sigma ** 2 * delta_t)) -
                              norm.cdf(-(this - prev - unit_mu * delta_t),
                                       0,
                                       np.sqrt(unit_sigma ** 2 * delta_t))))

        # find best
        P = P_colmatch * P_abstmatch  # what about nasty dependence of variables?
        i_best = np.argmin(P)  # index!

        if P[i_best] <= 0.5:
            meas.loc[index, 'final'] = row['abstiche'][i_best]
            meas.loc[index, 'marker_error'] = row['marker_errors'][i_best]
            meas.loc[index, 'scale_error'] = row['scale_errors'][i_best]

            keep = row['abstiche'][i_best]
            keep_index = index

        else:
            meas.loc[index, 'final'] = np.nan
            meas.loc[index, 'marker_error'] = np.nan
            meas.loc[index, 'scale_error'] = np.nan

            keep = prev
            keep_index = prev_index
    else:
        # if first data point choose from color matching only
        i_best = np.argmin(P_colmatch)  # index!
        meas.loc[index, 'final'] = row['abstiche'][i_best]
        meas.loc[index, 'marker_error'] = row['marker_errors'][i_best]
        meas.loc[index, 'scale_error'] = row['scale_errors'][i_best]

        keep = row['abstiche'][i_best]
        keep_index = index

    prev = keep
    prev_index = keep_index

# drop NANs (from not in dAdt_range)
ii = meas['final'].notna()
print('Dropping ' + str(len(ii) - np.count_nonzero(ii)) + ' that are too far off:')
print(meas.index[~np.array(ii)])
meas = meas[ii]

# Save results:
with open('abstich.pkl', 'wb') as f:
    pickle.dump(meas, f)

print("end.")

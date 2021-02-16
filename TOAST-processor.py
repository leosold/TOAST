import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# import datetime
# import matplotlib.dates as mdates

from TOAST import tape_width, tape_spacing

max_pattern_match = 5  # maximum number of "good" pattern matches, drop if more
dAdt_range = [-30, 100]  # allowed change in mm from previous image


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
meas = pd.DataFrame(index=ts_list, columns=['abstiche', 'marker_errors', 'scale_errors',
                                            'final', 'marker_error', 'scale_error'])
# Dangerous: results dict must be sorted by key (i.e. timestamp)
for ts in sorted(results):
    print(ts)
    res_this = results[ts]
    res_chunks = res_this.groupby('chunk')
    abst = []
    marker_err = []
    scale_err = []
    for chunk, group in res_chunks:
        # print(chunk, group)
        pat = col2int(group['color'], colList)
        N = len(pat)
        match_arr = np.array(
            pd.Series(marker_list_int).rolling(window=N, min_periods=N).apply(
                lambda x: score_chunk(pat, x)))
        match_arr[0:N - 1] = 0
        match_arr = np.int8(match_arr)

        # Get positions of best pattern fitting
        ii = np.squeeze(np.argwhere(match_arr == np.max(match_arr))).tolist()
        if not isinstance(ii, list):
            ii = [ii]  # if maximum is unique
        for i in ii:
            # i is position number on stake (from top) of lowermost marker of the best fit/s of this chunk
            # Calculate distance of lower end of pattern from stake top PLUS last mmheight
            abst.append(np.double(i) * (tape_width + tape_spacing) +
                        group.iloc[-1]['mmheight'])
            scale_err.append(group.iloc[-1]['mmheight_scale_error'])
            print(group)
            if len(group) >= 2:

                # difference of regression-based segment width from theoretic marker spacing + width

                # # only for the bottom marker
                # marker_err.append(group.iloc[-2]['mmheight']-group.iloc[-1]['mmheight']-(tape_width + tape_spacing))

                # standard error of all markers in this group
                marker_err.append(np.sqrt(np.sum((np.diff(-group['mmheight'])
                                                  - (tape_width + tape_spacing)) ** 2) / len(group)))
            else:
                marker_err.append(np.nan)
            # print('mmheight of second last marker ' + str(group.iloc[-2]['mmheight']-group.iloc[-1]['mmheight']))
            # print(i)
        meas.loc[ts, 'abstiche'] = abst
        meas.loc[ts, 'marker_errors'] = marker_err
        meas.loc[ts, 'scale_errors'] = scale_err
        # for this chunk, find pattern in markerList (potentially multiple hits)
        # for every hit in markerlist: compute ABSTICH and store in list
        # per image: multiple chunks, each has multiple ABSTICH values


# drop if too many "good" matches were found
ii = [len(a) <= max_pattern_match for a in meas['abstiche']]
meas = meas[ii]
print('Dropping '+str(len(ii)-len(meas))+' that has too many pattern matches')


# melt down to straight time series
# pick closest-to-previous daily data value within range, drop otherwise
prev = None
for ts in meas.index:
    this = np.array(meas.loc[ts, 'abstiche'])
    this_marker_err = np.array(meas.loc[ts, 'marker_errors'])
    this_scale_err = np.array(meas.loc[ts, 'scale_errors'])
    if prev is not None:
        # use the abstich value that is closest to the previous one
        closest = this[np.argmin(abs(this-prev))]
        closest_marker_err = this_marker_err[np.argmin(abs(this - prev))]
        closest_scale_err = this_scale_err[np.argmin(abs(this - prev))]
        delta = closest-prev
        if dAdt_range[0] <= delta <= dAdt_range[1]:
            res = closest
            res_marker_err = closest_marker_err
            res_scale_err = closest_scale_err
            keep = res
        else:
            res = np.nan  # set NAN if difference is too large
            res_marker_err = np.nan
            res_scale_err = np.nan
            keep = prev
    else:
        # Oh Oh: for first image in series the Abstich is just chosen by order
        res = this[0]
        res_marker_err = this_marker_err[0]
        res_scale_err = this_scale_err[0]
        keep = res
    meas.loc[ts, 'final'] = res
    meas.loc[ts, 'marker_error'] = res_marker_err
    meas.loc[ts, 'scale_error'] = res_scale_err
    prev = keep

# drop NANs (from not in dAdt_range)
ii = meas['final'].notna()
meas = meas[ii]
print('Dropping '+str(len(ii)-len(meas))+' that were too far off')

# Save results:
with open('abstich.pkl', 'wb') as f:
    pickle.dump(meas, f)

# Just the plot
plt.figure(1)
plt.ylabel('Abstich (mm)')
for ts in meas.index:
    this = np.array(meas.loc[ts, 'abstiche'])
    plt.scatter(np.full(len(this), ts), this, s=2)
plt.scatter(meas.index, 'final', data=meas, s=10)
# plt.xlim(datetime.date(2020, 7, 1),datetime.date(2020, 7, 2))
plt.gcf().autofmt_xdate()
plt.show()

plt.figure(2)
plt.ylabel('Error estimate (mm)')
for ts in meas.index:
    this = np.array(meas.loc[ts, 'marker_errors'])
    plt.scatter(np.full(len(this), ts), this, s=2)
plt.scatter(meas.index, 'marker_error', data=meas, s=10)
# plt.xlim(datetime.date(2020, 7, 1),datetime.date(2020, 7, 2))
plt.gcf().autofmt_xdate()
plt.show()

plt.figure(3)
plt.ylabel('Error estimate (mm)')
for ts in meas.index:
    this = np.array(meas.loc[ts, 'scale_errors'])
    plt.scatter(np.full(len(this), ts), this, s=2)
plt.scatter(meas.index, 'scale_error', data=meas, s=10)
# plt.xlim(datetime.date(2020, 7, 1),datetime.date(2020, 7, 2))
plt.gcf().autofmt_xdate()
plt.show()

# Todo: calculate differences over time (differentiate) error estimates because this is what counts

print("end.")

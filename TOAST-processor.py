import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

tapeWidth = 19.  # width of marker tape (mm)
tapeSpacing = 21.

def int2col(arr, li):
    return [li[v] for v in arr]
def col2int(arr, li):
    # if not isinstance(arr, list):
    #     arr = [arr]
    return [li.index(v) for v in arr]  # Only for lists/arrays, not for scalars
def scoreChunk(chunk, orig):
    score = np.sum(np.array(orig) == np.array(chunk))
    return score  # /sum(x!=0 for x in chunk) # 0 is 'na'

colList = ['na', 'blu', 'wht', 'red', 'gry', 'grn', 'blk', 'yel', 'stp']
markerList = ['blu', 'wht', 'red', 'gry', 'blu', 'wht', 'red', 'grn', 'blu', 'wht', 'red', 'yel', 'blu', 'wht', 'red', 'stp', 'blu', 'wht', 'red', 'blk', 'blu', 'wht', 'gry', 'grn', 'blu', 'wht', 'gry', 'yel', 'blu', 'wht', 'gry', 'stp', 'blu', 'wht', 'gry', 'blk', 'blu', 'wht', 'grn', 'yel', 'blu', 'wht', 'grn', 'stp', 'blu', 'wht', 'grn', 'blk', 'blu', 'wht', 'yel', 'stp', 'blu', 'wht', 'yel', 'blk', 'blu', 'wht', 'stp', 'blk', 'blu', 'red', 'gry', 'grn', 'blu', 'red', 'gry', 'yel', 'blu', 'red', 'gry', 'stp', 'blu', 'red', 'gry', 'blk', 'blu', 'red', 'grn', 'yel', 'blu', 'red', 'grn', 'stp', 'blu', 'red', 'grn', 'blk', 'blu', 'red', 'yel', 'stp', 'blu', 'red', 'yel', 'blk', 'blu', 'red', 'stp', 'blk', 'blu', 'gry', 'grn', 'yel', 'blu', 'gry', 'grn', 'stp', 'blu', 'gry', 'grn', 'blk', 'blu', 'gry', 'yel', 'stp', 'blu', 'gry', 'yel', 'blk', 'blu', 'gry', 'stp', 'blk', 'blu', 'grn', 'yel', 'stp', 'blu', 'grn', 'yel', 'blk', 'blu', 'grn', 'stp', 'blk', 'blu', 'yel', 'stp', 'blk', 'wht', 'red', 'gry', 'grn', 'wht', 'red', 'gry', 'yel', 'wht', 'red', 'gry', 'stp', 'wht', 'red', 'gry', 'blk', 'wht', 'red', 'grn', 'yel', 'wht', 'red', 'grn', 'stp', 'wht', 'red', 'grn', 'blk', 'wht', 'red', 'yel', 'stp', 'wht', 'red', 'yel', 'blk', 'wht', 'red', 'stp', 'blk', 'wht', 'gry', 'grn', 'yel', 'wht', 'gry', 'grn', 'stp', 'wht', 'gry', 'grn', 'blk', 'wht', 'gry', 'yel', 'stp', 'wht', 'gry', 'yel', 'blk', 'wht', 'gry', 'stp', 'blk', 'wht', 'grn', 'yel', 'stp', 'wht', 'grn', 'yel', 'blk', 'wht', 'grn', 'stp', 'blk', 'wht', 'yel', 'stp', 'blk', 'red', 'gry', 'grn', 'yel', 'red', 'gry', 'grn', 'stp', 'red', 'gry', 'grn', 'blk', 'red', 'gry', 'yel', 'stp', 'red', 'gry', 'yel', 'blk', 'red', 'gry', 'stp', 'blk', 'red', 'grn', 'yel', 'stp', 'red', 'grn', 'yel', 'blk', 'red', 'grn', 'stp', 'blk', 'red', 'yel', 'stp', 'blk', 'gry', 'grn', 'yel', 'stp', 'gry', 'grn', 'yel', 'blk', 'gry', 'grn', 'stp', 'blk', 'gry', 'yel', 'stp', 'blk', 'grn', 'yel', 'stp', 'blk']
markerListInt = col2int(markerList, colList)

# Getting back the data:
with open('results.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    results = pickle.load(f)

# drop empty dictionary entries
for key, value in dict(results).items():
    if value is None:
        del results[key]
        print("No Data: " + str(key))

tsList = list(results)
meas = pd.DataFrame(index=tsList, columns=['abstiche','final'])
for ts in sorted(results):  # Dangerous: results dict must be sorted by key (i.e. timestamp)
    # Matching
    # step though marker segments (with color)
        # check if it has a correspondant (in previous img) that is "near" (e.g. 100px, with shift to bottom)
        # if multiple options: include all options, remove by comparing to 1.5xIQR
    print(ts)
    resThis = results[ts]
    resChunks = resThis.groupby('chunk')
    abst = []
    for chunk, group in resChunks:
        # print(chunk, group)
        pat = col2int(group['color'], colList)
        N = len(pat)
        matchArr = np.array(pd.Series(markerListInt).rolling(window=N, min_periods=N).apply(lambda x: scoreChunk(pat, x)))
        matchArr[0:N-1]=0
        matchArr = np.int8(matchArr)
        # matchArr = pd.Series(markerListInt).rolling(window=N, min_periods=N).apply(lambda x: (x == pat).all())
        ii = np.squeeze(np.argwhere(matchArr == np.max(matchArr))).tolist()  # np.where(matchArr >= 0.99)
        if not isinstance(ii, list):
            ii = [ii] # if maximum is unique
        for i in ii:
            abst.append(i*(tapeWidth+tapeSpacing)+group.iloc[-1]['mmheight']) # Distance of lower end of pattern from stake top PLUS last mmheight
            # print(i)
        meas.loc[ts, 'abstiche'] = abst
        # for this chunk, find pattern in markerList (potentially multiple hits)
        # for every hit in markerlist: cumpute ABSTICH and store in list
        # per image: multiple chunks, each has multiple ABSTICH values

    # for i in range(len(resThis)):
    #     print(resThis.loc[i])
        # print(resThis.loc[i, 'color'])
    # Check if pattern is complete, i.e. if height+width gives next segment height
    print("hh")


prev = None
for ts in meas.index:
    this = np.array(meas.loc[ts,'abstiche'])
    if prev is not None:
        # use the abstich value that is closest to the previous one
        closest = this[np.argmin(abs(this-prev))]
        delta = closest-prev
        if -30 <= delta <= 100:  # set NAN if difference is too large
            res = closest
            keep = res
        else:
            res = np.nan
            keep = prev
    else:
        res = this[0]
        keep = res
    meas.loc[ts, 'final'] = res
    prev = keep

# # plt.figure()
# dates = matplotlib.dates.date2num(meas.index())
# meas.plot(style='k.')
# plt.show()

plt.ylabel('Abstich (mm)')
# for ts in meas.index:
#     this = np.array(meas.loc[ts,'abstiche'])
#     plt.scatter(np.full(len(this), ts), this)
plt.scatter(meas.index, 'final', data=meas)
plt.gcf().autofmt_xdate()
plt.show()
print("t")
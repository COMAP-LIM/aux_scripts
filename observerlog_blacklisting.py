import requests as rs
import csv
import numpy as np

# "Reasons" (from the google sheet) which OVRO lists as fine, but we want to remove:
additional_blacklists = ["New firmware, old clock 2.00 GHz, details in the pathfinder log",
                         "New firmware, new clock 2.10 GHz, details in the pathfinder log"]

# link to the google sheet containing blacklisted obsids. maintaned by OVRO:
url = "https://docs.google.com/spreadsheets/d/1ab23NlqiUetoygd6PWlbmwtgF70V1WcOhUhFo5XtSWo/export?format=csv&id=1ab23NlqiUetoygd6PWlbmwtgF70V1WcOhUhFo5XtSWo&gid=0"

with rs.get(url=url) as res:
    open("google.csv", "wb").write(res.content)

blacklist = []
blacklist_reason = []
blacklist_mask = np.zeros((100000, 20, 4), dtype=bool)
with open("google.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    next(csv_reader)
    for row in csv_reader:
        startID = int(row[0])
        stopID = row[1]
        feeds = row[2]
        sidebands = row[3]
        if sidebands == "all":
            masked_sbs = (0,1,2,3)
        elif sidebands == "bandA":
            masked_sbs = (0,1)
        elif sidebands == "bandB":
            masked_sbs = (2,3)
        mask = np.zeros((20,4), dtype=bool)
        try:
            feed = int(feeds)
            mask[feed-1,masked_sbs] = True
        except:
            pass
        if feeds == "all":
            mask[:,masked_sbs] = True
        if "," in feeds:
            feeds = feeds.split(",")
            for feed in feeds:
                mask[int(feed)-1,masked_sbs] = True
        try:
            dataflag = int(row[4])
        except:
            dataflag = 0
        reason = row[5]
        if stopID.isdigit():
            stopID = int(stopID)
        elif stopID == "":
            stopID = startID
        elif stopID == "ongoing":
            stopID = 99999

        if dataflag == 0 or reason in additional_blacklists:
            for i in range(startID, stopID+1, 1):
                blacklist.append(i)
                blacklist_reason.append(reason)

                blacklist_mask[i] |= mask

with open("/mn/stornext/d16/cmbco/comap/data/aux_data/blacklists/blacklist_observerlog.txt", "w") as f:
    for i in range(len(blacklist)):
        f.write(f"{blacklist[i]} {blacklist_reason[i]}\n")
np.save("/mn/stornext/d16/cmbco/comap/data/aux_data/blacklists/blacklist_observerlog.npy", np.array(blacklist))
np.save("/mn/stornext/d16/cmbco/comap/data/aux_data/blacklists/blacklist_mask_observerlog.npy", np.array(blacklist_mask))
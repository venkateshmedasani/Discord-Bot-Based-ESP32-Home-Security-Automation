import os
import datetime
import schedule


def clear_recordings():
    recordings = os.listdir('recorded_videos/')
    for recording in recordings:
        os.remove(recording)

schedule.every(5).seconds.do(clear_recordings)


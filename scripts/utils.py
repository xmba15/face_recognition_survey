#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess


def get_usb_cam():
    task = subprocess.Popen("ls /dev | grep video",
                            shell = True,
                            stdout = subprocess.PIPE)

    list_video = task.stdout.read().decode("utf-8")
    if list_video == "":
        return None
    list_video = list_video.strip().split('\n')

    return list_video

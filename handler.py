# This is just a quick tool to run two scrips for me

import os
import sys
import subprocess

def main():
    file1 = 'test_pipeline.py'
    file2 = 'data_handler.py'

    subprocess.call(['python', file1])
    subprocess.call(['python', file2])

main()

import sys
import traceback

import os
import numpy
import shutil


numpy.set_printoptions(linewidth=255)

class TracePrints(object):
	def __init__(self):
		self.stdout = sys.stdout
	def write(self, s):
		self.stdout.write("Writing %r\n" % s)
		traceback.print_stack(file=self.stdout)

def clean_images_directory():
	imageDir = 'images'
	if os.path.exists(imageDir):
		shutil.rmtree('images')
	os.mkdir(imageDir)

def findPrintStatements():
	sys.stdout = TracePrints()





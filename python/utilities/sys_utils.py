
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

def delete_all_iamges(directory):
	# shutil.rmtree('images')
	for root, dirs, files in os.walk(directory, topdown=False):
		for name in files:
			os.remove(os.path.join(root, name))
		for name in dirs:
			os.rmdir(os.path.join(root, name))

def clean_images_directory():
	imageDir = 'images'
	if os.path.exists(imageDir):
		delete_all_iamges(imageDir)
	else:
		os.mkdir(imageDir)

def findPrintStatements():
	sys.stdout = TracePrints()


# probably not the best place for this method
def get_plot_size():
	return 15

def createObject():
	return type('', (object,), {"foo": 1})()
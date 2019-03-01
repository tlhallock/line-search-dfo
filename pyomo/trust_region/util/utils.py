
import numpy as np
import json


def write_json_to_file(object, path):
	with open(path, 'w') as outfile:
		write_json(object, outfile)


def write_json(object, outfile, also_print=False):
	class NumpyEncoder(json.JSONEncoder):
		def default(self, obj):
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			if type(obj).__name__ == "bool_":
				return bool(obj)
			return json.JSONEncoder.default(self, obj)

	string_to_write = json.dumps(object, indent=2, cls=NumpyEncoder)
	if also_print:
		print(string_to_write)
	outfile.write(string_to_write)


class Stack:
	def __init__(self):
		self.items = []

	def is_empty(self):
		return self.items == []

	def push(self, item):
		self.items.append(item)

	def pop(self):
		return self.items.pop()

	def peek(self):
		return self.items[len(self.items) - 1]

	def size(self):
		return len(self.items)

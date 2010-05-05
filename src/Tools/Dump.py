import os
import pickle


import config

class Dump:
	""" Previously parsed snippets, for templates """
	def __init__(self):
		pass

	@staticmethod
	def exists(name):
		""" Check if a given dump exists """
		return os.path.isfile(config.WORKDIR + 'freezer/' + name)


	@staticmethod
	def load(name):
		""" Load a frozen AST """
		file = open(config.WORKDIR + 'freezer/' + name, 'r') 
		tree =  pickle.load(file)
		file.close()
		return tree

	@staticmethod
	def save(name, tree):
		""" Freeze an AST """
		file = open(config.WORKDIR + 'freezer/' + name, 'w+') 
		pickle.dump(tree, file)
		file.close()

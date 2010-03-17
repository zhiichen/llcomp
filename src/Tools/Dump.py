import os
import pickle

class Dump:
	""" Previously parsed snippets, for templates """
	def __init__(self):
		pass



	@staticmethod
	def exists(name):
		""" Check if a given dump exists """
		return os.path.isfile('freezer/' + name)


	@staticmethod
	def load(name):
		""" Load a frozen AST """
		file = open('freezer/' + name, 'r') 
		tree =  pickle.load(file)
		file.close()
		return tree

	@staticmethod
	def save(name, tree):
		""" Freeze an AST """
		file = open('freezer/' + name, 'w+') 
		pickle.dump(tree, file)
		file.close()

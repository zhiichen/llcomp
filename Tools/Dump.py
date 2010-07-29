import os
import pickle


import config

class Dump:
	"""Serializer class for Internal Representation """
	def __init__(self):
		pass

	@staticmethod
	def exists(name):
		"""Check if a given tree serialization exists 
            
             :param name: Name of the frozen tree
        """
		return os.path.isfile(config.WORKDIR + name)


	@staticmethod
	def load(name):
		"""Deserialize a frozen AST 

             :param name: Name of the frozen tree
        """
		file = open(config.WORKDIR + name, 'r') 
		tree =  pickle.load(file)
		file.close()
		return tree

	@staticmethod
	def save(name, tree):
		"""Serialize an AST 

            :param name: Name of the frozen tree
        """
		file = open(config.WORKDIR + name, 'w+') 
		pickle.dump(tree, file)
		file.close()

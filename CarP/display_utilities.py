from .utilities import *
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns #this is for heatmaps
import texttable as tx #this is for commandline table output
import sys #this is for changing the form of output to write to files

def generate_heatmap( matrix, max_value=0.5): 
	"""	Returns a heatmap of the 2D matrix

	:param matrix: (list) 2D list 
	:param max_value: a number for a max cutoff value to be represented in the heatmap
	"""
	numpy_matrix = np.array(matrix)
	#print(numpy_matrix.ndim)
	#print(numpy_matrix.shape)
	with sns.axes_style("white"):
		f, ax = plt.subplots(figsize=(12, 10)) #change size of figure here
		ax = sns.heatmap(numpy_matrix,linewidths=.2 , vmax=max_value, square=True)

#? maybe make this built into the conf_space objects, saves making a copy and some functions
def return_2d_lists(conf_space_object): 
	""" Returns 3 different 2D lists calculating rmsd, rmsd without considering H atoms and pendry R factor

	:param conf_space_object: an initialized conformer space, this should be a list of conformer objects containing molecule data
	:return: (list) 3 different 2D lists
	"""
	rmsd_all_atoms=[]
	rmsd_no_hydrogen=[]
	pendry_all=[]
	for d1 in range(len(conf_space_object)):
		rmsd_all_atoms.append([])
		rmsd_no_hydrogen.append([])
		pendry_all.append([])
		for d2 in range(len(conf_space_object)):
			rmsd_all_atoms[d1].append(calculate_rmsd(conf_space_object[d1], conf_space_object[d2]))
			rmsd_no_hydrogen[d1].append(calculate_rmsd(conf_space_object[d1],conf_space_object[d2],'H'))
			pendry_all[d1].append(rfac(conf_space_object[d1].IR,conf_space_object[d2].IR)) #this one takes forever
	return rmsd_all_atoms, rmsd_no_hydrogen, pendry_all

def make_plots(conf_space_object, index=0, bar=True, scatter=True): 
	""" Displays plots

	:param conf_space_object: an initialized conformer space, this should be a list of conformer objects containing molecule data
	:param index: (int) identifies which conformer in the list of conformers, the default is the first conformer at index 0 
	:param bar: (bool) specifies if bar plots should be generated, this is a triple bar plot (rmsd, rmsd without H, pendry factor); default set to TRUE
	:param scatter: (bool) specifies if scatterplot should be generated, there will be 3 things plotted (rmsd, rmsd without H, pendry factor); default set to TRUE
	"""
	#!!! CHECK IF INDEX IS WITHIN RANGE
	molecule_ids=[] ; molecule_names=[] ; rmsd_all=[] ; rmsd_no_H=[] ; pendry=[]
	for i in range(len(conf_space_object)):
		molecule_ids.append(i)
		molecule_names.append(conf_space_object[index]._id)
		rmsd_all.append(calculate_rmsd(conf_space_object[index],conf_space_object[i]))
		rmsd_no_H.append(calculate_rmsd(conf_space_object[index],conf_space_object[i],'H'))
		pendry.append(rfac(conf_space_object[index].IR,conf_space_object[i].IR))
		#the pendry function prints 2 numbers, I'm only returning the second right now. 
        #I'm not sure what the first represents but the second is more comparable in magnitude to the rmsd values.
	
	#outputs the table
	display_table(index,molecule_ids,molecule_names,rmsd_all,rmsd_no_H,pendry)

	#check to make bar
	if bar == True:
		ind = np.arange(len(molecule_ids))  # the x locations for the groups
		#print(ind)
		width = 0.35  # the width of the bars
		fig, ax = plt.subplots()
		rects1 = ax.bar(ind - width/2, rmsd_all, width, label='rmsd hydrogens')
		rects2 = ax.bar(ind + width/2, rmsd_no_H, width, label='rmsd without hydrogens')
		#rects3 = ax.bar(ind + width, pendry, width, label='pendry')
		#Add some text for labels, title and custom x-axis tick labels, etc.
		ax.set_ylabel('Value')
		ax.set_title((conf_space_object[index]._id+' compared to all'))
		ax.set_xticks(ind)
		ax.set_xticklabels(molecule_ids)
		ax.legend()
		plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')
		fig.tight_layout()
		plt.show()
		fig.savefig((conf_space_object[index]._id+'rmsd.png'), dpi=200) #? specifiy file name using the specified index
	
	#check to make scatter
	if scatter==True:
		d1 = (rmsd_all, pendry)
		d2 = (rmsd_all, rmsd_all)
		d3 = (rmsd_all, rmsd_no_H)
		data = (d1, d2, d3)
		colors = ("red", "green", "blue")
		groups = ("pendry", "rmsd_all", "rmsd_no_H")

		# Create plot
		fig = plt.figure()
		fig.set_size_inches(10, 5)
		ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
		for data, color, group in zip(data, colors, groups):
			x, y = data
			ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
			z = np.polyfit(x, y, 1)
			p = np.poly1d(z)
			print ("y=%.6fx+(%.6f)"%(z[0],z[1]))
			line_name = "y=%.6fx+(%.6f)"%(z[0],z[1])
			ax.plot(x,p(x),c=color, label=line_name)
		plt.title('Matplot scatter plot')
		plt.legend(loc=2)
		plt.show()


def display_table(index,molecule_ids,molecule_names,rmsd_all,rmsd_no_H,pendry):
	""" Prints into terminal and saves as .txt a table displaying each conformer and the associated value rmsd, rmsd without H and pendry R factor

	:param index: (int) all the values are calculated comparing all conformers to one conformer which is sepcified by this index
	:param molecule_ids: (list) the list of molecule id number
	:param molecule_names: (list) the lost of molecule names
	:param rmsd_all: (list) list of rmsd values
	:param rmsd_no_H: (list) list of rmsd values without H
	:param pendry: (list) list of all pendry values
	"""
	index_vs_all=[] #comparing every conformer to the conformer specified by index
	index_vs_all.append(["index",("comparing molecules to "+molecule_names[index]),"rmsd","rmsd hydrogens removed","pendry"])
	#? maybe make this csv writing thing a separate function
	#i need to clear the file i'm appending to so that i dont keep the old values
	with open('rmsd.csv','a') as file:
		file.truncate(0)
	for i in range(len(molecule_ids)):
		temp=[]
		temp.append(i)
		temp.append(molecule_names[i])
		temp.append(rmsd_all[i])
		temp.append(rmsd_no_H[i])
		temp.append(pendry[i]) 
		index_vs_all.append(temp)
		#Writing this as a csv
		original_output = sys.stdout #save ref of original output to reset later
		with open('rmsd.csv','a') as file:
			sys.stdout = file #set output to the file
			print(str(i)+','+str(molecule_names[i])+','+str(rmsd_all[i])+','+str(rmsd_no_H[i])+','+str(pendry[i]))
			sys.stdout = original_output #reset output stream        
	table = tx.Texttable()
	#table.header(["comparing molecules to","rmsd","rmsd hydrogens removed"]) #idk why this not working, got it from https://pypi.org/project/texttable/ documentation
	table.set_cols_dtype(['a','t','f','f','f']) #identifies the type of each item in the matrix
	#table.set_cols_align(["l", "r", "r", "r", "l"])
	table.add_rows(index_vs_all)
	print (table.draw())






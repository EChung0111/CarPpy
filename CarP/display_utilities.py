from .utilities import *
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import sys #this is for changing the form of output to write to files

def generate_heatmap(matrix, max_value=0.5, color='viridis'):
    """ Returns a heatmap of the 2D matrix using pure matplotlib
    :param matrix: (list) 2D list
    :param max_value: a number for a max cutoff value to be represented in the heatmap
    """
    numpy_matrix = np.array(matrix)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))  # change size of figure here
    
    # Create heatmap
    im = ax.imshow(numpy_matrix, cmap=color, vmax=max_value)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    
    # Create white grid to mimic seaborn's linewidths
    ax.set_xticks(np.arange(-.5, numpy_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, numpy_matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Make it square
    ax.set_aspect('equal')
    
    return fig, ax

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

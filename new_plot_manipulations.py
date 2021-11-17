import numpy as np 

###### SARIK'S METHODS ^^^^^^^^ ######
######   OUR METHODS   VVVVVVVV #######
def insert_antonym_2(self, plots):
		sents_plots = plots.split('#')
		new_sents_plots = []

		plots_with_antonyms = []
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			sent_plots = [i for i in sent_plots if i]
			for plt in sent_plots:
				if plt in list(self.plt_antonyms.keys()) and plt not in plots_with_antonyms:
					plots_with_antonyms.append(ind) 
					break

		num_plt_to_add = math.ceil((50*len(plots_with_antonyms)) / 100)
		random_plts = np.random.choice(plots_with_antonyms, size=num_plt_to_add, replace=False) #np.arange(1,len(sents_plots)-1)
		#print("RANDOM PLOTS: ",random_plts)
		checkpoint = 0
		for plot_index in random_plts:
			new_sents_plots.extend(sents_plots[checkpoint:plot_index])
			plots = sents_plots[plot_index].strip().split('\t')
			plots = [i for i in plots if i]
			antonyms = [np.random.choice(self.plt_antonyms.get(plt,[plt]),size=1,replace=False)[0] for plt in plots]
			antonyms = '\t'.join(antonyms)
			new_sents_plots.extend([antonyms,sents_plots[plot_index],antonyms])
			checkpoint = plot_index+1
		new_sents_plots.extend(sents_plots[checkpoint:])
		new_plots = '#'.join(new_sents_plots)	
		if new_plots.startswith('\t'):
			new_plots = '\t'.join(new_plots.split('\t')[1:])
		return new_plots	


def contradiction_repetition():
    # take entire plot 
    # get antonym for each and join(found wallet floor -> "lost purse ceiling" instead of "found purse wallet floor"
    # insert it to the front and back as a new plot (i.e. neg plot..but..pos plot..and..neg plot)
    return



def plot_manipulation_loop():
    # do a subset of plot manipulations 
    # get the new implausible output story
    # extract new plots from it
    # do manupulations on the new plots and repeat
    return 



import numpy as np 

###### SARIK'S METHODS ^^^^^^^^ ######
######   OUR METHODS   VVVVVVVV #######
# take entire plot 
# get antonym for each and join(found wallet floor -> "lost purse ceiling" instead of "found purse wallet floor"
# insert it to the front and back as a new plot (i.e. neg plot..but..pos plot..and..neg plot)

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

def contradiction_LogicalReordereing(self, plots, text):
		sents_plots = plots.split('#')[:5]
		new_sents_plots = sents_plots
		list_plts_verb,list_plts_verb_tense = [],[]    
	
		sents = text.split('</s>')[1:]
		sents_pos = {i:nltk.pos_tag(nltk.word_tokenize(sent)) for i,sent in enumerate(sents)}
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			for plt in sent_plots:
				plt = plt.strip()
				pos_plt = None
				if ' ' not in plt:
					sentence = sents_pos[ind]
					sentence_tokens,sentence_tags = [s[0] for s in sentence],[s[1] for s in sentence]
					try:	pos_plt= sentence_tags[sentence_tokens.index(plt)]
					except ValueError:	pos_plt=None
				else:
					sentence = nltk.pos_tag(nltk.word_tokenize(plt))
					sentence_tags = [s[1] for s in sentence]
					has_vb = [pos if('VB' in pos) else None for pos in sentence_tags]
					has_vb = list(filter(None, has_vb))
					if(len(has_vb)>0):
						pos_plt = has_vb[0]		
				lemmatizer = WordNetLemmatizer()		
				if pos_plt \
					and pos_plt in ['VBD', 'VB',  'VBZ', 'VBP']\
					and (plt in list(self.plt_antonyms.keys()) or lemmatizer.lemmatize(plt,pos="v") in list(self.plt_antonyms.keys()))\
					and plt not in list_plts_verb :
						list_plts_verb.append(plt)
						list_plts_verb_tense.append(pos_plt)
		if list_plts_verb == []:
			return plots

		num_plt_to_add = math.ceil((15*len(list_plts_verb)) / 100)
		random_plts = np.random.choice(list_plts_verb, size=num_plt_to_add, replace=False)

			
		for random_plt in random_plts:
			random_plt = random_plt.strip()
			ind_input_event_text = list_plts_verb.index(random_plt)
			tense_input_event_text = list_plts_verb_tense[ind_input_event_text]
			random_plt_tokens = nltk.word_tokenize(random_plt)
			if len(random_plt_tokens)>1: #get the verb from random_plt and change it to its simple tense  #for an ngram plot it returns all the verbs that exist
				random_plt_pos = nltk.pos_tag(random_plt_tokens)
				ind_verb_random_plt_pos = [i if('VB' in pos[1]) else None for i,pos in enumerate(random_plt_pos)]
				ind_verb_random_plt_pos = list(filter(None, ind_verb_random_plt_pos))
				if len(ind_verb_random_plt_pos) != 1:#only select a plot from plot_list that has exactly one verb
					continue
				random_plt_verb_simple_tense = nlp(random_plt_tokens[ind_verb_random_plt_pos[0]])[0]._.inflect('VB')	
			else: #single word subplot
				random_plt_verb_simple_tense = nlp(random_plt_tokens[0])[0]._.inflect('VB')

			print("PRE_OUTS",random_plt)
			relation = np.random.choice(['HasPrerequisite', 'Causes', 'HasFirstSubevent', 'HasLastSubevent'], size=1, replace=False)[0]
			antonym_plt = None
			if(self.plt_antonyms.get(random_plt,None)):
				antonym_plt = self.plt_antonyms[random_plt]
			else:
				antonym_plt = self.plt_antonyms[lemmatizer.lemmatize(random_plt,pos="v")]
			antonym_plt = np.random.choice(antonym_plt, size=1, replace=False)[0]
			outputs = interactive.get_conceptnet_sequence(antonym_plt, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation) #extract concepts that have specified relation with the random_plt (subject plot) in COMET
			print("OUTS",outputs)
			for output in outputs[list(outputs.keys())[0]]['beams']:
				print(output)
				output_event = ''
				output = output.replace('your', '')
				output = output.replace('you', '')
				
				if  output != '' and random_plt_verb_simple_tense != None and random_plt_verb_simple_tense not in output:#Object extracted from COMET should not include the verb which is in the subject plot
					if ' ' not in output and nlp(output)[0]._.inflect('VB') != None: #output is a verb type unigram
						output_event = nlp(output)[0]._.inflect(tense_input_event_text).strip() #change output to the tense of subject plot
					elif ' ' in output:
						output_tokens = nltk.word_tokenize(output)
						output_pos = nltk.pos_tag(output_tokens)
						ind_verb_output_pos = [i if('VB' in pos[1]) else None for i,pos in enumerate(output_pos)]
						ind_verb_output_pos = list(filter(None, ind_verb_output_pos))
						if ind_verb_output_pos == [] or len(ind_verb_output_pos) != 1: #skip if none of the tokens in the object ngram is a verb or it has more than one verb
							continue
						verb_chg_tense = nlp(output_tokens[ind_verb_output_pos[0]])[0]._.inflect(tense_input_event_text) #change the tense of object to be compatible with subject tense
						if verb_chg_tense == None:
							continue
						output_tokens[ind_verb_output_pos[0]] = verb_chg_tense
						output_event = ' '.join(output_tokens).strip()
					else:
						continue								
					break
			print("\nOUTPUT EVENT: ",output_event)
			if output_event == '':
				continue

			plts = [(s.split('\t').index(random_plt),i) if random_plt in s.split('\t') else None for i,s in enumerate(sents_plots)]
			plot_ind,ind = list(filter(None, plts))[0][0],list(filter(None, plts))[0][1]
			#concat_word = np.random.choice(['then', '', 'later', 'subsequently'], size=1)[0]#concat_word = ''
			if relation == "HasPrerequisite" or relation== "HasFirstSubevent": #if concat_word == '':
				changing_plots = random_plt  + '\t' + output_event	 #else: changing_plots = random_plt  + ' ' + concat_word + ' ' + output_event	
			elif relation == "Causes" or relation=="HasLastSubevent": #if concat_word == '':  
				changing_plots = output_event  + '\t' + random_plt #else:	changing_plots = output_event  + ' '+ concat_word + ' ' + random_plt
			sents_plots[ind] = '\t'.join(sents_plots[ind].split('\t')[:plot_ind]) + '\t' + changing_plots  + '\t' + '\t'.join(sents_plots[ind].split('\t')[plot_ind+1:])
			new_sents_plots[ind] = sents_plots[ind]

		new_sents_plots = '#'.join(new_sents_plots)
		if new_sents_plots.startswith('\t'):
			new_sents_plots = '\t'.join(new_sents_plots.split('\t')[1:])
		return new_sents_plots

def random_deletion(self, plots):
		sents_plots = plots.split('#')
		new_sents_plots = []
		num_plt_to_add = math.ceil((5*len(sents_plots)) / 100)
		random_plts = np.random.choice(len(sents_plots), size=num_plt_to_add, replace=False) #np.arange(1,len(sents_plots)-1)
		#print("RANDOM PLOTS: ",random_plts)
		checkpoint = 0
		for plot_index in random_plts:
			new_sents_plots.extend(sents_plots[checkpoint:plot_index])
			plots = sents_plots[plot_index].strip().split('\t')
			plots_ = []
			for p in plots:
				subplot = p
				if(" " in p):
					choice_to_del = np.random.choice(len(p.split(' ')),size=1,replace=False)
					subplot.remove(choice_to_del)
				plots_.append(subplot)
			plots = '\t'.join(plots_)
			new_sents_plots.append(plots)
			checkpoint = plot_index+1
		new_sents_plots.extend(sents_plots[checkpoint:])
		new_plots = '#'.join(new_sents_plots)	
		if new_plots.startswith('\t'):
			new_plots = '\t'.join(new_plots.split('\t')[1:])
		return new_plots
 

def plot_manipulation_loop():
    # do a subset of plot manipulations 
    # get the new implausible output story
    # extract new plots from it
    # do manupulations on the new plots and repeat
    return 



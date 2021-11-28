import torch
import os
import numpy as np
import nltk
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
import math
import spacy
import pyinflect
import argparse 
import src.data.data as data
import src.data.config as cfg
import src.interactive.functions as interactive
nlp = spacy.load('en_core_web_sm')
np.random.seed(100)


class Plt_manipulations():
	def __init__(self, COMET_model_file, COMET_sampling_algorithm, device):
		print("FILE: ",COMET_model_file)
		opt, state_dict = interactive.load_model_file(COMET_model_file)
		print("OPT: ",opt)
		self.data_loader, self.text_encoder = interactive.load_data("conceptnet", opt)
		n_ctx = self.data_loader.max_e1 + self.data_loader.max_e2 + self.data_loader.max_r
		n_vocab = len(self.text_encoder.encoder) + n_ctx
		self.comet_model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
		sampling_algorithm = COMET_sampling_algorithm	
		if device != "cpu":
			cfg.device = int(device)
			cfg.do_gpu = True
			torch.cuda.set_device(cfg.device)
			self.comet_model.cuda(cfg.device)
		else:
			cfg.device = "cpu"
		self.sampler = interactive.set_sampler(opt, sampling_algorithm, self.data_loader)

		fr = open("/content/drive/My Drive/Colab Notebooks/NLP_Project/base_project/Data_/conceptnet_antonym.txt", "r")	
		#self.conceptnet_antonyms = fr.readlines()
		anotomy_word = {}
		for line in fr.readlines():
			tmp = line.strip().split("|||")
			if len(tmp) == 3:
				h, t = tmp[0], tmp[2].split()
			if h in anotomy_word:
				anotomy_word[h] += t
			else:
				anotomy_word[h] = t[:]
		self.plt_antonyms = anotomy_word

	
	#NEW METHODS
	def change_location(self, plots):
		location_adverbs = ["about","above","abroad","anywhere","away","along","back","backwards","backward","below","below","behind","upstairs"
							"down","downstairs","elsewhere","far","here","in","indoors","inside","near","nearby","off","on","out","outside","under",
							"overseas","somewhere","there","right","left","off","east","west","north","south","southwest","southeast","underground","over"]
		sents_plots = plots.split('</s>')#,story.split('</s>')
		#assert len(sents) == len(sents_plots)
		#sents_pos = {i:nltk.pos_tag(nltk.word_tokenize(sent)) for i,sent in enumerate(sents)}
		outer_adverbs,outer_nouns,modified_plots = [],[],[]
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			adverbs,nouns,all = [],[],[]
			for ind2,plt in enumerate(sent_plots):
				sub_plots = plt.strip().split(' ')
				sub_adverbs,sub_nouns,all2 = [],[],[]
				for s_p in sub_plots:
					if(s_p ==""):
						s_p=" "
					all2.append(s_p)
					plot_pos = nltk.pos_tag([s_p])[0][1]
					if(plot_pos.startswith("NN")):
						sub_nouns.append(s_p) #(ind,ind2,ind3,s_p)
					elif(plot_pos=="RB" and s_p in location_adverbs):
						sub_adverbs.append(s_p)
				adverbs.append(sub_adverbs)
				nouns.append(sub_nouns)
				all.append(all2)
			outer_adverbs.append(adverbs)
			outer_nouns.append(nouns)
			modified_plots.append(all)
		if len(outer_adverbs) == 0: 
			return plots

		#check if there are two nouns and a location adverb
		outer_level,mid_level,sub_level = [],[],[]
		out_list_adv,out_list_noun = [],[]
		for ind1,(i1_adv,i1_noun) in enumerate(zip(outer_adverbs,outer_nouns)): #plots
			mid_list_adv,mid_list_noun = [],[]
			for ind2,(i2_adv,i2_noun) in enumerate(zip(i1_adv,i1_noun)): #over one plot
				if(len(i2_adv)>=1 and len(i2_noun)>=2):
					sub_level.append(((ind1,ind2),i2_adv,i2_noun))
				mid_list_adv.extend(i2_adv)
				mid_list_noun.extend(i2_noun)
				#for i3_adv,i3_noun in zip(i2_adv,i2_noun): #subplot	
			if(len(mid_list_adv)>=1 and len(mid_list_noun)>=2):
				mid_level.append((ind1,mid_list_adv,mid_list_noun))
			out_list_adv.extend(mid_list_adv)
			out_list_noun.extend(mid_list_noun)
		if(len(out_list_adv)>=1 and len(out_list_noun)>=2):
			outer_level.append((-1,out_list_adv,out_list_noun))
		
		output = ""
		relation = "AtLocation"
		if(len(sub_level)>0):
			for candidate_loc,advs,nouns in sub_level:
				to_modify = modified_plots[candidate_loc[0]][candidate_loc[1]]
				adverb = advs[0]
				noun1,noun2 = nouns[0],nouns[1]
				noun1_idx = to_modify.index(noun1)
				noun2_idx = to_modify.index(noun2)
				output1 = interactive.get_conceptnet_sequence(noun1, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
				output1 = output1[list(output1.keys())[0]]['beams'][0]
				output2 = interactive.get_conceptnet_sequence(noun2, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
				output2 = output2[list(output2.keys())[0]]['beams'][0]
				to_modify[noun1_idx] = " ".join([noun1,adverb,output2])
				to_modify[noun2_idx] = " ".join([noun2,adverb,output1])
				modified_plots[candidate_loc[0]][candidate_loc[1]] = to_modify
			out1 = []
			for mp in modified_plots:
				out2 = []
				for mp2 in mp:
					out2.append(" ".join(mp2))
				out1.append("\t".join(out2))
			output = "\t</s>\t".join(out1)
		elif(len(mid_level)>0):
			for candidate_loc,advs,nouns in mid_level:
				to_modify = modified_plots[candidate_loc]
				if(to_modify is None):
					continue
				print(to_modify)
				to_modify_flat = []
				for t_m in to_modify: 
					to_modify_flat.extend(t_m)#.append("\t")
				adverb = advs[0]
				noun1,noun2 = nouns[0],nouns[1]
				noun1_idx = to_modify_flat.index(noun1)
				noun2_idx = to_modify_flat.index(noun2)
				#print("NNs: ",noun1,noun2)
				output1 = interactive.get_conceptnet_sequence(noun1, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
				#print("OUT1: ",output1)
				output1 = output1[list(output1.keys())[0]]['beams'][0]
				#print("OUT11: ",output1)
				output2 = interactive.get_conceptnet_sequence(noun2, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
				#print("OUT2: ",output2)
				output2 = output2[list(output2.keys())[0]]['beams'][0]
				#print("OUT22: ",output2)
				output_ = ("\t".join(to_modify_flat[:noun1_idx])+"\t"+" ".join([noun1,adverb,output2])+"\t"+"\t".join(to_modify_flat[noun1_idx+1:noun2_idx])+"\t"+" ".join([noun2,adverb,output1])+"\t"+"\t".join(to_modify_flat[noun2_idx+1:]))
				#print(output_)
				modified_plots[candidate_loc] = output_
			out1 = []
			#print("MP: ",modified_plots)
			for mp in modified_plots:
				out2 = []
				for mp2 in mp:
					out2.append(" ".join(mp2))
				if(isinstance(mp, list)):
					out1.append("\t".join(out2))
				else:
					out1.append(mp)
			output = "\t</s>\t".join(out1)
		else:
			output = plots
		if output.startswith('\t'):
			output = '\t'.join(output.split('\t')[1:])
		return output

	def property_switching(self,plots):
		sents_plots = plots.split('</s>')#,story.split('</s>')
		nouns,modified_plots = [],[]
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			all = []
			for ind2,plt in enumerate(sent_plots):
				nouns_temp = nouns
				sub_plots = plt.strip().split(' ')
				all2 = []
				for ind3,s_p in enumerate(sub_plots):
					if(s_p ==""):
						s_p=" "
					all2.append(s_p)
					plot_pos = nltk.pos_tag([s_p])[0][1]
					if(plot_pos.startswith("NN") and s_p != " "):
						nouns.append((s_p,ind,ind2,ind3)) #
						break
				all.append(all2)
				if(len(nouns)>len(nouns_temp)): break
			modified_plots.append(all)
		if len(nouns) == 0: 
			return plots
		properties = []
		relation = "hasProperty"
		for noun in nouns:
			output = interactive.get_conceptnet_sequence(noun, self.comet_model, self.sampler, self.data_loader, self.text_encoder, relation)
			output = output[list(output.keys())[0]]['beams'][0]
			properties.append(output)
		new_nouns,selected_props = [],[]
		for noun in nouns:
			n = noun[0]
			rand_prop = np.random.choice(properties,1)
			while(rand_prop in selected_props):
				rand_prop = np.random.choice(properties,1)
			selected_props.append(rand_prop)
			new_word = " ".join([rand_prop,n])
			new_nouns.append((new_word,noun[1],noun[2],noun[3]))
		for n in new_nouns:
			modified_plots[n[1]][n[2]][n[3]] = n[0]
		out1 = []
		for mp in modified_plots:
			out2 = []
			for mp2 in mp:
				out2.append(" ".join(mp2))
			out1.append("\t".join(out2))
		final_plot = "\t</s>\t".join(out1)
		if final_plot.startswith('\t'):
			final_plot = '\t'.join(final_plot.split('\t')[1:])
		return final_plot


	#MODIFICATIONS TO SARIK's methods
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
		sents_plots = plots.split('</s>')
		new_sents_plots = sents_plots
		list_plts_verb,list_plts_verb_tense = [],[]    
	
		sents = text.split('</s>')
		sents_pos = {i:nltk.pos_tag(nltk.word_tokenize(sent)) for i,sent in enumerate(sents)}
		for ind, sent_plots in enumerate(sents_plots):
			sent_plots = sent_plots.strip().split('\t')
			sentence = sents_pos[ind]
			sentence_tokens,sentence_tags = [s[0] for s in sentence],[s[1] for s in sentence]
			for plt in sent_plots:
				plt = plt.strip()
				pos_plt = None
				if ' ' not in plt and sents_pos:
					try:	pos_plt= sentence_tags[sentence_tokens.index(plt)]
					except ValueError:	pos_plt=None
				else: #get POS tags for the plot
					sentence = nltk.pos_tag(nltk.word_tokenize(plt))
					sentence_tags = [s[1] for s in sentence]
					has_vb = [pos for pos in sentence_tags if('VB' in pos) ]
					if(len(has_vb)>0):
						pos_plt = has_vb[0]
				'''
				pos_plt,adverb_token = None,None
				if ' ' not in plt and sents_pos: #get POS tags from the story

					try:	pos_plt= sentence_tags[sentence_tokens.index(plt)]
					except ValueError:	pos_plt=None
					if(pos_plt and pos_plt!="RB"):
						pos_plt=None 

				else: #get POS tags for the plot
					plot_pos_pairs = nltk.pos_tag(nltk.word_tokenize(plt))
					plot_tokens,plot_tags = [s[0] for s in plot_pos_pairs],[s[1] for s in plot_pos_pairs]
					try:	pos_plt= plot_tokens[plot_tags.index("RB")]
					except ValueError:	pos_plt=None 
				if(pos_plt and plt in location_adverbs):
					has_adverb.add(plt)
				'''		
				lemmatizer = WordNetLemmatizer()		
				if pos_plt \
					and pos_plt in ['VBD', 'VB',  'VBZ', 'VBP']\
					and (plt in list(self.plt_antonyms.keys()) or lemmatizer.lemmatize(plt,pos="v") in list(self.plt_antonyms.keys()))\
					and plt not in list_plts_verb :
						list_plts_verb.append(plt)
						list_plts_verb_tense.append(pos_plt)
		if list_plts_verb == []:
			return plots

		#candidates = []
		#if len(has_adverb) == 0: 
		#	candidates = []
		#	rand_plot = sents_plots[np.random.choice(len(sents_plots))]
		#	candidates.extend([np.random.choice(rand_plot.strip().split('\t'))])
		#else:
		#	candidates = has_adverb

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
					subplot = p.split(" ")
					choice_to_del = np.random.choice(len(subplot),size=1,replace=False)[0]
					del subplot[choice_to_del]
					subplot = " ".join(subplot)
				plots_.append(subplot)
			plots = '\t'.join(plots_)
			new_sents_plots.append(plots)
			checkpoint = plot_index+1
		new_sents_plots.extend(sents_plots[checkpoint:])
		new_plots = '#'.join(new_sents_plots)	
		if new_plots.startswith('\t'):
			new_plots = '\t'.join(new_plots.split('\t')[1:])
		return new_plots
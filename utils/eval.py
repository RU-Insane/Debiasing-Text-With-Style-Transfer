import pandas as pd
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

#BERT model
CLS_MODEL_PATH = "C:/Users/athar/OneDrive/Desktop/Rutgers/NLP/model"
model_sc = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_PATH)

# Content Preservation Score models
model_st = SentenceTransformer('bert-base-nli-mean-tokens')
cls_explainer = SequenceClassificationExplainer(model_sc, tokenizer)


def evaluate_sti(source_proba,pred_proba,target_proba,min_sti_train = 0.67):
	#calculate EMD
	#get minimum sti from train data
	#min_sti_train = (evaluation_df['EMD Source-Predicted'] + (1/(1+evaluation_df['EMD Target-Predicted']))).min() 
	# #calculate every time training is done

	#emd_source_tgt = wasserstein_distance(source_proba,target_proba)
	emd_source_pred = wasserstein_distance(source_proba,pred_proba)
	emd_target_pred = wasserstein_distance(pred_proba,target_proba)

	sti = emd_source_pred + 1/(1+emd_target_pred)
	sti = sti - min_sti_train
	return sti


def get_attributions(text):
	word_attributions = cls_explainer(text)
	# Create a DataFrame
	df_attrb = pd.DataFrame(word_attributions, columns=['token', 'score'])
	df_attrb["abs_norm"] = df_attrb['score'].abs()/df_attrb["score"].abs().sum()
	df_attrb = df_attrb.sort_values(by='abs_norm',ascending=False)
	df_attrb['abs_norm'] = df_attrb['abs_norm'].cumsum()
	df_attrb["cumulative"] = df_attrb["abs_norm"].cumsum()

	return df_attrb


def remove_style_words(text):
	atrrb_df = get_attributions(text)
	#remove the top scoring word
	#if length is less than 3, do not remove any word
	if len(atrrb_df['token']) <= 3:
		None
	else: 
		#if length more than 3 remove top score word
		
		#get the max score
		max_score = max(atrrb_df['score'])
        #list the words with subjectivity for future use
		subj_wrds = atrrb_df['token'][(atrrb_df['score']>=max_score)]
		non_subj_words = atrrb_df['token'][(atrrb_df['score']<max_score)]
		masked_text = non_subj_words.sort_index()[1:-1].str.cat(sep=' ')
		
	return masked_text,subj_wrds


def get_cps_score(source_text, target_text):

	#generate embeddings (768 dimensional)
	source_embeddings = model_st.encode(source_text)
	predicted_embeddings = model_st.encode(target_text)
	content_scores = 1 - cosine(source_embeddings, predicted_embeddings)	
	
	return content_scores


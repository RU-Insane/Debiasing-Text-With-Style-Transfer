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
CLS_MODEL_PATH = "../../../model/bert-finetuned"
model_sc = AutoModelForSequenceClassification.from_pretrained(CLS_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_PATH)

# Content Preservation Score models
model_st = SentenceTransformer('bert-base-nli-mean-tokens')
cls_explainer = SequenceClassificationExplainer(model_sc, tokenizer)


def evaluate_sti(source_proba, pred_proba, target_proba, min_sti_train=0.67):
	"""
	Calculates the Style Transfer Intensity (STI) based on the Earth Mover's Distance (EMD) between source, predicted, and target probabilities.

	Parameters:
	- source_proba (array-like): The probability distribution of the source style.
	- pred_proba (array-like): The probability distribution of the predicted style.
	- target_proba (array-like): The probability distribution of the target style.
	- min_sti_train (float, optional): The minimum STI value from the training data. Default is 0.67.

	Returns:
	- sti (float): The calculated Style Transfer Intensity (STI).
	"""
	# Calculate EMD
	emd_source_pred = wasserstein_distance(source_proba, pred_proba)
	emd_target_pred = wasserstein_distance(pred_proba, target_proba)

	# Calculate STI
	sti = emd_source_pred + 1 / (1 + emd_target_pred)
	sti = sti - min_sti_train

	return sti


def get_attributions(text):
	"""
	Calculates the word attributions for the given text.

	Parameters:
	text (str): The input text for which word attributions need to be calculated.

	Returns:
	pandas.DataFrame: A DataFrame containing the word attributions, sorted by the absolute normalized score.
	The DataFrame includes columns for the token, score, absolute normalized score, cumulative sum of absolute normalized score.
	"""
	word_attributions = cls_explainer(text)
	# Create a DataFrame
	df_attrb = pd.DataFrame(word_attributions, columns=['token', 'score'])
	df_attrb["abs_norm"] = df_attrb['score'].abs()/df_attrb["score"].abs().sum()
	df_attrb = df_attrb.sort_values(by='abs_norm',ascending=False)
	df_attrb['abs_norm'] = df_attrb['abs_norm'].cumsum()
	df_attrb["cumulative"] = df_attrb["abs_norm"].cumsum()

	return df_attrb


def remove_style_words(text):
    """
    This function removes the word with the highest attribution score from the text, 
    provided the text has more than three words. The score is calculated 
    using the `get_attributions` function.

    Parameters:
    text (str): The input text from which the style words need to be removed.

    Returns:
    masked_text (str): The text after removing the word with the highest score.
    subj_wrds (Series): A pandas Series containing the words with the highest score.

    Note:
    If the length of the text is less than or equal to 3, no words are removed.
    """
    atrrb_df = get_attributions(text)
    if len(atrrb_df['token']) <= 3:
        return None
    else: 
        max_score = max(atrrb_df['score'])
        subj_wrds = atrrb_df['token'][(atrrb_df['score']>=max_score)]
        non_subj_words = atrrb_df['token'][(atrrb_df['score']<max_score)]
        masked_text = non_subj_words.sort_index()[1:-1].str.cat(sep=' ')
        
    return masked_text, subj_wrds


def get_cps_score(source_text, target_text):
	"""
	Calculates the content preservation score between the source text and the target text.

	Parameters:
	source_text (str): The original text.
	target_text (str): The modified text.

	Returns:
	float: The content preservation score between the source text and the target text.
	"""
	#generate embeddings (768 dimensional)
	source_embeddings = model_st.encode(source_text)
	predicted_embeddings = model_st.encode(target_text)
	content_scores = 1 - cosine(source_embeddings, predicted_embeddings)    
	
	return content_scores

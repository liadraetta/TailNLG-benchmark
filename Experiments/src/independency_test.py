import json,evaluate
from scipy.stats import ttest_ind,wilcoxon
from typing import List,Dict,Tuple
from evaluate import load
import random,numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



class IndependentExtrinsic:
    def __init__(self,metric:str='bertscore'):
        self.metric_name = metric
        self.metric = load(metric)
        

    def _get_preds_and_gold(self,path:str,lang:str=None)-> Tuple[List[str],List[str]]: 
        jsn  = json.load(open(path,'r'))

        if lang:
            if lang=='it':
                jsn = [x for x in jsn if x['language']=='it' or x['language']=='it-PE']
            else:
                jsn = [x for x in jsn if x['language']==lang]
        
        predictions = [x['prediction_pe'] for x in jsn]
        gold = [x['actual'] for x in jsn]
    
        return predictions, gold
    
    def _compute_score(self,predictions:str,references:str,random_seed:int,sample:int=1000)-> List[float]:
        random.seed(random_seed)
        random.shuffle(predictions)
        predictions = predictions[:sample]
        random.shuffle(references) 
        references = references[:sample]

        if self.metric_name == 'bertscore':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref],model_type='distilbert-base-multilingual-cased')['f1'][0] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'meteor':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['meteor'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'bleu':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['bleu'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'rouge':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['rouge1'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'chrf':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['score'] for pred, ref in zip(predictions, references)]
            
        
        return pred_scores
    
    def test_independency(self,path_a:str,path_b:str,random_seeds:List[int]=[42,16,883,4,2005,777,1999,33,2666,24],sample:int=1000,lang:str=None)-> Tuple[float,float]:
        
        preds_a, gold_a = self._get_preds_and_gold(path_a,lang)
        preds_b, gold_b = self._get_preds_and_gold(path_b,lang)
        
        ind_dict = {'t_stat':[], 'p_value':[],'runs':[]}
        for random_seed in tqdm(random_seeds):

            scores_a = self._compute_score(preds_a, gold_a, random_seed, sample)
            scores_b = self._compute_score(preds_b, gold_b, random_seed, sample)
            

            t_stat, p_value = ttest_ind(scores_a, scores_b)
            ind_dict['t_stat'].append(t_stat)
            ind_dict['p_value'].append(p_value)
            
            if p_value<0.05:
                ind_dict['runs'].append(1)
            else: 
                ind_dict['runs'].append(0)

        
        ind_dict['t_stat'] = np.mean(ind_dict['t_stat'])
        ind_dict['p_value'] = np.mean(ind_dict['p_value'])

        return ind_dict
#bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")


class IndependentIntrinsic:
    def __init__(self,metric:str='bertscore',path='experiments/datasets/entities.csv'):
        self.metric_name = metric
        self.metric = load(metric)
        df = pd.read_csv(path)
        self.entities = {row.eid:row.type for _,row in df.iterrows()}
        

    def _get_preds_and_gold(self,path:str,lang:str=None)-> Tuple[List[str],List[str]]: 
        jsn  = json.load(open(path,'r'))

        if lang:
            if lang=='it':
                jsn = [x for x in jsn if x['language']=='it' or x['language']=='it-PE']
            else:
                jsn = [x for x in jsn if x['language']==lang]
        
        tail_predictions = [x['prediction_pe'] for x in jsn if self.entities[x['eid']]=='long_tail']
        tail_gold = [x['actual'] for x in jsn if self.entities[x['eid']]=='long_tail']
        head_predictions = [x['prediction_pe'] for x in jsn if self.entities[x['eid']]=='top_head']
        head_gold = [x['actual'] for x in jsn if self.entities[x['eid']]=='top_head']
    
        return tail_predictions,tail_gold,head_predictions,head_gold
    
    def _compute_score(self,predictions:str,references:str,random_seed:int,sample:int=1000)-> List[float]:
        random.seed(random_seed)

        predictions = predictions.copy()
        random.shuffle(predictions)
        predictions = predictions[:sample]
        random.shuffle(references) 
        references = references.copy()
        references = references[:sample]

        if self.metric_name == 'bertscore':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref],model_type='distilbert-base-multilingual-cased')['f1'][0] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'meteor':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['meteor'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'bleu':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['bleu'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'rouge':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['rougeL'] for pred, ref in zip(predictions, references)]
        elif self.metric_name == 'chrf':
            pred_scores = [self.metric.compute(predictions=[pred], references=[ref])['score'] for pred, ref in zip(predictions, references)]
            
        
        return pred_scores
    
    def _calculate_perplexity(
        self,
        texts: List[str],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "mps"
        ) -> List[float]:
        """
        Calculate perplexity for a batch of texts.
        
        Args:
            texts: List of texts to evaluate
            model: Loaded model
            tokenizer: Loaded tokenizer
            batch_size: Batch size for processing
            device: Device to use ('cuda' or 'cpu')
        
        Returns:
            List of perplexity scores
        """
        batch_size = 1
        model.eval()
        perplexities = []
        tokenizer.pad_token = tokenizer.eos_token 
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Calculate loss for each item in batch
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

        return perplexities
    
    def test_independency(self,path:str,
                          random_seeds:List[int]=[42,16,883,4,2005,777,1999,33,2666,24],
                          sample:int=1000,
                          lang:str=None,
                          perplexity=False,
                          model:str=None,
                          tokenizer=None)-> Tuple[float,float]:
        
        preds_a, gold_a,preds_b,gold_b = self._get_preds_and_gold(path,lang)
        ind_dict = {'t_stat':[], 'p_value':[],'runs':[]}
        if perplexity:
            my_model = AutoModelForCausalLM.from_pretrained(model).to("mps")
            my_tokenizer = AutoTokenizer.from_pretrained(model)
            for random_seed in random_seeds:
                scores_a = self._calculate_perplexity(preds_a, my_model, my_tokenizer, device="mps")
                scores_b = self._calculate_perplexity(preds_b, my_model, my_tokenizer, device="mps")
                t_stat, p_value = ttest_ind(scores_a, scores_b)
                ind_dict['t_stat'].append(t_stat)
                ind_dict['p_value'].append(p_value)
                
                if p_value<0.05:
                    ind_dict['runs'].append(1)
                else: 
                    ind_dict['runs'].append(0)
            ind_dict['t_stat'] = np.mean(ind_dict['t_stat'])
            ind_dict['p_value'] = np.mean(ind_dict['p_value'])
                

        else:
            for random_seed in tqdm(random_seeds):

                scores_a = self._compute_score(preds_a, gold_a, random_seed, sample)
                scores_b = self._compute_score(preds_b, gold_b, random_seed, sample)
                

                t_stat, p_value = ttest_ind(scores_a, scores_b)
                ind_dict['t_stat'].append(t_stat)
                ind_dict['p_value'].append(p_value)
                
                if p_value<0.05:
                    ind_dict['runs'].append(1)
                else: 
                    ind_dict['runs'].append(0)

            
            ind_dict['t_stat'] = np.mean(ind_dict['t_stat'])
            ind_dict['p_value'] = np.mean(ind_dict['p_value'])

        return ind_dict
#bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")




class NewMetric:
    def __init__(self,path='experiments/datasets/entities.csv'):
        df = pd.read_csv(path)
        self.entities = {row.eid:row.type for _,row in df.iterrows()}
    
    def _get_scores_tailnlg(self,path:str,metric:str,sample:str,lang:str=None)-> Tuple[List[str],List[str]]: 
        jsn  = json.load(open(path,'r'))

        if lang:
            if lang=='it':
                jsn = [x for x in jsn if x['language']=='it' or x['language']=='it-PE']
            else:
                jsn = [x for x in jsn if x['language']==lang]
        
        tail = [x[metric] for x in jsn if self.entities[x['eid']]=='long_tail']
        tail = tail.copy()
        random.shuffle(tail)
        tail = tail[:sample]
        head = [x[metric] for x in jsn if self.entities[x['eid']]=='top_head']
        head = head.copy()
        random.shuffle(head)
        head = head[:sample]


    
        return tail,head
    
    def _get_scores_webnlg(self,path:str,metric:str,sample:str,lang:str=None)-> Tuple[List[str],List[str]]: 
        jsn  = json.load(open(path,'r'))

        if lang:
            if lang=='it':
                jsn = [x for x in jsn if x['language']=='it' or x['language']=='it-PE']
            else:
                jsn = [x for x in jsn if x['language']==lang]
        
        webnlg = [x[metric] for x in jsn]
        webnlg = webnlg.copy()
        random.shuffle(webnlg)
        webnlg = webnlg[:sample]
        
        return webnlg
    
    def test_independency_intrinsic(self,path:str,
                          metric:str,
                          random_seeds:List[int]=[42,16,883,4,2005,777,1999,33,2666,24],
                          sample:int=500,
                          lang:str=None,
                          )-> Tuple[float,float]:
        ind_dict = {'statistics':[], 'p_value':[],'direction':[],'runs':[]}
        for random_seed in tqdm(random_seeds):
            preds_a,preds_b = self._get_scores_tailnlg(path,metric,sample,lang)
            statistics, p_value = wilcoxon(preds_a, preds_b)
            ind_dict['statistics'].append(statistics)
            ind_dict['p_value'].append(p_value)
            ind_dict['direction'] = np.median(np.array(preds_a)-np.array(preds_b))
            if p_value<0.05:
                ind_dict['runs'].append(1)
            else: 
                ind_dict['runs'].append(0)

            
        ind_dict['statistics'] = np.mean(ind_dict['statistics'])
        ind_dict['p_value'] = np.mean(ind_dict['p_value'])
        ind_dict['direction'] = np.mean(ind_dict['direction'])

        return ind_dict
    
    def test_independency_extrinsic(self,path_a:str,
                            path_b:str,
                          metric:str,
                          random_seeds:List[int]=[42,16,883,4,2005,777,1999,33,2666,24],
                          sample:int=500,
                          lang:str=None,
                          )-> Tuple[float,float]:
        
        ind_dict = {'statistics':[], 'p_value':[],'direction':[],'runs':[]}
        for random_seed in tqdm(random_seeds):
            preds_a,preds_b = self._get_scores_tailnlg(path_a,metric,sample,lang)
            webnlg = self._get_scores_webnlg(path_b,metric,sample,lang)
            
            statistics, p_value = wilcoxon(preds_a, webnlg)

            ind_dict['direction'] = np.median(np.array(preds_a)-np.array(webnlg))
            ind_dict['statistics'].append(statistics)
            ind_dict['p_value'].append(p_value)
            if p_value<0.05:
                    ind_dict['runs'].append(1)
            else: 
                ind_dict['runs'].append(0)

            
        ind_dict['statistics'] = np.mean(ind_dict['statistics'])
        ind_dict['p_value'] = np.mean(ind_dict['p_value'])
        ind_dict['direction'] = np.mean(ind_dict['direction'])

        return ind_dict
    

        


    

    

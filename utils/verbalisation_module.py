from utils.finetune import Graph2TextModule
from typing import Dict, List, Tuple, Union, Optional
import torch
import re
import pandas as pd
import logging
import ast

logger = logging.getLogger("prove")

# Safety globals for Pytorch Lightning loading in newer versions of Torch
try:
    from pytorch_lightning.utilities.parsing import AttributeDict
    torch.serialization.add_safe_globals([AttributeDict])
except:
    pass

# Aligned with your Physical GPU 2 reservation
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = './verbalisation_model.ckpt_OG/val_avg_bleu=68.1000-step_count=5.ckpt'
MAX_LENGTH = 384
SEED = 42

class VerbModule():
    
    def __init__(self, override_args: Dict[str, str] = None):
        if not override_args:
            override_args = {}
        
        # FIX: Load to CPU first, then move to DEVICE to avoid OOM spikes
        self.g2t_module = Graph2TextModule.load_from_checkpoint(
            CHECKPOINT, 
            map_location='cpu',  # Load weights to system RAM first
            strict=False, 
            weights_only=False, 
            **override_args
        )
        
        # Clear cache before moving large model to GPU
        torch.cuda.empty_cache()
        self.g2t_module.model.to(DEVICE)
        
        self.tokenizer = self.g2t_module.tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.convert_some_japanese_characters = True
        self.unk_char_replace_sliding_window_size = 2
        self.unknowns = []

    def verbalise_claims(self, claims_df: pd.DataFrame) -> List[str]:
        """
        Hardened verbalization logic:
        1. Cleans Wikidata JSON into strings.
        2. Always returns exactly len(claims_df) sentences to ensure 1-to-1 alignment.
        """
        if claims_df.empty:
            return []
            
        # FIX: Ensure this name matches the return variable at the bottom
        verbalised_results = [] 
        logger.info(f"Verbalizing {len(claims_df)} claims on {DEVICE}...")

        self.g2t_module.model.to(DEVICE)
        self.g2t_module.model.eval()

        for _, row in claims_df.iterrows():
            subj = str(row['entity_label'])
            prop = str(row['property_id'])
            obj = "unknown"
            
            try:
                raw_val = row['datavalue']
                if isinstance(raw_val, str) and '{' in raw_val:
                    try:
                        import ast
                        raw_val = ast.literal_eval(raw_val)
                    except: pass
                
                if isinstance(raw_val, dict):
                    v_content = raw_val.get('value', raw_val)
                    if isinstance(v_content, dict):
                        obj = v_content.get('id', v_content.get('amount', v_content.get('text', str(v_content))))
                    else:
                        obj = str(v_content)
                else:
                    obj = str(raw_val)

                input_text = f"translate Graph to English: <H> {subj} <R> {prop} <T> {obj}"
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
                
                with torch.no_grad():
                    outputs = self.g2t_module.model.generate(
                        inputs["input_ids"], 
                        max_length=128,
                        num_beams=self.g2t_module.eval_beams,
                        early_stopping=True
                    )
                
                sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                self.add_label_to_unk_replacer(subj)
                self.add_label_to_unk_replacer(obj)
                final_sentence = self.replace_unks_on_sentence(sentence)
                
                # Appending to the correct variable name
                verbalised_results.append(final_sentence)
                
            except Exception as e:
                fallback = f"{subj} has property {prop} as {obj}"
                verbalised_results.append(fallback)
                logger.error(f"Generation failed: {str(e)}")
        
        # Final Alignment Check
        if len(verbalised_results) != len(claims_df):
            while len(verbalised_results) < len(claims_df):
                verbalised_results.append("Verbalization failed.")

        return verbalised_results

    def __generate_verbalisations_from_inputs(self, inputs: Union[str, List[str]]):
        try:
            # Modern tokenizer call to avoid prepare_seq2seq_batch warnings
            inputs_encoding = self.tokenizer(
                inputs, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors='pt'
            ).to(DEVICE)
            
            self.g2t_module.model.eval()
            with torch.no_grad():
                self.g2t_module.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
                
                gen_output = self.g2t_module.model.generate(
                    input_ids=inputs_encoding['input_ids'],
                    attention_mask=inputs_encoding['attention_mask'],
                    max_length=self.g2t_module.eval_max_length,
                    num_beams=self.g2t_module.eval_beams,
                    length_penalty=1.0,
                    early_stopping=True,
                )
        except Exception:
            print(f"Error generating from: {inputs}")
            raise
        return gen_output
    
    def __decode_ids_to_string_custom(
        self, token_ids: List[int], skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True
    ) -> str:
        filtered_tokens = self.tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and \
                token != self.tokenizer.unk_token and \
                token in self.tokenizer.all_special_tokens:
                continue
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.tokenizer.convert_tokens_to_string(current_sub_text))
        text = " ".join(sub_texts)

        if clean_up_tokenization_spaces:
            clean_text = self.tokenizer.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def __decode_sentences(self, encoded_sentences: Union[torch.Tensor, List[List[int]]]):
        return [self.__decode_ids_to_string_custom(i, skip_special_tokens=True) for i in encoded_sentences]
        
    def verbalise_sentence(self, inputs: Union[str, List[str]]):
        was_str = isinstance(inputs, str)
        if was_str:
            inputs = [inputs]
        
        gen_output = self.__generate_verbalisations_from_inputs(inputs)
        decoded_sentences = self.__decode_sentences(gen_output)

        if was_str and len(decoded_sentences) == 1:
            return decoded_sentences[0]
        else:
            return decoded_sentences

    def verbalise_triples(self, input_triples: Union[Dict[str, str], List[Dict[str, str]], List[List[Dict[str, str]]]]):
        if isinstance(input_triples, dict):
            input_triples = [input_triples]

        verbalisation_inputs = []
        for triple in input_triples:
            if isinstance(triple, dict):
                verbalisation_inputs.append(
                    f'translate Graph to English: <H> {triple["subject"]} <R> {triple["predicate"]} <T> {triple["object"]}'
                )
            elif isinstance(triple, list):
                input_sentence = ['translate Graph to English:']
                for subtriple in triple:
                    input_sentence.append(f'<H> {subtriple["subject"]} <R> {subtriple["predicate"]} <T> {subtriple["object"]}')
                verbalisation_inputs.append(' '.join(input_sentence))

        return self.verbalise_sentence(verbalisation_inputs)
        
    def verbalise(self, input: Union[str, List, Dict]):
        try:
            if (isinstance(input, str)) or (isinstance(input, list) and isinstance(input[0], str)):
                return self.verbalise_sentence(input)
            return self.verbalise_triples(input)
        except Exception:
            print(f'ERROR VERBALISING {input}')
            raise
                
    def add_label_to_unk_replacer(self, label: str):
        N = self.unk_char_replace_sliding_window_size
        self.unknowns.append({})
        
        if self.convert_some_japanese_characters:
            label = label.replace('（','(').replace('）',')').replace('〈','<').replace('／','/').replace('〉','>')        
        
        label_encoded = self.tokenizer.encode(label)
        label_tokens = [t for t in self.tokenizer.convert_ids_to_tokens(label_encoded) if t not in [self.tokenizer.eos_token, self.tokenizer.pad_token]]
        label_token_to_string = self.tokenizer.convert_tokens_to_string(label_tokens)
        unk_token_to_string = self.tokenizer.convert_tokens_to_string([self.tokenizer.unk_token])
        
        match_unks_in_label = re.findall('(?:(?: )*<unk>(?: )*)+', label_token_to_string)
        if len(match_unks_in_label) > 0:
            if (match_unks_in_label[0]) == label_token_to_string:
                self.unknowns[-1][label_token_to_string.strip()] = label
            else:
                for idx, token in enumerate(label_tokens):
                    idx_before, idx_ahead = max(0,idx-N), min(len(label_tokens), idx+N+1)
                    if token == self.tokenizer.unk_token:
                        span = self.tokenizer.convert_tokens_to_string(label_tokens[idx_before:idx_ahead])        
                        to_replace = re.escape(span).replace(re.escape(unk_token_to_string), '.+?')
                        try:
                            replaced_span = re.search(to_replace, label)[0]
                            self.unknowns[-1][span.strip()] = replaced_span
                        except: pass

    def replace_unks_on_sentence(self, sentence: str, loop_n : int = 3, empty_after : bool = False):
        while '<unk>' in sentence and loop_n > 0:
            loop_n -= 1
            for unknowns in self.unknowns:
                for k,v in unknowns.items():
                    if k == '<unk>' and loop_n > 0: continue
                    sentence = sentence.replace(k.strip(), v.strip(), 1)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        sentence = re.sub(r'\s([?.!",](?:\s|$))', r'\1', sentence)
        if empty_after: self.unknowns = []
        return sentence

if __name__ == '__main__':
    verb_module = VerbModule()
    # Test sample triple
    sample = {'subject': 'World Trade Center', 'predicate': 'height', 'object': '200 meter'}
    print(verb_module.verbalise(sample))
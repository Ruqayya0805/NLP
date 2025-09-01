#!/usr/bin/env python3

import time
import spacy
import stanza
from typing import List, Dict, Any
import pandas as pd

class ParsingComparator:
    def __init__(self):
        print("Loading spaCy model...")
        self.spacy_nlp = spacy.load("en_core_web_sm")
        
        print("Loading Stanford NLP model...")
        self.stanford_nlp = stanza.Pipeline('en', verbose=False)
        
        print("Models loaded successfully!\n")
    
    def parse_with_spacy(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        doc = self.spacy_nlp(text)
        parse_time = time.time() - start_time
        
        dependencies = []
        for token in doc:
            dependencies.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
                'children': [child.text for child in token.children]
            })
        
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                   for ent in doc.ents]
        
        return {
            'library': 'spaCy',
            'text': text,
            'dependencies': dependencies,
            'entities': entities,
            'parse_time': parse_time,
            'token_count': len(doc),
            'sentence_count': len(list(doc.sents))
        }
    
    def parse_with_stanford(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        doc = self.stanford_nlp(text)
        parse_time = time.time() - start_time
        
        dependencies = []
        entities = []
        
        for sentence in doc.sentences:
            for word in sentence.words:
                dependencies.append({
                    'text': word.text,
                    'lemma': word.lemma,
                    'pos': word.upos,
                    'dep': word.deprel,
                    'head': sentence.words[word.head-1].text if word.head > 0 else 'ROOT',
                    'head_pos': sentence.words[word.head-1].upos if word.head > 0 else 'ROOT'
                })
            
            for entity in sentence.ents:
                entities.append((entity.text, entity.type, entity.start_char, entity.end_char))
        
        return {
            'library': 'Stanford NLP',
            'text': text,
            'dependencies': dependencies,
            'entities': entities,
            'parse_time': parse_time,
            'token_count': sum(len(sent.words) for sent in doc.sentences),
            'sentence_count': len(doc.sentences)
        }
    
    def display_dependencies(self, result: Dict[str, Any]) -> None:
        print(f"\n{result['library']} - Dependency Analysis:")
        print(f"Text: {result['text']}")
        print(f"Parse Time: {result['parse_time']:.4f}s")
        print(f"Tokens: {result['token_count']}, Sentences: {result['sentence_count']}")
        print("\nDependency Relations:")
        print(f"{'Token':<12} {'POS':<8} {'Dep Rel':<12} {'Head':<12}")
        print("-" * 50)
        
        for dep in result['dependencies']:
            print(f"{dep['text']:<12} {dep['pos']:<8} {dep['dep']:<12} {dep['head']:<12}")
        
        if result['entities']:
            print(f"\nEntities: {result['entities']}")
    
    def compare_parsing_results(self, text: str) -> Dict[str, Any]:
        print(f"{'='*60}")
        print(f"COMPARING PARSING RESULTS")
        print(f"{'='*60}")
        
        spacy_result = self.parse_with_spacy(text)
        stanford_result = self.parse_with_stanford(text)
        
        self.display_dependencies(spacy_result)
        self.display_dependencies(stanford_result)
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON:")
        print(f"spaCy processing time: {spacy_result['parse_time']:.4f}s")
        print(f"Stanford NLP processing time: {stanford_result['parse_time']:.4f}s")
        print(f"Speed ratio: {stanford_result['parse_time']/spacy_result['parse_time']:.1f}x slower")
        
        return {
            'spacy': spacy_result,
            'stanford': stanford_result,
            'speed_ratio': stanford_result['parse_time']/spacy_result['parse_time']
        }
    
    def benchmark_performance(self, test_sentences: List[str], iterations: int = 3) -> pd.DataFrame:
        results = []
        
        print(f"Running benchmark with {len(test_sentences)} sentences, {iterations} iterations each...")
        
        for i, sentence in enumerate(test_sentences):
            print(f"Testing sentence {i+1}/{len(test_sentences)}")
            
            spacy_times = []
            for _ in range(iterations):
                result = self.parse_with_spacy(sentence)
                spacy_times.append(result['parse_time'])
            
            stanford_times = []
            for _ in range(iterations):
                result = self.parse_with_stanford(sentence)
                stanford_times.append(result['parse_time'])
            
            results.append({
                'sentence': sentence[:50] + "..." if len(sentence) > 50 else sentence,
                'length': len(sentence.split()),
                'spacy_avg_time': sum(spacy_times) / len(spacy_times),
                'stanford_avg_time': sum(stanford_times) / len(stanford_times),
                'speed_difference': (sum(stanford_times) / len(stanford_times)) / (sum(spacy_times) / len(spacy_times))
            })
        
        return pd.DataFrame(results)

def main():
    comparator = ParsingComparator()
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Apple is looking at buying U.K. startup for $1 billion.",
        "The students who studied hard passed the difficult exam.",
        "After the rain stopped, we went for a walk in the park.",
        "Machine learning algorithms can process natural language effectively.",
        "The company's quarterly earnings exceeded analyst expectations significantly.",
        "Despite the challenging weather conditions, the marathon runners persevered.",
        "Artificial intelligence has revolutionized many industries in recent years."
    ]
    
    print("DETAILED PARSING COMPARISON")
    print("=" * 80)
    
    for i, sentence in enumerate(test_sentences[:3]):
        print(f"\nTEST CASE {i+1}:")
        comparator.compare_parsing_results(sentence)
    
    print(f"\n{'='*80}")
    print("PERFORMANCE BENCHMARK")
    print(f"{'='*80}")
    
    benchmark_df = comparator.benchmark_performance(test_sentences)
    
    print("\nBenchmark Results:")
    print(benchmark_df.to_string(index=False))
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"Average spaCy time: {benchmark_df['spacy_avg_time'].mean():.4f}s")
    print(f"Average Stanford time: {benchmark_df['stanford_avg_time'].mean():.4f}s")
    print(f"Overall speed difference: {benchmark_df['speed_difference'].mean():.1f}x")
    print(f"spaCy std deviation: {benchmark_df['spacy_avg_time'].std():.4f}s")
    print(f"Stanford std deviation: {benchmark_df['stanford_avg_time'].std():.4f}s")

def demonstrate_advanced_features():
    print("\nADVANCED FEATURES DEMONSTRATION")
    print("=" * 50)
    
    nlp_spacy = spacy.load("en_core_web_sm")
    nlp_stanford = stanza.Pipeline('en', verbose=False)
    
    test_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    
    print("spaCy Advanced Features:")
    doc = nlp_spacy(test_text)
    
    print("Named Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print("Noun Chunks:", [chunk.text for chunk in doc.noun_chunks])
    
    print("\nStanford NLP Advanced Features:")
    doc_stanford = nlp_stanford(test_text)
    
    for sentence in doc_stanford.sentences:
        print("Dependency Relations:", [(word.text, word.deprel, word.head) for word in sentence.words])
        print("Entities:", [(ent.text, ent.type) for ent in sentence.ents])

if __name__ == "__main__":
    try:
        main()
        demonstrate_advanced_features()
    except Exception as e:
        print(f"Error running comparison: {e}")
        print("Make sure you have the required libraries and models installed.")
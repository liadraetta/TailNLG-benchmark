import os
import xml.etree.ElementTree as ET
from typing import Optional, List

from .config import DATASETS_DIR, EXCLUDED_TRIPLES


class WebNLGLoader:
    DATASETS = {
        "en": {
            'dataset_split': ['train', 'dev', 'test'],
            'triple_numbers': ['1', '2', '3', '4', '5', '6', '7'],
            'lang_field': 'en',
            'path': DATASETS_DIR / "en_webnlg",
        },
        "es": {
            'dataset_split': ['train', 'dev', 'test'],
            'triple_numbers': ['1', '2', '3', '4', '5', '6', '7'],
            'lang_field': 'es',
            'path': DATASETS_DIR / "es_webnlg",
        },
        "it": {
            'dataset_split': ['train', 'dev', 'test'],
            'triple_numbers': ['1', '2', '3', '4', '5', '6', '7'],
            'lang_field': 'it-PE',
            'path': DATASETS_DIR / "it_webnlg",
        },
    }

    LONGTAIL_WEBNLG_DATASET = {
        "path": DATASETS_DIR / "longTail_webnlg",
    }

    LOAD_METHODS = ["merged", "separated"]

    @staticmethod
    def load_longtail_webnlg_dataset(languages: Optional[List[str]] = None):
        if not languages:
            languages = list(WebNLGLoader.DATASETS.keys())

        for lang in languages:
            if lang not in WebNLGLoader.DATASETS:
                raise ValueError(f"Unsupported language: {lang}. Supported languages are: {list(WebNLGLoader.DATASETS.keys())}")

        datasets = WebNLGLoader.get_longtail_triples(languages)
        return datasets

    @staticmethod
    def load_webnlg_dataset(languages: Optional[List[str]] = None, method: str = "merged"):
        # Validate inputs
        if method not in WebNLGLoader.LOAD_METHODS:
            raise ValueError(f"Invalid load method: {method}. Supported methods are: {WebNLGLoader.LOAD_METHODS}")
        
        if not languages:
            languages = list(WebNLGLoader.DATASETS.keys())

        for lang in languages:
            if lang not in WebNLGLoader.DATASETS:
                raise ValueError(f"Unsupported language: {lang}. Supported languages are: {list(WebNLGLoader.DATASETS.keys())}")

        datasets = {}
        for lang in languages:
            datasets[lang] = {}  # Initialize lang dict
            lang_config = WebNLGLoader.DATASETS[lang]
            dataset_splits = lang_config['dataset_split']

            for split in dataset_splits:
                if split in ['train', 'dev']:
                    datasets[lang][split] = WebNLGLoader.get_train_val_triples(lang, split)
                else:
                    datasets[lang][split] = WebNLGLoader.get_test_triples(lang, split)

        if method == "merged":
            merged_dataset = {}
            # Get all splits from first language (they should be the same for all)
            dataset_splits = WebNLGLoader.DATASETS[languages[0]]['dataset_split']
            for split in dataset_splits:
                merged_dataset[split] = []
                for lang in languages:
                    merged_dataset[split].extend(datasets[lang][split])
            return merged_dataset
        else:  # separated
            return datasets
        
    @staticmethod
    def get_train_val_triples(lang, split):
        lang_config = WebNLGLoader.DATASETS[lang]
        triple_numbers = lang_config['triple_numbers']
        lang_field = lang_config['lang_field']
        base_path = lang_config['path']

        data_entries = []

        for triple_num in triple_numbers:
            dir_path = base_path / split / f"{triple_num}triples"
            if not dir_path.exists():
                continue

            for file_name in os.listdir(dir_path):
                file_path = dir_path / file_name
                if os.path.isfile(file_path):
                    tree = ET.parse(file_path)
                    root = tree.getroot()

                    for entry in root.iter('entry'):
                        eid = entry.get('eid')
                        category = entry.get('category')
                        shape = entry.get('shape')
                        shape_type = entry.get('shape_type')
                        size = entry.get('size')

                        data_unit = WebNLGLoader._get_data_unit(entry)

                        # Filter lexs by language
                        if lang_field == 'en':
                            lexs = [lex for lex in entry.iter('lex')]
                        else:
                            lexs = [lex for lex in entry.iter('lex') if lex.get('lang') == lang_field]

                        for lex in lexs:
                            data_entries.append({
                                "num_triples": size,
                                "file_name": file_name,
                                "eid": eid,
                                "category": category,
                                "shape": shape,
                                "shape_type": shape_type,
                                "data_unit": data_unit,
                                "sentence": lex.text,
                                "language": lang_field
                            })

        return data_entries
    
    @staticmethod
    def get_test_triples(lang, split):
        lang_config = WebNLGLoader.DATASETS[lang]
        lang_field = lang_config['lang_field']
        base_path = lang_config['path']

        data_entries = []

        dir_path = base_path / split
        test_file_name = f"rdf-to-text-generation-test-data-with-refs-{lang}.xml"
        file_path = dir_path / test_file_name
        if not file_path.exists():
            return data_entries

        tree = ET.parse(file_path)
        root = tree.getroot()

        for entry in root.iter('entry'):
            eid = entry.get('eid')
            category = entry.get('category')
            shape = entry.get('shape')
            shape_type = entry.get('shape_type')
            size = entry.get('size')

            data_unit = WebNLGLoader._get_data_unit(entry)

            # Filter lexs by language
            if lang_field == 'en':
                lexs = [lex.text for lex in entry.iter('lex')]
            else:
                lexs = [lex.text for lex in entry.iter('lex') if lex.get('lang') == lang_field]


            data_entries.append({
                "num_triples": size,
                "file_name": test_file_name,
                "eid": eid,
                "category": category,
                "shape": shape,
                "shape_type": shape_type,
                "data_unit": data_unit,
                "sentence": lexs,
                "language": lang_field
            })

        return data_entries
    
    @staticmethod
    def get_longtail_triples(languages: Optional[List[str]] = None):
        dir_path = WebNLGLoader.LONGTAIL_WEBNLG_DATASET['path']

        data_entries = []

        file_name = f"longTailWebNLG-v1.0.xml"
        file_path = dir_path / file_name
        if not file_path.exists():
            return data_entries

        tree = ET.parse(file_path)
        root = tree.getroot()

        for entry in root.iter('entry'):
            unique_id = entry.get('unique_id')
            if unique_id in EXCLUDED_TRIPLES:
                continue

            eid = entry.get('eid')
            category = entry.get('category')
            shape = entry.get('shape')
            shape_type = entry.get('shape_type')
            size = entry.get('size')
            type_ = entry.get('type')
            subtype = entry.get('subtype')

            data_unit = WebNLGLoader._get_data_unit(entry)

            # Filter lexs by language
            for lex in entry.iter('lex'):
                if languages is None or lex.get('lang') in languages:
                    data_entries.append({
                        "num_triples": size,
                        "file_name": file_name,
                        "eid": eid,
                        "category": category,
                        "shape": shape,
                        "shape_type": shape_type,
                        "type": type_,
                        "subtype": subtype,
                        "data_unit": data_unit,
                        "sentence": lex.text,
                        "language": lex.get('lang')
                    })

        return data_entries
    
    @staticmethod
    def _get_data_unit(entry):
        mts_list = []
        for modifiedtripleset in entry.iter('modifiedtripleset'):
            triples = []
            for mtriple in modifiedtripleset.iter('mtriple'):
                triple_parts = mtriple.text.split(" | ")
                triples.append((triple_parts[0], triple_parts[1], triple_parts[2]))
            triples.sort(key=lambda x: x[1]) # Sort by predicate

            for triple in triples:
                mts_list.append(
                    {
                        "subject": triple[0],
                        "predicate": triple[1],
                        "object": triple[2]
                    }
                )

        return mts_list
import json
import ast
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pathlib import Path

def parse_triples(triples_str):
    """Parse the triples string into a list of tuples."""
    try:
        triples_list = ast.literal_eval(triples_str)
        return triples_list
    except:
        return []

def prettify_xml(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ", encoding='utf-8').decode('utf-8')

def convert_unified_corpus_to_webnlg_xml(input_json_path, output_xml_path):
    """
    Convert unified_corpus.json to WebNLG XML format.
    
    Args:
        input_json_path: Path to unified_corpus.json
        output_xml_path: Path to save the converted XML
    """
    # Load the unified corpus
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create root element
    root = ET.Element('benchmark')
    entries_elem = ET.SubElement(root, 'entries')
    
    for idx, entry in enumerate(data, start=1):
        # Parse triples for each language
        triples_en = parse_triples(entry.get('triples_en', '[]'))
        triples_es = parse_triples(entry.get('triples_es', '[]'))
        triples_it = parse_triples(entry.get('triples_it', '[]'))
        
        # Get configuration to determine shape_type
        config = entry.get('configuration', '')
        shape_type_map = {
            'conf_1': 'sibling',
            'conf_2': 'chain',
            'conf_3': 'mixed'
        }
        shape_type = shape_type_map.get(config, 'NA')
        
        # Determine type and subtype
        original_type = entry.get('type', '')
        entry_subtype = original_type  # Keep original as subtype
        
        # Determine main type
        if 'long_tail' in original_type:
            entry_type = 'long_tail'
        elif 'top_head' in original_type or 'rare_claims' in original_type:
            entry_type = 'top_head'
        else:
            entry_type = original_type
        
        # Create entry element with attributes (including unique_id, type and subtype)
        entry_elem = ET.SubElement(entries_elem, 'entry', {
            'category': entry.get('category', ''),
            'eid': f"Id{idx}",
            'shape': '',
            'shape_type': shape_type,
            'size': str(entry.get('triples_number', 0)),
            'unique_id': entry.get('unique_id', ''),
            'qid': entry.get('QID', ''),
            'type': entry_type,
            'subtype': entry_subtype
        })
        
        # Add originaltripleset (using English triples)
        originaltripleset = ET.SubElement(entry_elem, 'originaltripleset')
        for triple in triples_en:
            otriple = ET.SubElement(originaltripleset, 'otriple')
            otriple.text = f"{triple[0]} | {triple[1]} | {triple[2]}"
        
        # Add modifiedtripleset (using English triples)
        modifiedtripleset = ET.SubElement(entry_elem, 'modifiedtripleset')
        for triple in triples_en:
            mtriple = ET.SubElement(modifiedtripleset, 'mtriple')
            mtriple.text = f"{triple[0]} | {triple[1]} | {triple[2]}"
        
        # Add spanishtripleset (using Spanish triples)
        spanishtripleset = ET.SubElement(entry_elem, 'spanishtripleset')
        for triple in triples_es:
            striple = ET.SubElement(spanishtripleset, 'striple')
            striple.text = f"{triple[0]} | {triple[1]} | {triple[2]}"
        
        # Add italiantripleset (using Italian triples)
        italiantripleset = ET.SubElement(entry_elem, 'italiantripleset')
        for triple in triples_it:
            itriple = ET.SubElement(italiantripleset, 'itriple')
            itriple.text = f"{triple[0]} | {triple[1]} | {triple[2]}"
        
        # Determine which annotations to use based on is_gold flags
        annotations = {}
        
        # English
        if entry.get('annotation_en_is_gold', False):
            annotations['en'] = entry.get('annotation_en', '')
        else:
            # Use corrected translation if available
            corrected = entry.get('annotation_en_xtower_corrected_translation')
            annotations['en'] = corrected if corrected else entry.get('annotation_en', '')
        
        # Spanish
        if entry.get('annotation_es_is_gold', False):
            annotations['es'] = entry.get('annotation_es', '')
        else:
            # Use corrected translation if available
            corrected = entry.get('annotation_es_xtower_corrected_translation')
            annotations['es'] = corrected if corrected else entry.get('annotation_es', '')
        
        # Italian
        if entry.get('annotation_it_is_gold', False):
            annotations['it'] = entry.get('annotation_it', '')
        else:
            # Use corrected translation if available
            corrected = entry.get('annotation_it_xtower_corrected_translation')
            annotations['it'] = corrected if corrected else entry.get('annotation_it', '')
        
        # Add lex elements for each language
        lex_id = 1
        for lang in ['en', 'es', 'it']:
            if annotations.get(lang):
                # Determine if this is gold standard
                is_gold_key = f'annotation_{lang}_is_gold'
                is_gold = entry.get(is_gold_key, False)
                comment = 'gold' if is_gold else 'silver'
                
                lex = ET.SubElement(entry_elem, 'lex', {
                    'quality': comment,
                    'lid': f"Id{lex_id}",
                    'lang': lang
                })
                lex.text = annotations[lang]
                lex_id += 1
    
    # Write to file with pretty formatting
    xml_string = prettify_xml(root)
    
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_string)
    
    print(f"Conversion completed successfully!")
    print(f"Input: {input_json_path}")
    print(f"Output: {output_xml_path}")
    print(f"Total entries converted: {len(data)}")


def main():
    # Define paths
    input_path = Path("manaul_evaluation/corrected_unified_corpus.json")
    output_path = Path("../experiments/datasets/longTail_webnlg/longTailWebNLG.xml")
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return
    
    # Convert the corpus
    convert_unified_corpus_to_webnlg_xml(input_path, output_path)


if __name__ == "__main__":
    main()

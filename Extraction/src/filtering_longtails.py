import pandas as pd
import os
import json
from collections import Counter

# Selecting only the entities that has a number of claims under the pareto cuttoff
def filtering_long_tail(cutoffs_file, entities_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    cutoffs_df = pd.read_csv(cutoffs_file)
    cutoff_dict = dict(zip(cutoffs_df["category"], cutoffs_df["cutoff_claims"]))
    print(cutoff_dict["Building"])


    for file in os.listdir(entities_folder):
        if file.endswith(".csv"):
            category = os.path.splitext(file)[0]

            if category not in cutoff_dict:
                continue
            df = pd.read_csv(os.path.join(entities_folder, file))
            print(df["claims"])
            cutoff = cutoff_dict[category]
            print(cutoff)
            filtered_df = df[df["claims"] > cutoff]

            output_path = os.path.join(output_folder, f"{category}_longtail.csv")
            filtered_df.to_csv(output_path, index=False)

## From the raw Entities file all the long tail from the previous function
def raw_to_filtered(filtered_folder, json_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    longtail_entities = set()

    for file in os.listdir(filtered_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(filtered_folder, file))
            if "entity" in df.columns:
                longtail_entities.update(df["entity"].astype(str))
    for file in os.listdir(json_folder):
        if file.endswith(".jsonl"):
            input_path = os.path.join(json_folder, file)
            output_path = os.path.join(output_folder, file)

            kept = 0
            with open(input_path, "r", encoding="utf-8") as fin, \
                    open(output_path, "w", encoding="utf-8") as fout:

                for line in fin:
                    try:
                        entity_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip broken lines
                    entity_id = entity_data.get("id") or entity_data.get("title")

                    if entity_id in longtail_entities:
                        fout.write(json.dumps(entity_data, ensure_ascii=False) + "\n")
                        kept += 1



# KEEPING ONLY THE ENTITIES THAT HAVE THE WIKI PAGE FOR ONE OF THE THREE LANGUAGES
def filter_by_wiki_presence(results_folder, jsonl_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(results_folder):
        print(file)
        if file.endswith(".csv"):
            category = os.path.splitext(file)[0]  # e.g. "Airport"
            print(category)
            csv_path = os.path.join(results_folder, file)
            jsonl_path = os.path.join(jsonl_folder, f"{category}.jsonl")
            output_path = os.path.join(output_folder, f"{category}.jsonl")

            if not os.path.exists(jsonl_path):
                continue

            df = pd.read_csv(csv_path)

            valid_entities = set(df.query("it == 0 and es == 1 and en == 0")["entity"].astype(str))

            kept = 0
            with open(jsonl_path, "r", encoding="utf-8") as fin, \
                 open(output_path, "w", encoding="utf-8") as fout:

                for line in fin:
                    try:
                        entity_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # skip broken lines

                    entity_id = entity_data.get("id") or entity_data.get("title")

                    if entity_id in valid_entities:
                        fout.write(json.dumps(entity_data, ensure_ascii=False) + "\n")
                        kept += 1

            print(f"{category}: saved {kept} entities -> {output_path}")


def filter_head_languages(jsonl_folder, csv_folder, output_folder, top_n=10):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(jsonl_folder):
        if not file.endswith(".jsonl"):
            continue

        category = os.path.splitext(file)[0]
        jsonl_path = os.path.join(jsonl_folder, file)
        csv_path = os.path.join(csv_folder, f"{category}_head.csv")

        if not os.path.exists(csv_path):
            continue

        csv_df = pd.read_csv(csv_path)
        valid_entities = set(csv_df["entity"].astype(str))

        entity_data = []
        with open(jsonl_path, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    obj = json.loads(line)
                    qid = obj.get("id")
                    if qid in valid_entities:
                        num_langs = len(obj.get("sitelinks", {}))
                        entity_data.append((qid, num_langs, obj))
                except json.JSONDecodeError:
                    continue

        if not entity_data:
            continue

        entity_data.sort(key=lambda x: x[1], reverse=True)

        top_entities = entity_data[:top_n]

        output_path = os.path.join(output_folder, f"{category}_top{top_n}.jsonl")
        with open(output_path, "w", encoding="utf-8") as outfile:
            for _, _, obj in top_entities:
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write("\n")



def rank_entities_by_rare_properties(jsonl_folder, output_folder, summary_output="claims_ranking.jsonl", top_n=10, top_m_claims=20):
    os.makedirs(output_folder, exist_ok=True)
    rare_claims_summary = {}

    for file in os.listdir(jsonl_folder):
        if not file.endswith(".jsonl"):
            continue

        category = os.path.splitext(file)[0]
        jsonl_path = os.path.join(jsonl_folder, file)

        entities = []
        all_props = []

        ##Step 1: Read entities and collect wikibase-item properties
        with open(jsonl_path, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    obj = json.loads(line)
                    qid = obj.get("id")
                    claims = obj.get("claims", {})

                    wikibase_item_props = []
                    for prop, claim_list in claims.items():
                        if any(
                                c.get("mainsnak", {}).get("datatype") == "wikibase-item"
                                for c in claim_list
                        ):
                            wikibase_item_props.append(prop)

                    if wikibase_item_props:
                        entities.append((qid, wikibase_item_props, obj))
                        all_props.extend(wikibase_item_props)
                except json.JSONDecodeError:
                    continue

        if not entities:
            continue

        # Count property frequencies
        prop_counts = Counter(all_props)
        total_entities = len(entities)
        print(total_entities)

        # Compute rarity scores for entities, which entities have the rarest claim
        ranked = []
        for qid, props, obj in entities:
            rarity_score = sum(1 / prop_counts[p] for p in props if p in prop_counts)
            ranked.append((qid, rarity_score, obj))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Take top N rare entities
        top_entities = ranked[:top_n]
        output_path = os.path.join(output_folder, f"{category}_top{top_n}_rare.jsonl")

        with open(output_path, "w", encoding="utf-8") as outfile:
            for _, score, obj in top_entities:
                obj["_rarity_score"] = score
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write("\n")


        # Identify rarest wikibase-item claims for this category
        rare_claims = sorted(prop_counts.items(), key=lambda x: x[1])[:top_m_claims]
        rare_claims_summary[category] = [
            {"claim": claim, "frequency": freq, "rarity": round(1 / freq, 4)}
            for claim, freq in rare_claims
        ]

    # Step 6: Save the rare claims summary to a single JSON
    summary_path = os.path.join(output_folder, summary_output)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(rare_claims_summary, summary_file, ensure_ascii=False, indent=2)

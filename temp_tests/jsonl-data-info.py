import os
import json

import matplotlib.pyplot as plt

ALL_TOPICS = [
    "Science - physics, chemistry, biology, astronomy, etc.",
    "Mathematics - algebra, geometry, calculus, statistics, etc.",
    "Technology - computers, engineering, AI, robotics, etc.",
    "Business - economics, finance, marketing, management, entrepreneurship",
    "History - ancient, medieval, modern, world history, military history",
    "Geography - countries, capitals, landscapes, maps, oceans, rivers",
    "Literature - poetry, novels, plays, short stories, genres, authors",
    "Philosophy - logic, ethics, political philosophy, existentialism",
    "Psychology - cognition, development, disorders, therapy, social psychology",
    "Sociology - culture, demographics, institutions, social change, inequality",
    "Politics - political systems, ideologies, voting, campaigns, public policy",
    "Law - constitutional law, criminal law, contracts, litigation, civil rights",
    "Medicine - anatomy, diseases, treatments, pharmaceuticals, medical specialties",
    "Religion - Christianity, Islam, Judaism, Buddhism, Hinduism, atheism",
    "Mythology - Greek, Roman, Norse, Egyptian, Native American myths",
    "Art - art history, painting, sculpture, architecture, music, theater",
    "Sports - individual and team sports, athletes, championships, training",
    "Cooking - recipes, ingredients, techniques, regional and ethnic cuisine",
    "Movies & TV - genre analysis, directors, actors, awards, popular shows",
    "News & Current Events - reporting on latest happenings around the world",
    "Culture - customs, values, gender roles, holidays, language, clothing",
    "Relationships - family, friends, romantic relationships, dating, marriage",
    "Education - teaching methods, curriculum, policy, higher education, vocational",
    "Transportation - cars, planes, trains, ships, public transit, infrastructure",
    "Communication - language acquisition, linguistics, rhetoric, social media",
    "Agriculture - farming techniques, crops, livestock, fisheries, forestry",
    "Housing & Architecture - interior design, urban planning, real estate, remodeling",
    "Nature & Environment - ecology, sustainability, conservation, renewable energy",
    "Travel & Tourism - destinations, lodging, adventure, culture, ecotourism",
    "Music - theory, genres, instruments, bands, composers, music history",
    "Fashion - designers, trends, modeling, retail, cosmetics, accessories",
    "Government - political systems, public administration, foreign policy, voting",
    "Warfare - military history, weapons, strategy, special operations forces",
    "Space - astronomy, spaceflight, exploration, space technology, universe",
    "Weather & Climate - meteorology, forecasting, natural disasters, seasons",
    "Food & Cooking - nutrition, recipes, diets, food science, restaurants",
    "Pets & Animals - breeds, care, veterinary medicine, wildlife, animal behavior",
    "Gardening - plants, landscaping, flowers, vegetables, lawn care, tools",
    "Home Improvement - repair, decor, renovation, tools, plumbing, electricity",
    "Personal Finance - budgeting, investing, taxes, insurance, retirement",
    "Exercise & Fitness - techniques, equipment, sports medicine, motivation",
    "Health & Medicine - biology, anatomy, diseases, treatments, wellness",
    "Mental Health - psychology, disorders, counseling, self-help, mindfulness",
    "Race & Ethnicity - cultures, discrimination, identity, immigration, diversity",
    "Gender & Sexuality - LGBTQ issues, feminism, roles, relationships, equality",
    "Employment - careers, human resources, resumes, workplace culture, unions",
    "Crime & Justice - laws, law enforcement, courts, prisons, investigations",
    "Social Issues - poverty, homelessness, human rights, community service",
    "Technology - computers, engineering, artificial intelligence, innovations",
    "Entertainment - movies, television, games, comedy, performing arts",
]

# Seed dataset is a jsonl file
def load_instances(file_path: str):
    instances = []
    with open(file_path) as f:
        for line in f:
            instances.append(json.loads(line))   
    return instances

dataset_path = '/path/to/outputfile/sensei-4c84.jsonl'
instances = load_instances(file_path=dataset_path)

# print("= First Instance =")
# print(instances[0])

min_len_in = 4000
max_len_in = 0

min_len_out = 4000
max_len_out = 0

empty_response = 0
empty_instruction = 0

tags_in = 0
tags_out = 0

topic_leaked = 0
question_str = 0
subject_area_str = 0

input_prompt_leaked = 0
output_prompt_leaked = 0

input_lens = []
output_lens = []

TAGS = ['<s>','[INST]','[/INST]','</s>']

for instance in instances:
    if min_len_in > len(instance["input"]): min_len_in = len(instance["input"])
    if max_len_in < len(instance["input"]): max_len_in = len(instance["input"])
    if min_len_out > len(instance["output"]): min_len_out = len(instance["output"])
    if max_len_out < len(instance["output"]): max_len_out = len(instance["output"])
    if len(instance["input"]) == 0: empty_instruction+=1
    if len(instance["output"]) == 0: empty_response+=1
    
    # Input 
    # Detect prompt leakage 
    if instance["input_prompt"].lower() in instance["input"].lower():
        input_prompt_leaked+=1

    # Detect topic leakage 
    for topic in ALL_TOPICS:
        if topic in instance["input"]: 
            topic_leaked+=1
    # Detect 'Subject Area:' leakage
    if 'Subject Area:'.lower() in instance["input"].lower():
        subject_area_str+=1
    # Detect 'Question:' leakage
    if 'Question:'.lower() in instance["input"].lower():
        question_str+=1

    # Output
    # Detect prompt leakage
    if instance["output_prompt"].lower() in instance["output"].lower():
        output_prompt_leaked+=1

    # Input and output
    # Check tags leakage
    for tag in TAGS:
        if tag in instance["input"]: tags_in+=1
        if tag in instance["output"]: tags_out+=1

    # Comput avg input and output lengths

    # Add lengths to sample
    input_lens.append(len(instance["input"]))
    output_lens.append(len(instance["output"]))

def Average(lst): 
    return sum(lst) / len(lst) 

avg_len_in = Average(input_lens)
avg_len_out = Average(output_lens)

print(f"Dataset: {dataset_path}")
print(f"# Instances: {len(instances)}")
print(f"Instructions min length: {min_len_in}")
print(f"Instructions max length: {max_len_in}")
print(f"Instructions avg length: {avg_len_in}")
print(f"Responses min length: {min_len_out}")
print(f"Responses max length: {max_len_out}")
print(f"Responses avg length: {avg_len_out}")

print(f"# Empty Instructions: {empty_instruction}") # should be zero, as we filter them out during generation
print(f"# Empty Responses: {empty_response}") # should be zero, as we filter them out during generation
print(f"# Instructions with leaked tags: {tags_in}")
print(f"# Responses with leaked tags: {tags_out}")

print(f"# Instructions with leaked prompt: {input_prompt_leaked}")
print(f"# Instructions with leaked topic: {topic_leaked}")
print(f"# Instructions containing 'Question:': {question_str}")
print(f"# Instructions containing 'Subject Area:': {subject_area_str}")
print(f"# Responses with leaked prompt: {output_prompt_leaked}")

# Histogram instruction lengths
plt.hist(input_lens, bins=50, color='skyblue', edgecolor='black', range=[0, 6000])
plt.xlabel('Input Lengths')
plt.ylabel('Samples Count')
plt.title('Input Lengths Distribution')
plt.savefig('hist_input_lengths.png')

# Histogram output lengths
plt.hist(output_lens, bins=100, color='skyblue', edgecolor='black', range=[0, 10000])
plt.xlabel('Output Lengths')
plt.ylabel('Samples Count')
plt.title('Output Lengths Distribution')
plt.savefig('hist_output_lengths.png')
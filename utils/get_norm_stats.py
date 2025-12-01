import json

def calculate_global_action_stats(jsonl_path):
    global_min = [float('inf')] * 7
    global_max = [float('-inf')] * 7
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            action_min = data['stats']['action']['min']
            action_max = data['stats']['action']['max']
            
            for i in range(7):
                if action_min[i] < global_min[i]:
                    global_min[i] = action_min[i]
            
            for i in range(7):
                if action_max[i] > global_max[i]:
                    global_max[i] = action_max[i]
    
    print(f"action.min global minimums per position: {global_min}")
    print(f"action.max global maximums per position: {global_max}")

calculate_global_action_stats("/datasets/mantis_libero_lerobot/meta/episodes_stats.jsonl")

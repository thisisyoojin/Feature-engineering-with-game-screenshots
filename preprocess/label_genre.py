import os
from db import cursor


genre_map = {
    "action": 0,
    "adventure": 1,
    "rpg": 2,
    "shooting": 3,
    "simulation": 4,
    "strategy": 5,
    "arcade": 6,
    "puzzle": 7,
    "casual": 8,
}


def label_genre(tags, labels_cnt):
    """
    Creating the label for genre
    
    params
    ======
    tags(str): a text multiple tags joined with separator
    labels_cnt(dict): dictionary the number of tags(values) by genre(key)

    return
    ======

    """
    labels = [0]*len(genre_map)
    
    # when find the specific genre in the string of tags, raise the count
    def add_to_label(genre):
        idx = genre_map[genre]
        labels[idx] = 1
        cur = labels_cnt.get(genre, 0)
        labels_cnt[genre] = cur+1

    if ('action' in tags):
        add_to_label('action')

    if ('adventure' in tags):
        add_to_label('adventure')

    if ('shooter' in tags) or ('fps' in tags) or ("shoot'em up" in tags) or ('sniper' in tags):
        add_to_label('shooting')
    
    if ('strategy' in tags):
        add_to_label('strategy')
    
    if ('sim' in tags) or ('racing' in tags) or ('sports' in tags):
        add_to_label('simulation')

    if ('arcade' in tags) or ('platformer' in tags):
        add_to_label('arcade')

    if ('casual' in tags):
        add_to_label('casual')
    
    if ('rpg' in tags):
        add_to_label('rpg')

    if ('puzzle' in tags):
        add_to_label('puzzle')

    # creates the string all joined labels with comma
    labels = ','.join([str(label) for label in labels])

    return labels, labels_cnt
    

    
def create_label_csv(root_dir):
    """
    Create csv file for label(genres)

    params
    ======
    root_dir(str): the folder to store data(images and csv file)
    """
    
    FILE_PATH = f'{root_dir}/multilabel.csv'

    if os.path.isfile(FILE_PATH):
        os.remove(FILE_PATH)
    with open(f'{root_dir}/multilabel.csv', 'a') as f:
        f.write("appid,label,filepaths\n")
    
    
    screenshot = {}
    for f in os.listdir(root_dir):
        app_id = f.split('_')[0]
        if app_id in screenshot.keys():
            screenshot[app_id].append(f)
        else:
            screenshot[app_id] = [f]
    
    labels_cnt = {}
    
    for app_id, pics in screenshot.items():
        
        result = cursor.execute(f"SELECT * FROM tags WHERE appid = '{app_id}';")        
        tags = result.fetchall()
        
        if len(tags) > 0:
            tag = tags[0][1].lower()
            label, labels_cnt = label_genre(tag, labels_cnt)            
            with open(f'{root_dir}/multilabel.csv', 'a') as f:
                f.write(f"{app_id},\"{label}\",\"{','.join(pics)}\"\n")

    return labels_cnt


create_label_csv("./steam-data")

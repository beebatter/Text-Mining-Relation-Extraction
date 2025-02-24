import requests

def get_wikipedia_entity(mention):
    """ 从 Wikipedia API 获取实体链接 """
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={mention}&limit=1&format=json"
    response = requests.get(url)
    try:
        entity_link = response.json()[3][0]
        return entity_link
    except:
        return "No entity found"

if __name__ == "__main__":
    mention = "Obama"
    entity = get_wikipedia_entity(mention)
    print(f"实体链接: {entity}")
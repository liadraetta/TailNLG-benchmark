from src.wiki_popularity import PopularityRetriever

page = 'Arrow_of_God'


wp = PopularityRetriever()

page = wp.get_wikipedia_title('Arrow_of_God')

wd = wp.get_wikimedia_item_from_title(page)

pages = wp.get_wikipedia_pages(wd)


entity = dict()
entity[page] = {}
entity['wd'] = wd

for item in pages:
    try:
        x = wp.get_pageviews(article=pages[item],project=f"{item}.wikipedia.org")
        y = wp.get_page_text_length(article=pages[item],project=item)
        entity[page][item] = {
            'title': pages[item],
            'pageviews': x,
            'text_length': y
        }
    except:
        entity[page][item] = {
            'title': pages[item],
            'pageviews': 0,
            'text_length': 0
        }

print(entity)
from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/acer-ek-series-60-45-cm-24-inch-full-hd-led-backlit-ips-panel-250-nits-brightness-hdmi-vga-ports-cable-eye-care-monitor-ek240y-p6/p/itma2e8121907189?pid=MONH8A7DZTZYGRR9&lid=LSTMONH8A7DZTZYGRR9C6GOXQ&marketplace=FLIPKART&store=6bo%2Fg0i%2F9no&srno=b_1_3&otracker=browse&otracker1=hp_rich_navigation_PINNED_neo%2Fmerchandising_NA_NAV_EXPANDABLE_navigationCard_cc_3_L2_view-all&fm=organic&iid=27dda7e7-0a59-4871-b16e-9e2ea430ff82.MONH8A7DZTZYGRR9.SEARCH&ppt=None&ppn=None&ssid=g9w7gc9cn40000001761619395155'
loader = WebBaseLoader(url)

docs = loader.load()


print(docs[0].page_content)
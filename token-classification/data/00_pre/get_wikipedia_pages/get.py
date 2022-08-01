from io import StringIO
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        self.text.write(d)

    def get_data(self):
        return self.text.getvalue()


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def get_wikipedia_text(name):
    """
    example
    url : "https://de.wikipedia.org/wiki/Angela_Merkel"
    """
    print(name)
    url = f"https://de.wikipedia.org/wiki/{name}"

    with urllib.request.urlopen(url) as f:
        data = f.read().decode("utf-8")

    soup = BeautifulSoup(data, "html.parser")

    paratext = soup.find_all("p")
    len(paratext)

    D = pd.DataFrame()

    for ix_paragraph, text in enumerate(paratext):
        new_text = strip_tags(str(text)).strip()

        newD = pd.DataFrame({"new_text": new_text}, index=[ix_paragraph])
        D = pd.concat([D, newD], axis=0, ignore_index=True)

    # context = context[:512]
    # print(context)

    filename = f"token-classification/data/00_pre/get_wikipedia_pages/{name}"
    D.to_csv(f"{filename}.csv", index=False)
    return None


if __name__ == "__main__":
    names = [
        # "Albert_Einstein",
        # "Alexei_Alexejewitsch_Abrikossow",
        # "Isamu_Akasaki",
        # "Schores_Iwanowitsch_Alfjorow",
        "Hannes_Alfven",
        "Luis_Walter_Alvarez",
        "Hiroshi_Amano",
        "Carl_David_Anderson",
        "Philip_Warren_Anderson",
        "Edward_Victor_Appleton",
        "Arthur_Ashkin",
        "John_Bardeen",
        "Barry_Barish",
        "Charles_Glover_Barkla",
        "Nikolai_Gennadijewitsch_Bassow",
        "Henri_Becquerel",
        "Georg_Bednorz",
        "Hans_Bethe",
        "Gerd_Binnig",
        "Patrick_Blackett,_Baron_Blackett",
        "Felix_Bloch",
        "Nicolaas_Bloembergen",
        "Aage_Niels_Bohr",
        "Niels_Bohr",
        "Max_Born",
        "Walther_Bothe",
        "Willard_Boyle",
        "William_Henry_Bragg",
        "William_Lawrence_Bragg",
        "Walter_Houser_Brattain",
        "Ferdinand_Braun",
        "Percy_Williams_Bridgman",
        "Bertram_Brockhouse",
        "Louis_de_Broglie",
    ]

    for name in names:
        get_wikipedia_text(name)

    print("finished")

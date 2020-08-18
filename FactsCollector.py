# -*- coding: utf-8 -*-
"""
    Consuming REST api from politifact.com
    Data pre-processing and cleaning steps
    Writing the data to csv file.

    ***Note***
    Execution time will be higher as per the number of pages to scan.

"""

import requests
import pandas as pd
from bs4 import BeautifulSoup

finalData = []
# keywords = ['covid','corona','virus']
keywords = ['facebook', 'social', 'twitter']
for page in range(1, 2000):
    # finalData = []
    url = 'https://www.politifact.com/api/factchecks/?page=' + str(page)
    response = requests.get(url)
    data = response.json()
    result = []
    if 'results' in data:
        result = data['results']

    if result != []:
        for res in result:

            Statement = BeautifulSoup(res['statement'], "lxml").text
            Statement = Statement.encode('ascii', 'ignore')
            Statement = Statement.decode('utf-8', 'ignore')

            Sources = BeautifulSoup(res['sources'], "lxml").text
            Sources = Sources.encode('ascii', 'ignore')
            Sources = Sources.decode('utf-8', 'ignore')

            # if b'corona' in Statement.lower():
            if any(word in Sources.lower() for word in keywords):
                # if True:
                # if b'covid' in Statement:
                Sequence = res['id']

                Ruling_Comments = BeautifulSoup(res['ruling_comments'], "lxml").text
                Ruling_Comments = Ruling_Comments.encode('ascii', 'ignore')
                Ruling_Comments = Ruling_Comments.decode('utf-8', 'ignore')

                Ruling_Slug = res['ruling_slug']

                Slug = res['slug']

                Speaker = res['speaker']['full_name']
                Speaker_Slug = res['speaker']['slug']

                finalData.append(
                    [Sequence, Ruling_Comments, Slug, Sources, Speaker, Speaker_Slug, Statement, Ruling_Slug])
dataset = pd.DataFrame(finalData)
dataset.columns = ['Sequence', 'Ruling_Comments', 'Slug', 'Sources', 'Speaker', 'Speaker_Slug', 'Statement',
                   'Ruling_Slug']
# finalData
# dataset
# response.json()
dataset

dataset = pd.DataFrame(dataset)
dataset.to_csv('data/APIData1608-FB.csv')
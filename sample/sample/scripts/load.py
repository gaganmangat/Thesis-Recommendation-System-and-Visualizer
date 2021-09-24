import csv
import sys
from home.models import Destination

def run():
    # f = open('./data_project.csv',encoding='utf-8')
    f = open('./data_cleaned.csv',encoding='utf-8')
    reader = csv.reader(f)

    Destination.objects.all().delete()
    count = 0
    for row in reader:
        # print(type(row[7]))
        # sys.exit()
        
        d,created = Destination.objects.get_or_create(
            id = count,
            Title = str(row[1]),
            Author = row[2],
            Supervisor = row[3],
            Degree = row[4],
            Department = row[5],
            Abstract = row[6],
            URL = row[8],
            Date1 = (row[7])
        )

        count += 1

# print('helo')
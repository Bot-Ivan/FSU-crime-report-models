#fsu crime stuff
#We want to generate a crime report for the summer of 2024, July to september, based on a machine learning prediction created using an algorithm we have not decided on yet.
 

# calendar.weekday(year, month, day)
# Returns the day of the week (0 is Monday) for year (1970–…), month (1–12), day (1–31).
#calendar.day_name[weekday] // turns the numbered days to names ex. 0 to Monday


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import sklearn.linear_model

def getWeekday(day):
        date_parts = str(day).split()
        date_str = str(date_parts[0])
        # # print (date_str)
        month, day, year = map(int, date_str.split("/"))
        weekday = calendar.weekday(year, month, day)

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_name = day_names[weekday]
        return weekday_name
def getHour(time):
        time_parts = str(time).split()
        time_str = str(time_parts[1])
        hour_parts = time_str.split(":")
        hour_str = hour_parts[0]

        return int(hour_str)

def get_zip_code(fullLocation):
       return fullLocation[len(fullLocation)-5:]



id = pd.read_csv("FSUcrime.csv")
id.drop(id.index[-1], inplace = True)
id["Weekdays"] = id["Reported Date/Time"].apply(getWeekday)
new_row1 = id.pop("Weekdays")
id.insert(0, "Weekday", new_row1)

id["Hour"] = id["Reported Date/Time"].apply(getHour)
custom_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a custom categorical data type with the desired order
weekday_category = pd.Categorical(id['Weekday'], categories=custom_order, ordered=True)

# Assign the categorical data type back to the "weekdays" column
id['Weekday'] = weekday_category

# Sort the DataFrame based on the "weekdays" column
id.sort_values(by='Weekday', inplace=True)

# Reset the index to have a clean index order
id.reset_index(drop=True, inplace=True)
id = id.drop(columns=["Number", "Disposition", "Reported Date/Time", "Occurred From Date Time"])
id["Zip Code"] = id["Formatted Location"].apply(get_zip_code)
# print(id)
# id.plot(x = id["Hour"], y = id["Reported Date/Time"]["Disturbance\nVerbal/Noise"])
# plt.show()

id = pd.get_dummies(id, columns = ["Occurred Incident Type"], prefix = "incident_type")
id = pd.get_dummies(id, columns = ["Zip Code"], prefix = "")
print(id.columns)

# Possibly use scikit-learn for training
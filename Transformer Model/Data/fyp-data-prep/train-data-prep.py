#import time
import pandas as pd
#from IPython.display import clear_output

df = pd.read_csv("df.csv")
#print(df)
iio = pd.DataFrame(columns=["instruction", "input", "output"])

for index, row in df.iterrows():
  #358980
  if (index%193 == 0):
    #clear_output(wait=True)
    print(str(index+1) + "/358980\n")
    percent = int((index+1)/358980 * 100)
  
    for i in range(int(percent/2)):
       print("▓", end="")

    for i in range(int(50 - (percent/2))):
       print("░", end="")

    print("    " + str(percent) + "%")
  #time.sleep(1)

  q1_input = ""
  q2_input = str(int(df.at[index, "q1"])) + " "
  q3_input = str(int(df.at[index, "q1"])) + " " + str(int(df.at[index, "q2"])) + " "
  q4_input = str(int(df.at[index, "q1"])) + " " + str(int(df.at[index, "q2"])) + " " + str(int(df.at[index, "q3"])) + " "

  iio.loc[len(iio)] = [df.at[index, "class"], q1_input, str(int(df.at[index, "q1"]))]
  iio.loc[len(iio)] = [df.at[index, "class"], q2_input, str(int(df.at[index, "q2"]))]
  iio.loc[len(iio)] = [df.at[index, "class"], q3_input, str(int(df.at[index, "q3"]))]
  iio.loc[len(iio)] = [df.at[index, "class"], q4_input, str(int(df.at[index, "q4"]))]

  #print(str(df.at[index, "class"]) + "\t" + "" + "\t" + str(int(df.at[index, "q1"])))
  #print(str(df.at[index, "class"]) + "\t" + str(int(df.at[index, "q1"])) + "\t" + str(int(df.at[index, "q2"])))
  #print(str(df.at[index, "class"]) + "\t" + str(int(df.at[index, "q1"])) + " " + str(int(df.at[index, "q2"])) + "\t" + str(int(df.at[index, "q3"])))
  #print(str(df.at[index, "class"]) + "\t" + str(int(df.at[index, "q1"])) + " " + str(int(df.at[index, "q2"])) + " " + str(int(df.at[index, "q3"])) + "\t" + str(int(df.at[index, "q4"])))

print("DONE!!")
# Start 11:00 pm
# 5%  in 6 min   eta 2 hr
# 10% in 25 min  eta 4 hr 10 min
# 20% in 95 min  eta 7 hr 55 min
# 25% in 150 min 
# 30% in 210 min
# Convert DataFrame to JSON and save to a file
iio.to_json('iio_train.json', orient='records', lines=True)


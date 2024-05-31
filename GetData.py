import pandas as pd

# Load the CSV file
csv_file_path = 'C:/Users/bgpra/OneDrive/Documents/GitHub/Baby-GPT/Friends.csv'
df = pd.read_csv(csv_file_path)
x=0
# Filter out rows where the "Speaker" field is blank
filtered_df = df[df['Speaker'].notna()]

# Open the text file for writing
with open('C:/Users/bgpra/OneDrive/Documents/GitHub/Baby-GPT/Friends_script.txt', 'w', encoding='utf-8') as file:
    # Iterate through the DataFrame and write to the text file in the specified format
    for index, row in filtered_df.iterrows():
        speaker = row['Speaker']
        text = row['Text']
        x=x+1
        print(x)

        if 'scene' in text.lower() or 'scene' in speaker.lower():
            continue
        file.write(f'{speaker.upper()}: {text}\n')


print("Script has been successfully written to Friends_script.txt")

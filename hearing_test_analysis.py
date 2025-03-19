import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load similarity ratings JSON
    SAVE_FILE = "similarity_ratings.json"
    with open(SAVE_FILE, "r") as f:
        ratings = json.load(f)

    def select_by_program_and_create_df(ratings, program_number):
        # Create a list to store function names and Likert scores
        data = []

        # Iterate over the ratings dictionary
        for key, score in ratings.items():
            # Split the key to extract the program number
            parts = key.split("_")
            program_id = int(parts[-2])  # The second to last element is the program number

            # If the program number matches the input, process the key
            if program_id == program_number:
                # Extract the function name
                base_name = "_".join(parts[:2])

                # Remove numbers from the end of function names like "JTFS_1"
                if base_name.startswith("JTFS"):
                    base_name = "JTFS"

                # Append the function name and score to the data list
                data.append([base_name, score])
        
        # Create a DataFrame from the data list
        df = pd.DataFrame(data, columns=["Function Name", "Likert Score"])
        
        return df

    # Example usage: Select ratings for program 1 and create the DataFrame
    program_number = 3
    df = select_by_program_and_create_df(ratings, program_number)

    # Print the resulting DataFrame
    print(df)

    return (
        SAVE_FILE,
        df,
        f,
        json,
        pd,
        plt,
        program_number,
        ratings,
        select_by_program_and_create_df,
        sns,
    )


@app.cell
def __(df, plt, sns):


    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Function Name", y="Likert Score", data=df)
    plt.title("Likert Score Comparison by Function")
    plt.show()

    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

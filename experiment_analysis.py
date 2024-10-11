import marimo

__generated_with = "0.5.2"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from helpers.experiment_scripts import load_json
    return load_json, mo


app._unparsable_cell(
    r"""
    d = load_json(\"results/experiments.json\")
    d = 
    """,
    name="__"
)


if __name__ == "__main__":
    app.run()

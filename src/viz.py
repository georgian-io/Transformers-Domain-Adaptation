import pandas as pd
import plotly.express as px


def parallel_coord_plot(df: pd.DataFrame, metric: str = "test_f1"):
    # Treat string columns as categorical and convert into numbers for visualization
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]

    # Remove columns if all value are the same
    df = df[[col for col in df.columns if df[col].nunique() > 1]]

    # Filter metrics that are not part the `metric`
    df = df.drop([[col for col in df.columns if 'test' in col and col != metric]], axis=1)

    return px.parallel_coordinates(df, color=metric)

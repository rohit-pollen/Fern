import inspect

def print_df_shapes_auto(*dfs):
    """
    Prints the names and shapes of all passed DataFrames.
    """
    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals

    for df in dfs:
        matching_names = [name for name, val in local_vars.items() if val is df]
        name = matching_names[0] if matching_names else "<unknown>"
        print(f"{name}: shape = {df.shape}")
from vnstock import Listing
try:
    indices = Listing().all_indices()
    print(indices)
    # Check if it's a DataFrame or list
    import pandas as pd
    if isinstance(indices, pd.DataFrame):
        print("\nColumns:", indices.columns.tolist())
        print("\nFirst row:", indices.iloc[0].to_dict())
    else:
        print("\nType:", type(indices))
except Exception as e:
    print(e)

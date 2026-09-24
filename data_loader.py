import pandas as pd  


def load_room_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        parse_dates=["start_time", "end_time"]
    )

    return data

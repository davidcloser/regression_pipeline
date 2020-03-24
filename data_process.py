import pandas as pd
import os


class LoadData:
    def __init__(self, output_folder="output"):
        self.weather_data = pd.read_csv("data/Basel_2020-01-30_WeatherData.csv", index_col=0, sep=";")
        self.weather_data.index = pd.to_datetime(self.weather_data.index)

        self.price_data = pd.read_csv("data/RhineFreightData.csv", index_col=0, sep=";")
        self.price_data.index = pd.to_datetime(self.price_data.index)

        self.kaub_level_url = "https://www.contargo.net/en/goodtoknow/lws/history/"
        self.kaub_level_df = self.get_kaub_from_url()

        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def process_data(self) -> pd.DataFrame:
        # Merge kaub_level_df
        self.price_data.loc[self.price_data["Kaub_level"].isna(), "Kaub_level"] = self.kaub_level_df["Level Kaub"]
        self.price_data.interpolate(method="linear", inplace=True)
        assert self.price_data.isna().sum().sum() == 0
        assert self.weather_data.isna().sum().sum() == 0

        processed_data = self.price_data.merge(self.weather_data, right_index=True, left_index=True)
        return processed_data

    def get_kaub_from_url(self):
        kaub_level_df = pd.concat(pd.read_html(self.kaub_level_url)[1:-1])
        kaub_level_df["Date"] = pd.to_datetime(kaub_level_df["Date"], format="%d %b %Y")
        kaub_level_df.set_index("Date", inplace=True)
        return kaub_level_df


if __name__ == "__main__":
    dataprocess = LoadData()
    processed_data = dataprocess.process_data()


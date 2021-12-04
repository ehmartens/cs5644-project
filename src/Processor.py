from src.Reader import BarClosuresReader, CasesAndDeathsReader, GatheringBansReader, MaskMandatesReader, RestaurantClosuresReader, StayAtHomeOrdersReader, VaccinationsReader
import pandas as pd

class PreProcessor:
    def __init__(self, home_dir='', state_filter=None):
        self.current_data = None
        self.home_dir = home_dir
        self.state_filter = state_filter
        self.barClosuresReader = BarClosuresReader(home_dir=home_dir)
        self.gatheringBansReader = GatheringBansReader(home_dir=home_dir)
        self.maskMandatesReader = MaskMandatesReader(home_dir=home_dir)
        self.restaurantClosuresReader = RestaurantClosuresReader(home_dir=home_dir)
        self.stayAtHomeOrdersReader = StayAtHomeOrdersReader(home_dir=home_dir)
        self.casesAndDeathsReader = CasesAndDeathsReader(home_dir=home_dir)
        self.vaccinationsReader = VaccinationsReader(home_dir=home_dir)

    def clear(self):
        self.current_data = None

    def get_current_data(self):
        return self.current_data

    def set_current_data(self, data):
        self.current_data = data

    def load_processed_data(self):
        self.load_processed_data_without_fips_as_columns() # TODO fix this so that it checks if the current data is already loaded instead of reloading.
        merged_df = self.get_current_data()
        # Transform FIPs into boolean columns
        merged_df = self.barClosuresReader.convert_categorical(merged_df) # TODO this is hacky, fix this
        # Set current data
        self.current_data = merged_df

    def load_processed_data_without_fips_as_columns(self):
        # Bring in CDC data and merge with  Vaccinations data
        merged_df = self.init_cdc_data().merge(self.init_vaccinations_data(), how='left', on=['date', 'FIPS'])
        # Bring in Cases and Deaths data and merge to current data
        merged_df = merged_df.merge(self.init_cases_and_deaths_data(), how='left', on=['date', 'FIPS'])
        # Fill NAs in cases and deaths data with 0s
        merged_df['cases'] = merged_df['cases'].fillna(0)
        merged_df['deaths'] = merged_df['deaths'].fillna(0)
        # Make columns numeric (except FIPS and date)
        num_cols = merged_df.columns.drop(['date', 'FIPS'])
        merged_df[num_cols] = merged_df[num_cols].apply(pd.to_numeric, errors='ignore')
        # Fillnas of Vaccine columns with 0
        merged_df['Series_Complete_Pop_Pct'] = merged_df['Series_Complete_Pop_Pct'].fillna(0)
        merged_df['Administered_Dose1_Pop_Pct'] = merged_df['Administered_Dose1_Pop_Pct'].fillna(0)
        merged_df['Series_Complete_Pop_Pct_UR_Equity'] = merged_df['Series_Complete_Pop_Pct_UR_Equity'].fillna(0)
        # Filnas in the metro column based on the values in the metro column for that same FIPS code
        merged_df['is_metro'] = merged_df['is_metro'].fillna(merged_df.groupby(['FIPS'])['is_metro'].transform('max'))
        # Create a 'month' feature based on the date column
        merged_df['date'] = merged_df['date'].astype('datetime64[ns]')
        merged_df['month'] = merged_df['date'].apply(lambda x: x.strftime('%B'))
        # Create a time index based on the number of days since t0
        min_date = merged_df['date'].min()
        merged_df['days_from_start'] = merged_df['date'].apply(lambda x: (x - min_date).days)
        # Transform FIPS codes for county into STATE column
        merged_df['STATE'] = merged_df['FIPS'].apply(lambda x: x[:2])
        # Set current data
        self.current_data = merged_df

    def init_cdc_data(self):
        cdc_regs_df = self.gatheringBansReader.read_and_process_data(state_filter=self.state_filter)
        # Using inner joins for dates to narrow to most restrictive range
        cdc_regs_df = cdc_regs_df.merge(
                            self.stayAtHomeOrdersReader.read_and_process_data(state_filter=self.state_filter)
                            , how='inner'
                            , on=['date', 'FIPS']
        )

        cdc_regs_df = cdc_regs_df.merge(
                        self.maskMandatesReader.read_and_process_data(state_filter=self.state_filter)
                        , how='inner'
                        , on=['date', 'FIPS']
                        )

        cdc_regs_df = cdc_regs_df.merge(
                        self.barClosuresReader.read_and_process_data(state_filter=self.state_filter)
                        , how='inner'
                        , on=['date', 'FIPS']
                        )
        cdc_regs_df = cdc_regs_df.merge(
                        self.restaurantClosuresReader.read_and_process_data(state_filter=self.state_filter)
                        , how='inner'
                        , on=['date', 'FIPS']
                        )
        return cdc_regs_df

    def init_cases_and_deaths_data(self):
        return self.casesAndDeathsReader.read_and_process_data(state_filter=self.state_filter)

    def init_vaccinations_data(self):
        return self.vaccinationsReader.read_and_process_data(state_filter=self.state_filter)
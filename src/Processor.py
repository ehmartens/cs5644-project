from src import Reader
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

    def load_processed_data(self,):
        # Bring in CDC data and merge with  Vaccinations data
        self.current_data = self.init_cdc_data().merge(self.init_vaccinations_data(), how='left', on=['date', 'FIPS'])
        # Bring in Cases and Deaths data and merge to current data
        self.current_data = self.current_data.merge(self.init_cases_and_deaths_data(), how='left', on=['date', 'FIPS'])
        # Transform FIPS codes into columns.
        # self.current_data = convert_categorical()

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
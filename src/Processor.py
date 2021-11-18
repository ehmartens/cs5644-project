from src.Reader import BarClosuresReader, GatheringBansReader, MaskMandatesReader, RestaurantClosuresReader, StayAtHomeOrdersReader
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

    def clear(self):
        self.current_data = None

    def get_current_data(self):
        return self.current_data

    def set_current_data(self, data):
        self.current_data = data

    def load_processed_data(self,):
        # Bring in CDC data as current data
        self.current_data = self.init_cdc_data()
        # Bring in Vaccinations data and merge to current data
        # Bring in Cases and Deaths data and merge to current data

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
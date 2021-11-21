from datetime import datetime 
from os.path import exists
import pandas as pd
import numpy as np
from sklearn import preprocessing

class Reader:
    def __init__(self, home_dir, data_path):
        self.home_dir = home_dir
        self.data_path = data_path
        self.data_exists = self.check_data_exists()
    
    def get_data_exists(self):
        return self.data_exists
    
    def set_data_exists(self, exists_flag):
        self.data_exists = exists_flag

    def check_data_exists(self):
        if not exists(self.home_dir + self.data_path):
            self.set_data_exists(False)
            return False
        else:
            self.set_data_exists(True)
            return True

    def convert_categorical(self, X):
        """
        Converts str columns into categorical columns
        Note that this function is from provided code for CS5834 Intro to Urban Computing.
        """
        columns = X.columns
        indices = X.index
        new_columns = []
        encoded_x = None
        for i in range(0, len(columns)):
            if X[columns[i]].dtype!='O': continue
            label_encoder = preprocessing.LabelEncoder()
            le = label_encoder.fit(X[columns[i]].apply(str))
            for class_ in le.classes_:
                new_columns.append("{}_{}".format(columns[i],class_))
            feature = le.transform(X[columns[i]].apply(str))
            feature = feature.reshape(X.shape[0], 1)
            onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
            onehot_encoder.fit(feature)
            feature = onehot_encoder.transform(feature)
            if encoded_x is None:
                encoded_x = feature
            else:
                encoded_x = np.concatenate((encoded_x, feature), axis=1)
            X = X.drop(columns[i], axis=1)
        new_columns.extend(X.columns)
        X = pd.DataFrame(np.concatenate((encoded_x,X),axis=1),index=indices,columns=new_columns)
        return X

    def zero_pad_str(self, code_str, length):
        result = code_str
        while len(result) < length:
            result = '0' + result
        return result

class GatheringBansReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cdc_regs/U.S._State_and_Territorial_Gathering_Bans__March_11__2020-August_15__2021_by_County_by_Day.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Gathering bans data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        gathering_bans_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , usecols=[
                'FIPS_State'
                , 'FIPS_County'
                , 'date'
                , 'General_GB_order_group'
                , 'Express_Preemption'
                , 'Source_of_Action'
                ]
            )
        return gathering_bans_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        gathering_bans_df = self.read_raw_data()
        # Filter to specific State FIPS Codes
        if state_filter:
            gathering_bans_df = gathering_bans_df[gathering_bans_df['FIPS_State'].isin(state_filter)]
        # Formatting dates
        gathering_bans_df['date'] = gathering_bans_df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        # Renaming columns to make it clear what they represent once joined to other CDC mandates data
        gathering_bans_df = gathering_bans_df.rename(
                                columns={
                                    'General_GB_order_group' :'gathering_ban_order_group'
                                    , 'Express_Preemption' : 'gathering_ban_express_preemption'
                                    , 'Source_of_Action' : 'gathering_ban_source_of_action'
                                }
                            )
        # Renaming gathering_ban_order_group fields to be more amenable to transformation:
        order_group_dict = {
            'Bans gatherings of any size' : 'ban_gatherings_of_any_size'
            , 'Ban of gatherings over 1-10 people': 'ban_over_1_to_10_ppl'
            , 'Ban of gatherings over 11-25 people' : 'ban_over_11_to_25_ppl'
            , 'Ban of gatherings over 26-50 people': 'ban_over_26_to_50_ppl'
            , 'Ban of gatherings over 51-100 people': 'ban_over_51_to_100_ppl'
            , 'Ban of gatherings over 101 or more people': 'ban_over_101_or_more_ppl'
            , 'No order found' :'no_order_found'
        }
        gathering_bans_df['gathering_ban_order_group'] = gathering_bans_df['gathering_ban_order_group'].apply(lambda x: order_group_dict[x]) 
        # Renaming gathering_ban_source_of_action fields to be more amenable to transformation:
        source_of_action_dict = {
            'News' : 'news'
            , 'News Media' : 'news'
            , 'Offcial': 'official' # Typo in source data
            , 'Official': 'official'
            , 'Official Announcement' : 'official_announcement'
            , 'Press Release' : 'press_release'
        }
        gathering_bans_df['gathering_ban_source_of_action'] = gathering_bans_df['gathering_ban_source_of_action'].apply(lambda x: x if pd.isna(x) else source_of_action_dict[x]) 
        # Transform to categorical variables as needed.
        gathering_bans_df = self.convert_categorical(gathering_bans_df)
        # Format FIPS county codes
        gathering_bans_df['FIPS_County'] = gathering_bans_df['FIPS_County'].apply(lambda x: self.zero_pad_str(str(x), 3))
        # Format FIPS state codes
        gathering_bans_df['FIPS_State'] = gathering_bans_df['FIPS_State'].apply(lambda x: self.zero_pad_str(str(x), 2))
        # Create single FIPS column
        gathering_bans_df['FIPS']  = gathering_bans_df['FIPS_State'] + gathering_bans_df['FIPS_County']
        # Drop unneccessary columns
        gathering_bans_df = gathering_bans_df.drop(['FIPS_State', 'FIPS_County'], axis=1)
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/gathering_bans_df.csv'
            gathering_bans_df.to_csv(export_path, index=False)
        # Return processed df
        return gathering_bans_df

class MaskMandatesReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cdc_regs/U.S._State_and_Territorial_Public_Mask_Mandates_From_April_10__2020_through_August_15__2021_by_County_by_Day.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Mask Mandates data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        mask_mandates_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , usecols=[
                'FIPS_State'
                , 'FIPS_County'
                , 'date'
                , 'order_code'
                , 'Face_Masks_Required_in_Public'
                , 'Source_of_Action'
                ]
            )
        return mask_mandates_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        mask_mandates_df = self.read_raw_data()
        # Filter to specific State FIPS Codes
        if state_filter:
            mask_mandates_df = mask_mandates_df[mask_mandates_df['FIPS_State'].isin(state_filter)]
        # Formatting dates
        mask_mandates_df['date'] = mask_mandates_df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        # Renaming columns to make it clear what they represent once joined to other CDC mandates data
        mask_mandates_df = mask_mandates_df.rename(
                                columns={
                                    'order_code' :'mask_mandate_order_code'
                                    , 'Express_Preemption' : 'mask_mandate_express_preemption'
                                    , 'Source_of_Action' : 'mask_mandate_source_of_action'
                                }
                            )
        # Transform categorical variables
        mask_mandates_df = self.convert_categorical(mask_mandates_df)
        # Format FIPS county codes
        mask_mandates_df['FIPS_County'] = mask_mandates_df['FIPS_County'].apply(lambda x: self.zero_pad_str(str(x), 3))
        # Format FIPS state codes
        mask_mandates_df['FIPS_State'] = mask_mandates_df['FIPS_State'].apply(lambda x: self.zero_pad_str(str(x), 2))
        # Create single FIPS column
        mask_mandates_df['FIPS']  = mask_mandates_df['FIPS_State'] + mask_mandates_df['FIPS_County']
        # Drop unneccessary columns
        mask_mandates_df = mask_mandates_df.drop(['FIPS_State', 'FIPS_County'], axis=1)
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/mask_mandates_df.csv'
            mask_mandates_df.to_csv(export_path, index=False)
        # Return processed df
        return mask_mandates_df

class StayAtHomeOrdersReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cdc_regs/U.S._State_and_Territorial_Stay-At-Home_Orders__March_15__2020___August_15__2021_by_County_by_Day.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Stay at Home Order data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        stay_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , usecols=[
                'FIPS_State'
                , 'FIPS_County'
                , 'date'
                , 'Order_code'
                , 'Stay_at_Home_Order_Recommendation'
                , 'Express_Preemption'
                , 'Source_of_Action'
                ]
            )
        return stay_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        stay_at_home_df = self.read_raw_data()
        # Filter to specific State FIPS Codes
        if state_filter:
            stay_at_home_df = stay_at_home_df[stay_at_home_df['FIPS_State'].isin(state_filter)]
        # Formatting dates
        stay_at_home_df['date'] = stay_at_home_df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        # Renaming columns to make it clear what they represent once joined to other CDC mandates data
        stay_at_home_df = stay_at_home_df.rename(
                                columns={
                                    'Order_code' :'stay_at_home_order_code'
                                    , 'Express_Preemption' : 'stay_at_home_express_preemption'
                                    , 'Source_of_Action' : 'stay_at_home_source_of_action'
                                }
                            )
        # Renaming stay_at_home_order_code fields to be more amenable to transformation:
        order_group_dict = {
            'Advisory/Recommendation' : 'advisory'
            , 'Mandatory for all individuals': 'mandatory_for_all_individuals'
            , 'Mandatory only for all individuals in certain areas of the jurisdiction' : 'mandatory_only_for_all_individuals_in_certain_areas_of_the_jurisdiction'
            , 'Mandatory only for at-risk individuals in the jurisdiction' : 'mandatory_only_for_at_risk_individuals_in_the_jurisdiction'
            , 'No order for individuals to stay home': 'no_order_for_individuals_to_stay_home'
        }
        stay_at_home_df['Stay_at_Home_Order_Recommendation'] = stay_at_home_df['Stay_at_Home_Order_Recommendation'].apply(lambda x: x if pd.isna(x) else order_group_dict[x]) 
        # Renaming stay_at_home_express_preemption fields to be more amenable to transformation:
        express_preemption_dict = {
            'Official': 'official'
            , 'Unknown' :'unknown'
            , 'Local orders moot due to statewide mandate': 'local_orders_moot_due_to_statewide_mandate'
            , 'Expressly Does Not Preempt': 'expressly_does_not_preempt'
            , 'Expressly Preempts' : 'expressly_preempts'
        }
        stay_at_home_df['stay_at_home_express_preemption'] = stay_at_home_df['stay_at_home_express_preemption'].apply(lambda x: x if pd.isna(x) else express_preemption_dict[x]) 
        # Transform categorical variables
        stay_at_home_df = self.convert_categorical(stay_at_home_df)
        # Format FIPS county codes
        stay_at_home_df['FIPS_County'] = stay_at_home_df['FIPS_County'].apply(lambda x: self.zero_pad_str(str(x), 3))
        # Format FIPS state codes
        stay_at_home_df['FIPS_State'] = stay_at_home_df['FIPS_State'].apply(lambda x: self.zero_pad_str(str(x), 2))
        # Create single FIPS column
        stay_at_home_df['FIPS']  = stay_at_home_df['FIPS_State'] + stay_at_home_df['FIPS_County']
        # Drop unneccessary columns
        stay_at_home_df = stay_at_home_df.drop(['FIPS_State', 'FIPS_County'], axis=1)
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/stay_at_home_df.csv'
            stay_at_home_df.to_csv(export_path, index=False)
        # Return processed df
        return stay_at_home_df

class BarClosuresReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cdc_regs/U.S._State_and_Territorial_Orders_Closing_and_Reopening_Bars_Issued_from_March_11__2020_through_August_15__2021_by_County_by_Day.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Bar closures data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        bars_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , usecols=[
                'FIPS_State'
                , 'FIPS_County'
                , 'date'
                , 'Action'
                , 'Source_of_Action'
                , 'Percent_Capacity_Outdoor'
                , 'Percent_Capacity_Indoor'
                , 'Limited_Open_Outdoor_Only'
                , 'Limited_Open_General_Indoor'
                ]
            )
        return bars_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        bars_df = self.read_raw_data()
        # Filter to specific State FIPS Codes
        if state_filter:
            bars_df = bars_df[bars_df['FIPS_State'].isin(state_filter)]
        # Formatting dates
        bars_df['date'] = bars_df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        # Renaming columns to make it clear what they represent once joined to other CDC mandates data
        bars_df = bars_df.rename(
                                columns={
                                    'Action' :'bars_action'
                                    , 'Source_of_Action' : 'bars_source_of_action'
                                    , 'Percent_Capacity_Outdoor' : 'bars_percent_capacity_outdoor'
                                    , 'Percent_Capacity_Indoor' : 'bars_percent_capacity_indoor'
                                    , 'Limited_Open_Outdoor_Only': 'bars_limited_open_outdoor_only'
                                    , 'Limited_Open_General_Indoor':'bars_limited_open_general_indoor'
                                }
                            )
        # Renaming bars_action fields to be more amenable to transformation:
        action_dict = {
            'Authorized to fully reopen': 'authorized_to_fully_reopen'
            , 'Curbside/carryout/delivery only' : 'curbside_or_carryout_or_delivery_only'
            , 'Open with social distancing/reduced seating/enhanced sanitation': 'open_w_social_distancing_or_reduced_seating_or_enhanced_sanitation'
            , 'Closed' : 'closed'
        }
        bars_df['bars_action'] = bars_df['bars_action'].apply(lambda x: x if pd.isna(x) else action_dict[x]) 
        # Renaming percent_capacity fields to be more amenable to transformation:
        pct_capacity_dict = {
            'Not specified': 'not_specified' # TODO consider changing this to a numberical variable
            , '30%' : '30_percent'
            , '35%' : '35_percent'
            , '50%' : '50_percent'
            , '60%' : '60_percent'
            , '75%' : '75_percent'
            , '100%' : '100_percent'
        }
        bars_df['bars_percent_capacity_outdoor'] = bars_df['bars_percent_capacity_outdoor'].apply(lambda x: x if pd.isna(x) else pct_capacity_dict[x]) 
        bars_df['bars_percent_capacity_indoor'] = bars_df['bars_percent_capacity_indoor'].apply(lambda x: x if pd.isna(x) else pct_capacity_dict[x]) 
        # Transform categorical variables
        bars_df = self.convert_categorical(bars_df)
        # Format FIPS county codes
        bars_df['FIPS_County'] = bars_df['FIPS_County'].apply(lambda x: self.zero_pad_str(str(x), 3))
        # Format FIPS state codes
        bars_df['FIPS_State'] = bars_df['FIPS_State'].apply(lambda x: self.zero_pad_str(str(x), 2))
        # Create single FIPS column
        bars_df['FIPS']  = bars_df['FIPS_State'] + bars_df['FIPS_County']
        # Drop unneccessary columns
        bars_df = bars_df.drop(['FIPS_State', 'FIPS_County'], axis=1)
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/bars_df.csv'
            bars_df.to_csv(export_path, index=False)
        # Return processed df
        return bars_df

class RestaurantClosuresReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cdc_regs/U.S._State_and_Territorial_Orders_Closing_and_Reopening_Restaurants_Issued_from_March_11__2020_through_August_15__2021_by_County_by_Day.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Restaurant closures data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        restaurants_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , usecols=[
                'FIPS_State'
                , 'FIPS_County'
                , 'date'
                , 'Action'
                , 'Source_of_Action'
                , 'Percent_Capacity_Outdoor'
                , 'Percent_Capacity_Indoor'
                , 'Limited_Open_Outdoor_Only'
                , 'Limited_Open_General_Indoor'
                ]
            )
        return restaurants_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        restaurants_df = self.read_raw_data()   
        # Filter to specific State FIPS Codes
        if state_filter:
            restaurants_df = restaurants_df[restaurants_df['FIPS_State'].isin(state_filter)]
        # Formatting dates
        restaurants_df['date'] = restaurants_df['date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))
        # Renaming columns to make it clear what they represent once joined to other CDC mandates data
        restaurants_df = restaurants_df.rename(
                                columns={
                                    'Action' :'restuarants_action'
                                    , 'Source_of_Action' : 'restuarants_source_of_action'
                                    , 'Percent_Capacity_Outdoor' : 'restuarants_percent_capacity_outdoor'
                                    , 'Percent_Capacity_Indoor' : 'restuarants_percent_capacity_indoor'
                                    , 'Limited_Open_Outdoor_Only': 'restuarants_limited_open_outdoor_only'
                                    , 'Limited_Open_General_Indoor':'restuarants_limited_open_general_indoor'
                                }
                            )

        # Renaming bars_action fields to be more amenable to transformation:
        action_dict = {
            'Authorized to fully reopen': 'authorized_to_fully_reopen'
            , 'Curbside/carryout/delivery only' : 'curbside_or_carryout_or_delivery_only'
            , 'Open with social distancing/reduced seating/enhanced sanitation': 'open_w_social_distancing_or_reduced_seating_or_enhanced_sanitation'
        }
        restaurants_df['restuarants_action'] = restaurants_df['restuarants_action'].apply(lambda x: x if pd.isna(x) else action_dict[x]) 

        # Renaming percent_capacity fields to be more amenable to transformation:
        pct_capacity_dict = {
            'Not specified': 'not_specified' # TODO consider changing this to a numberical variable
            , '25%' : '25_percent'
            , '30%' : '30_percent'
            , '35%' : '35_percent'
            , '50%' : '50_percent'
            , '60%' : '60_percent'
            , '75%' : '75_percent'
            , '100%' : '100_percent'
        }
        restaurants_df['restuarants_percent_capacity_outdoor'] = restaurants_df['restuarants_percent_capacity_outdoor'].apply(lambda x: x if pd.isna(x) else pct_capacity_dict[x]) 
        restaurants_df['restuarants_percent_capacity_indoor'] = restaurants_df['restuarants_percent_capacity_indoor'].apply(lambda x: x if pd.isna(x) else pct_capacity_dict[x]) 
        # Transform categorical variables
        restaurants_df = self.convert_categorical(restaurants_df)
        # Format FIPS county codes
        restaurants_df['FIPS_County'] = restaurants_df['FIPS_County'].apply(lambda x: self.zero_pad_str(str(x), 3))
        # Format FIPS state codes
        restaurants_df['FIPS_State'] = restaurants_df['FIPS_State'].apply(lambda x: self.zero_pad_str(str(x), 2))
        # Create single FIPS column
        restaurants_df['FIPS']  = restaurants_df['FIPS_State'] + restaurants_df['FIPS_County']
        # Drop unneccessary columns
        restaurants_df = restaurants_df.drop(['FIPS_State', 'FIPS_County'], axis=1)
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/restaurants_df.csv'
            restaurants_df.to_csv(export_path, index=False)
        # Return processed df
        return restaurants_df

class CasesAndDeathsReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/cases_and_deaths/us-counties.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Cases and Deaths data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        cases_deaths_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , dtype={'fips':str}
            , usecols=[
                'date'
                , 'fips'
                , 'cases'
                , 'deaths'
                ]
            )
        return cases_deaths_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        cases_deaths_df = self.read_raw_data()
        # Filter out NA FIPS codes
        cases_deaths_df = cases_deaths_df[cases_deaths_df['fips'].notna()].copy()
        # Create FIPS state codes
        cases_deaths_df['FIPS_State'] = cases_deaths_df['fips'].apply(lambda x: x[:2])
        # Filter to specific State FIPS Codes
        if state_filter:
            cases_deaths_df = cases_deaths_df[cases_deaths_df['FIPS_State'].astype('int32').isin(state_filter)]
        # Drop State codes
        cases_deaths_df = cases_deaths_df.drop(['FIPS_State'], axis=1)
        # Formatting dates
        cases_deaths_df['date'] = cases_deaths_df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # Rename 'fips' for consistency
        cases_deaths_df = cases_deaths_df.rename(columns={'fips':'FIPS'})
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/cases_deaths_df.csv'
            cases_deaths_df.to_csv(export_path, index=False)
        # Return processed df
        return cases_deaths_df


class VaccinationsReader(Reader):
    def __init__(self, home_dir=''):
        data_path = 'data/vaccinations/COVID-19_Vaccinations_in_the_United_States_County.csv'
        super().__init__(home_dir, data_path)
        if not self.check_data_exists():
            print(f'Vaccinations data not found at {home_dir + data_path}. Please manually download the data. See data/README.md for more information.')
    
    def read_raw_data(self):
        vaccinations_df = pd.read_csv(
            self.home_dir + self.data_path
            , delimiter=','
            , dtype={
                'FIPS' : str
                }
            , usecols=[
                'Date'
                , 'FIPS'
                , 'Series_Complete_Pop_Pct'
                , 'Administered_Dose1_Pop_Pct' # Are these also counted in the above
                , 'Metro_status' # Will need to convert this into a boolean fields 
                , 'Series_Complete_Pop_Pct_UR_Equity' # Not sure how this is working, it's either null or a value 1-8
                ]
            )
        return vaccinations_df

    def read_and_process_data(self, state_filter=None, export=False, export_path=None):
        vaccinations_df = self.read_raw_data()
        # Filter out NA FIPS codes
        vaccinations_df = vaccinations_df[vaccinations_df['FIPS'] != 'UNK']
        # Create FIPS state codes
        vaccinations_df['FIPS_State'] = vaccinations_df['FIPS'].apply(lambda x: x[:2])
        # Filter to specific State FIPS Codes
        if state_filter:
            vaccinations_df = vaccinations_df[vaccinations_df['FIPS_State'].astype('int32').isin(state_filter)]
        # Drop State codes
        vaccinations_df = vaccinations_df.drop(['FIPS_State'], axis=1)
        # Set date
        vaccinations_df['Date'] = vaccinations_df['Date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
        # Drop duplicates
        vaccinations_df = vaccinations_df.drop_duplicates()
        # Changing Metro status to boolean
        vaccinations_df['Metro_status'] = vaccinations_df['Metro_status'].replace('Metro', 1).replace('Non-metro', 0)
        # Renaming to match cdc regs files
        vaccinations_df = vaccinations_df.rename(columns={'Date': 'date', 'Metro_status' :'is_metro'})
        # Export if desired:
        if export:
            if export_path is None:
                export_path = self.home_dir + 'data/transformed_data/vaccinations_df.csv'
            vaccinations_df.to_csv(export_path, index=False)
        # Return processed df
        return vaccinations_df
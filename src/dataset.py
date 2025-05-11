from enum import Enum
import os
import pandas as pd
import numpy as np
import collections.abc
from typing import Dict, Union, List, Tuple, Optional, cast
import math
import re

import torch
from torch.utils.data import Dataset
from torch_geometric.utils.smiles import from_smiles
from torch_geometric.data import Data
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from src.smiles_utils import to_canonical

from rdkit import Chem


class Purpose(Enum):
    Train = 0
    Val = 1
    Test = 2

class Normalizer:
    def __init__(self, data: np.ndarray):
        self.means = np.nanmean(data, axis=0)
        self.stddevs = np.nanstd(data, axis=0)

    def encode(self, row: np.ndarray) -> np.ndarray:
        return (row - self.means) / self.stddevs

    def encode_stddev(self, row: np.ndarray) -> np.ndarray:
        return row / self.stddevs

    def decode(self, row: np.ndarray) -> np.ndarray:
        return row * self.stddevs + self.means

    def decode_stddev(self, row: np.ndarray) -> np.ndarray:
        return row * self.stddevs


def parse_category(item: Union[str, float, None]) -> float:
    if isinstance(item, str):
        match = re.search(r'< (\d+(\.\d+)?)%', item)
        if match:
            result = float(match.group(1))
        else:
            result = np.nan
    else:
        result = np.nan
    return result


def stddev_serie_to_values(category_serie: pd.Series) -> np.ndarray:
    serie_perc = category_serie.apply(parse_category)
    # Set 100% uncertainty for values with missing category annotation
    serie_perc[serie_perc.isna()] = 100.0
    serie_frac = 0.01 * serie_perc
    return serie_frac.to_numpy()


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)


# Use max(0.1*mean, 10-th_percentile) to clip stddevs from the bottom
def adaptive_clip(vals: np.ndarray) -> np.ndarray:
    if np.isnan(vals).all():
        return vals
    else:
        masked = vals[~np.isnan(vals)]
        mean_v = np.mean(masked).item()
        perc_20 = np.percentile(masked, 20).item()
        thresh = max(0.2*mean_v, perc_20)
        rectified = vals.copy()
        rectified[rectified < thresh] = thresh
    return rectified


class DatasetProcessor(Dataset[Data], collections.abc.Sequence[Data]):

    property_full_names = {
        "MW": "Molecular Weight",
        "SSTD": "Standard Absolute Entropy",
        "TC": "Critical Temperature",
        "HFUS": "Heat of Fusion at Melting Point",
        "PC": "Critical Pressure",
        "HCOM": "Standard Net Heat of Combustion",
        "VC": "Critical Volume",
        "FP": "Flash Point",
        "ZC": "Critical Compressibility Factor",
        "FLVL": "Lower Flammability Limit Composition",
        "ACEN": "Acentric Factor",
        "FLVU": "Upper Flammability Limit Composition",
        "NBP": "Normal Boiling Point",
        "AIT": "Autoignition Temperature",
        "MP": "Melting Point",
        "RG": "Radius of Gyration",
        "TPT": "Triple Point Temperature",
        "SOLP": "Solubility Parameter",
        "TPP": "Triple Point Pressure",
        "DM": "Dipole Moment",
        "LVOL": "Liquid Molar Volume",
        "VDWV": "van der Waals Volume",
        "HFOR": "Ideal Gas Enthalpy of Formation",
        "VDWA": "van der Waals Area",
        "GFOR": "Ideal Gas Gibbs Energy of Formation",
        "RI": "Refractive Index",
        "ENT": "Ideal Gas Absolute Entropy",
        "HSUB": "Heat of Sublimation",
        "HSTD": "Standard Heat of Formation",
        "PAR": "Parachor",
        "GSTD": "Standard Gibbs Energy of Formation",
        "DC": "Dielectric Constant",
        "ACCW": "Infinite Dilution Activity Coefficient of the Chemical in Water",
        "HLC": "Henry's Law Constant in Water",
        "FLTL": "Lower Flammability Limit Temperature",
        "FLTU": "Upper Flammability Limit Temperature",
        "SOLW": "Solubility in Water",
        "LCP1": "Liquid Heat Capacity at -40ºC",
        "LCP2": "Liquid Heat Capacity at 25ºC",
        "LCP3": "Liquid Heat Capacity at 75ºC",
        "LCP4": "Liquid Heat Capacity at 100ºC",
        "LDN1": "Liquid Density at -20ºC",
        "LDN2": "Liquid Density at 15ºC",
        "LDN3": "Liquid Density at 20ºC",
        "LDN4": "Liquid Density at 60ºC",
        "LTC1": "Liquid Thermal Conductivity at 25ºC",
        "LTC2": "Liquid Thermal Conductivity at 50ºC",
        "LTC3": "Liquid Thermal Conductivity at 75ºC",
        "LTC4": "Liquid Thermal Conductivity at 100ºC",
        "LVS1": "Liquid Viscosity at -40ºC",
        "LVS2": "Liquid Viscosity at -20ºC",
        "LVS3": "Liquid Viscosity at 25ºC",
        "LVS4": "Liquid Viscosity at 40ºC",
        "ST1": "Surface Tension at 0ºC",
        "ST2": "Surface Tension at 20ºC",
        "ST3": "Surface Tension at 40ºC",
        "ST4": "Surface Tension at 60ºC",
        "VP1": "Vapor Pressure at -28ºC",
        "VP2": "Vapor Pressure at 12ºC",
        "VP3": "Vapor Pressure at 25ºC",
        "VP4": "Vapor Pressure at 38ºC",
        "VP5": "Vapor Pressure at 78ºC",
        "VP6": "Vapor Pressure at 200ºC",
        "CN": "Cetane Number"
    } 

    validity_condition = {
        "MW": ">0",
        "SSTD": ">0",
        "TC": ">0",
        "HFUS": ">0",
        "PC": ">0",
        "HCOM": "!=0", 
        "VC": ">0",
        "FP": ">0",
        "ZC": ">0",
        "FLVL": ">0",
        "ACEN": "!=0", 
        "FLVU": ">0",
        "NBP": ">0",
        "AIT": ">0",
        "MP": ">0",
        "RG": ">0",
        "TPT": ">0",
        "SOLP": ">0",
        "TPP": ">0",
        "DM": ">=0", 
        "LVOL": ">0",
        "VDWV": ">0",
        "HFOR": "!=0",
        "VDWA": ">0",
        "GFOR": "!=0", 
        "RI": ">0",
        "ENT": ">0",
        "HSUB": ">0",
        "HSTD": "!=0", 
        "PAR": ">0",
        "GSTD": "<0", 
        "DC": ">0",
        "ACCW": ">0",
        "HLC": ">0",
        "FLTL": ">0",
        "FLTU": ">0",
        "SOLW": ">0",
        "LCP1": ">0",
        "LCP2": ">0",
        "LCP3": ">0",
        "LCP4": ">0",
        "LDN1": ">0",
        "LDN2": ">0",
        "LDN3": ">0",
        "LDN4": ">0",
        "LTC1": ">0",
        "LTC2": ">0",
        "LTC3": ">0",
        "LTC4": ">0",
        "LVS1": ">0",
        "LVS2": ">0",
        "LVS3": ">0",
        "LVS4": ">0",
        "ST1": ">0",
        "ST2": ">0",
        "ST3": ">0",
        "ST4": ">0",
        "VP1": ">0",
        "VP2": ">0",
        "VP3": ">0",
        "VP4": ">0",
        "VP5": ">0",
        "VP6": ">0",
        "CN": ">0"
    }

    filter_percentiles = {
        "MW": (0, 1),
        "SSTD": (0, 1),
        "TC": (0, 1-0.005),
        "HFUS": (0, 1-0.005),
        "PC": (0, 1-0.002),
        "HCOM": (0, 1),
        "VC": (0, 1-0.002),
        "FP": (0, 1),
        "ZC": (0, 1-0.002),
        "FLVL": (0, 1),
        "ACEN": (0, 1),
        "FLVU": (0, 1),
        "NBP": (0, 1-0.005),
        "AIT": (0, 1-0.002),
        "MP": (0, 1-0.005),
        "RG": (0, 1-0.002),
        "TPT": (0, 1-0.002),
        "SOLP": (0, 1-0.005),
        "TPP": (0.002, 1-0.002),
        "DM": (0, 1-0.01),
        "LVOL": (0, 1-0.005),
        "VDWV": (0, 1),
        "HFOR": (0, 1),
        "VDWA": (0, 1),
        "GFOR": (0, 1),
        "RI": (0, 1-0.005),
        "ENT": (0, 1-0.008),
        "HSUB": (0, 1-0.005),
        "HSTD": (0, 1),
        "PAR": (0, 1-0.005),
        "GSTD": (0+0.005, 1),
        "DC": (0, 1-0.005),
        "ACCW": (0, 1-0.005),
        "HLC": (0, 1-0.005),
        "FLTL": (0, 1-0.005),
        "FLTU": (0, 1-0.005),
        "SOLW": (0, 1-0.005),
        "LCP1": (0, 1-0.005),
        "LCP2": (0, 1-0.005),
        "LCP3": (0, 1-0.005),
        "LCP4": (0, 1-0.005),
        "LDN1": (0, 1-0.005),
        "LDN2": (0, 1-0.005),
        "LDN3": (0, 1-0.005),
        "LDN4": (0, 1-0.005),
        "LTC1": (0, 1-0.005),
        "LTC2": (0, 1-0.005),
        "LTC3": (0, 1-0.005),
        "LTC4": (0, 1-0.005),
        "LVS1": (0, 1-0.005),
        "LVS2": (0, 1-0.005),
        "LVS3": (0, 1-0.005),
        "LVS4": (0, 1-0.005),
        "ST1": (0, 1-0.005),
        "ST2": (0, 1-0.005),
        "ST3": (0, 1-0.005),
        "ST4": (0, 1-0.005),
        "VP1": (0, 1-0.005),
        "VP2": (0, 1-0.005),
        "VP3": (0, 1-0.005),
        "VP4": (0, 1-0.005),
        "VP5": (0, 1-0.005),
        "VP6": (0, 1-0.005),
        "CN": (0, 1-0.005)
    }

    accurate_error_perc = {
        "MW": 5,
        "SSTD": 5,
        "TC": 5,
        "HFUS": 5,
        "PC": 5,
        "HCOM": 5,
        "VC": 5,
        "FP": 5,
        "ZC": 5,
        "FLVL": 5,
        "ACEN": 5,
        "FLVU": 5,
        "NBP": 5,
        "AIT": 5,
        "MP": 5,
        "RG": 5,
        "TPT": 5,
        "SOLP": 5,
        "TPP": 5,
        "DM": 5,
        "LVOL": 5,
        "VDWV": 5,
        "HFOR": 5,
        "VDWA": 5,
        "GFOR": 5,
        "RI": 5,
        "ENT": 5,
        "HSUB": 5,
        "HSTD": 5,
        "PAR": 5,
        "GSTD": 5,
        "DC": 5,
        "ACCW": 5,
        "HLC": 5,
        "FLTL": 5,
        "FLTU": 5,
        "SOLW": 5,
        "LCP1": 5,
        "LCP2": 5,
        "LCP3": 5,
        "LCP4": 5,
        "LDN1": 5,
        "LDN2": 5,
        "LDN3": 5,
        "LDN4": 5,
        "LTC1": 5,
        "LTC2": 5,
        "LTC3": 5,
        "LTC4": 5,
        "LVS1": 5,
        "LVS2": 5,
        "LVS3": 5,
        "LVS4": 5,
        "ST1": 5,
        "ST2": 5,
        "ST3": 5,
        "ST4": 5,
        "VP1": 5,
        "VP2": 5,
        "VP3": 5,
        "VP4": 5,
        "VP5": 5,
        "VP6": 5,
        "CN": 5
    }

    def benchmark_dataset_processor(self, df):
        # Add placeholder values for the benchmark non-SAF datasets
        df['ChemID'] = 0*len(df)
        df['Name'] = 'placeholder'*len(df)
        df['Formula'] = 'placeholder'*len(df)
        df['CASN'] = 'placeholder'*len(df)
        df['CNAM'] = 'placeholder'*len(df)
        df['INAM'] = 'placeholder'*len(df)
        df['Structure'] = 'placeholder'*len(df)
        df['Family'] = ['n-Alkanes'] * len(df)
        df['Sub_Family'] = 'placeholder'*len(df)
        df['STP State'] = ['L'] * len(df)
        df['MW'] = 0*len(df)
        if 'smiles' in df.columns:
            df = df.rename(columns={'smiles': 'SMILES'})

        column_order = ['ChemID', 'Name', 'Formula', 'CASN', 'CNAM', 'INAM', 
                        'SMILES', 'Structure', 'Family', 'Sub_Family', 'STP State', 'MW']
        new_columns = [col for col in df.columns if col not in column_order]
        for col in new_columns:
            df[f"{col} Error [%]"] = 1  
            df[f"{col} Data Type"] = "Experimental" 
            df[f"{col} Notes"] = None
            df[f"{col} Ref#"] = None

        df = df[column_order + sum(([col] + [f"{col}{suffix}" for suffix in [" Error [%]", " Data Type", " Notes", " Ref#"]] for col in new_columns), [])]

        return df


    def __init__(self,
                 path: str, 
                 tab: str,
                 task_type: str,
                 filters: Optional[str] = None,
                 cache_data: bool = False,
                 ablated_props: list = [],
            ) -> None:
        
        self.task_type = task_type
        
        if path.endswith('.csv'):
            df_orig = pd.read_csv(path)
        else:
            df_orig = pd.read_excel(path, tab)

        if path[5:8] != 'SAF':
            self.non_saf_dataset = True
            df_orig = self.benchmark_dataset_processor(df_orig)
            self.property_full_names = {col: col for col in df_orig.columns}
            print("Used dataset:", path)
        else:
            self.non_saf_dataset = False


        mol_weight_col = df_orig['MW']
        print("Ablated properties:", ablated_props)
        if ablated_props != 'None':
            if "LOW" in ablated_props:
                ablated_props = ["MW", "SSTD", "TC", "HFUS", "PC", "VC", "ZC", "ACEN", "RG", "TPT", "SOLP", "TPP", "DM", "LVOL", "VDWV",
                                 "HFOR", "VDWA", "GFOR", "RI", "ENT", "HSUB", "HSTD", "PAR", "GSTD", "ACCW", "HLC", "FLTL", "FLTU", "SOLW"]
            for prop in ablated_props[:]: 
                ablated_props.append(prop + ' ')
            df_orig = df_orig.loc[:, ~df_orig.columns.str.contains(r'\b(?:' + '|'.join(ablated_props) + r')\b', case=False)]
            if "MW" not in df_orig.columns:
                df_orig.insert(loc=11, column='MW', value=mol_weight_col)

        all_columns = list(df_orig.columns)
        feature_columns = list(df_orig.iloc[:, :12].columns)
        property_names = list(df_orig.iloc[:, 12::5].columns)
        clean_property_names = []

        series_smiles = df_orig['SMILES']
        series_formula = df_orig['Formula']
        series_family = df_orig['Family']
        series_state = df_orig['STP State']
        df_features = df_orig[feature_columns]
        df_properties = pd.DataFrame(index=df_orig.index)
        df_error_categories = pd.DataFrame(index=df_orig.index)
        for i_prop, raw_prop_name in enumerate(property_names):
            clean_name = raw_prop_name.strip()
            clean_property_names.append(clean_name)

            col = 12 + i_prop * 5
            prop = df_orig.iloc[:, col+0].copy()
            error_category = df_orig.iloc[:, col+1].copy()
            data_type = df_orig.iloc[:, col+2].copy()

            mask = data_type.isin(['Experimental', 'Defined', 'Derived'])
            prop[~mask] = None

            if not self.non_saf_dataset:
                condition = self.validity_condition[clean_name]
                assert condition in {">0", ">=0", "!=0", "<0"}
                eps = 1e-35
                if condition == ">0":
                    prop[prop <= eps] = None
                elif condition == ">=0":
                    prop[prop < 0] = None
                elif condition == "!=0":
                    prop[(np.abs(prop) <= eps)] = None
                elif condition == "<0":
                    prop[prop >= -eps] = None
                else:
                    assert False

            def mask_within_perc(arr: np.ndarray, bot_perc: float, top_perc: float) -> np.ndarray:
                argsorted = np.argsort(arr)
                srtd = arr[argsorted]
                nnans = np.isnan(arr).sum()
                n = len(argsorted)
                b = int(bot_perc * n)
                t = int(top_perc * n)
                selected_indices = argsorted[b:t-nnans]
                sn = np.isnan(arr[selected_indices]).sum()
                mask = np.zeros_like(arr, dtype=bool)
                mask[selected_indices] = True
                s = mask.sum()
                return mask

            top_bot_perc = self.filter_percentiles[clean_name] if not self.non_saf_dataset else (0, 1)
            is_within_perc = mask_within_perc(prop.to_numpy(), *top_bot_perc)
            nans_before = np.isnan(prop.to_numpy()).sum()
            prop[~is_within_perc] = None 
            nans_after = np.isnan(prop.to_numpy()).sum()           
            df_properties[clean_name] = prop

            error_category[error_category == 0] = None 
            error_category[error_category == "Unknown"] = "< 1%" 
            df_error_categories[clean_name] = error_category              
        

        self.smiles_to_graph = lambda smiles: from_smiles(smiles, with_hydrogen=False)


        indices_to_remove = []
        def worker(args):
            idx, smiles = args
            if isinstance(smiles, float) and math.isnan(smiles):
                indices_to_remove.append(idx)
                return
            assert isinstance(smiles, str)
            try:
                data_tmp = self.smiles_to_graph(smiles)
            except ValueError:
                indices_to_remove.append(idx)
                return
            if data_tmp is None:
                indices_to_remove.append(idx)
                return
            if data_tmp.num_nodes == 0:
                indices_to_remove.append(idx)
                return


        with ThreadPool(8) as pool:
            list(tqdm(pool.imap_unordered(worker, series_smiles.items()), total=len(series_smiles)))

        print(f'Faulty smiles treatment will remove '
            f'{len(indices_to_remove)} indices/molecules!')

        series_smiles = series_smiles.drop(indices_to_remove)
        df_properties = df_properties.drop(indices_to_remove)
        df_error_categories = df_error_categories.drop(indices_to_remove)
        df_features = df_features.drop(indices_to_remove)
        series_formula = series_formula.drop(indices_to_remove)
        series_family = series_family.drop(indices_to_remove)
        series_state = series_state.drop(indices_to_remove)

        filtered_indices = dataset_filter(filters, series_formula, series_smiles,
                                          series_family, series_state)

        print(f'Outlier treatment will remove '
            f'{len(filtered_indices)} indices/molecules!')

        series_smiles = series_smiles.drop(filtered_indices)
        df_properties = df_properties.drop(filtered_indices)
        df_error_categories = df_error_categories.drop(filtered_indices)
        df_features = df_features.drop(filtered_indices)
        series_formula = series_formula.drop(filtered_indices)
        series_family = series_family.drop(filtered_indices)
        series_state = series_state.drop(filtered_indices)
        self.series_family = series_family

        property_names = list(df_orig.iloc[:, 12::5].columns)
        for i_prop, raw_prop_name in enumerate(property_names):
            clean_name = raw_prop_name.strip()
            prop = df_properties[clean_name].copy()
            df_properties[clean_name] = prop

        df_property_stddevs = pd.DataFrame(index=df_properties.index)
        for prop_name in clean_property_names:
            prop_serie = df_properties[prop_name]
            prop_np = prop_serie.to_numpy()
            category_serie = df_error_categories[prop_name]
            stddev_frac_np = stddev_serie_to_values(category_serie)
            abs_prop_np = np.abs(prop_np)
            stddev_np = abs_prop_np * stddev_frac_np
            stddev_np = adaptive_clip(stddev_np)
            df_property_stddevs[prop_name] = stddev_np

        series_canonical_smiles = series_smiles.apply(to_canonical)
        record_limit = None # 10
        if record_limit is not None:
            series_smiles = series_smiles.iloc[:record_limit]
            series_canonical_smiles = series_canonical_smiles.iloc[:record_limit]
            df_features = df_features.iloc[:record_limit]
            df_properties = df_properties.iloc[:record_limit]
            df_error_categories = df_error_categories.iloc[:record_limit]
            df_property_stddevs = df_property_stddevs.iloc[:record_limit]

        self.series_smiles = series_smiles
        self.series_canonical_smiles = series_canonical_smiles
        self.df_features = df_features
        self.df_properties = df_properties
        self.df_error_categories = df_error_categories
        self.df_property_stddevs = df_property_stddevs

        props_np = self.df_properties.to_numpy()

        self.error_thresh_perc = np.array([self.accurate_error_perc[n]
                                           for n in clean_property_names],
                                           dtype=np.float32) if not self.non_saf_dataset else None

        self._normalizer = Normalizer(props_np)

        name = "dataset_classic"
        self.cache_path = f"data_cache/{name}_{str(filters)}.pt"
        cache_data=False
        self.cached_data: Optional[Dict[int, Data]]
        if cache_data:
            self._create_or_load_cache()
        else:
            self.cached_data = None

    @property
    def normalizer(self) -> Normalizer:
        return self._normalizer

    def _create_or_load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            self.cached_data = torch.load(self.cache_path)
        else:
            cached_data = {}
            for index in tqdm(range(len(self))):
                item = self._calc_item(index)
                cached_data[index] = item
            os.makedirs(os.path.split(self.cache_path)[0], exist_ok=True)
            torch.save(cached_data, self.cache_path)
            self.cached_data = cached_data

    def __getitem__(self, index: Union[int, slice]) -> Data:
        if self.cached_data:
            item = self._load_item(index)
        else:
            item = self._calc_item(index)
        return item

    def _load_item(self, index: Union[int, slice]) -> Data:
        assert isinstance(index, int)
        assert self.cached_data is not None
        assert index in self.cached_data
        return self.cached_data[index]

    def _calc_item(self, index: Union[int, slice]) -> Data:
        assert isinstance(index, int)
        smiles = self.series_smiles.iloc[index]
        canonical_smiles = self.series_canonical_smiles.iloc[index]
        df_row = self.df_features.iloc[index]
        df_error_category_row = self.df_error_categories.iloc[index]
        data = self.smiles_to_graph(smiles)
        data.df_row = df_row # inject extra info for outlier forensics
        data.df_error_category_row = df_error_category_row
        properties_raw = self.df_properties.iloc[[index]].to_numpy().squeeze(0)
        property_stddevs_raw = self.df_property_stddevs.iloc[[index]].to_numpy().squeeze(0)
        if self.task_type == 'classification':
            properties = properties_raw  
        else:
            properties = self._normalizer.encode(properties_raw)
        property_stddevs = self._normalizer.encode_stddev(property_stddevs_raw)
        data.properties = to_tensor(properties)
        data.property_stddevs = to_tensor(property_stddevs)
        return data

    def __len__(self) -> int:
        return len(self.series_smiles)

    def keep_precise_only(self, data: Data) -> Data:
        result = data.__copy__()
        error_value = data.df_error_category_row.apply(parse_category).to_numpy()
        good_mask = error_value <= self.error_thresh_perc
        properties = data.properties.clone().squeeze(0)
        properties[~good_mask] = np.nan
        result.properties = properties.unsqueeze(0)
        return result


class Split(Dataset[Data], collections.abc.Sequence[Data]):
    def __init__(self, dataset: DatasetProcessor, indices: np.ndarray, task_type: str,
                 purpose: Purpose, oversample_factor: int = 1000):
        self.dataset = dataset
        self.indices = indices
        self.purpose = purpose
        self.oversample_factor = oversample_factor
        self.task_type = task_type

    def __getitem__(self, index: Union[int, slice]) -> Data:
        assert isinstance(index, int)
        if self.purpose == Purpose.Train:
            index_rem = index % len(self.indices)
            idx = self.indices[index_rem].item()
        else:
            idx = self.indices[index].item()
        data = self.dataset[idx]
        if self.purpose == Purpose.Val or self.purpose == Purpose.Test:
            data = self.dataset.keep_precise_only(data) if (self.task_type=='regression') else data
        return data

    def __len__(self):
        if self.purpose == Purpose.Train:
            return len(self.indices) * self.oversample_factor
        else:
            return len(self.indices)

    def get_indices(self) -> np.ndarray:
        return self.indices

    def as_dataset(self) -> Dataset:
        return cast(Dataset, self)


def dataset_filter(filters: Optional[str], series_formula: pd.Series, series_smiles: pd.Series,
                   series_family: pd.Series, series_state: pd.Series) -> List[int]:
    
    # Set the default filter here
    if filters is None:
        filters = "f"

    lowercase_filters = filters.lower()
    filters_list = [char for char in lowercase_filters]

    by_family = 'f' in filters_list
    by_atoms = 'a' in filters_list
    by_chain_length = 'c' in filters_list
    by_state = 's' in filters_list

    filtered_indices = []
    permitted_atoms = ['C', 'H', 'O']

    if by_family:
        
        family_list = ['other polyfunctional c, h, o',
                            'silanes/siloxanes',
                            'sulfides/thiophenes',
                            'polyols',
                            'c, h, f compounds',
                            'polyfunctional c, h, o, halide',
                            'polyfunctional c, h, o, n',
                            'aliphatic ethers',
                            'other amines, imines',
                            'aromatic amines',
                            'c, h, multihalogen compounds',
                            'aromatic esters',
                            'polyfunctional amides/amines',
                            'other ethers/diethers',
                            'polyfunctional esters',
                            'inorganic halides',
                            'elements',
                            'organic/inorganic compounds',
                            'nitriles',
                            'c3 & higher aliphatic chlorides',
                            'inorganic gases',
                            'organic salts',
                            'other inorganic salts',
                            'sodium salts',
                            'c, h, no2 compounds',
                            'mercaptans',
                            'polyfunctional acids',
                            'c, h, br compounds',
                            'other aliphatic amines',
                            'c1/c2 aliphatic chlorides',
                            'other inorganics',
                            'aromatic chlorides',
                            'peroxides',
                            'dicarboxylic acids',
                            'glycerides',
                            'inorganic acids',
                            'formates',
                            'propionates and butyrates',
                            'aromatic carboxylic acids',
                            'cycloaliphatic alcohols',
                            'other condensed rings',
                            'other polyfunctional organics',
                            'polyfunctional c, h, o, s',
                            'n-aliphatic primary amines',
                            'polyfunctional c, h, n, halide, (o)',
                            'isocyanates/diisocyanates',
                            'anhydrides',
                            'c, h, i compounds',
                            'nitroamines',
                            'inorganic bases',
                            'polyfunctional nitriles',
                            'epoxides',
                            'n-aliphatic acids',
                            'other saturated aliphatic esters',
                            'other hydrocarbon rings',
                            'other aliphatic acids']

        for idx, family in series_family.items():
            if family.lower() in family_list:
                if idx not in filtered_indices:
                    filtered_indices.append(idx)

    if by_atoms:
        for idx, formula in series_formula.items():
            if isinstance(formula, float) and math.isnan(formula):
                filtered_indices.append(idx)
                continue
            assert isinstance(formula, str)
            atoms = re.findall(r'[A-Z][a-z]*', formula)
            if not all(atom in permitted_atoms for atom in atoms):
                filtered_indices.append(idx)

    if by_chain_length:
        min_length = 6
        max_length = 16
        for idx, smiles in series_smiles.items():
            mol = Chem.MolFromSmiles(smiles) # type: ignore
            chain_length = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            if (chain_length < min_length) or (chain_length > max_length):
                filtered_indices.append(idx)

    if by_state:
        for idx, state in series_state.items():
            if state != 'L':
                filtered_indices.append(idx)

    filtered_indices = list(set(filtered_indices))

    return filtered_indices

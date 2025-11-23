"""
Dataset Handler for Yatra Travel Agent
Handles loading and querying the Kaggle travel dataset using pandas
"""

import pandas as pd
import os
from typing import List, Dict, Any, Optional


class DatasetHandler:
    """Handles all pandas operations on the travel destinations CSV dataset"""

    def __init__(self, csv_path: str = None):
        """
        Initialize the dataset handler

        Args:
            csv_path: Path to the CSV file. Defaults to data/destinations.csv
        """
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'data', 'destinations.csv')

        self.csv_path = csv_path
        self.df = None
        self._load_dataset()

    def _load_dataset(self):
        """Load the CSV dataset into a pandas DataFrame"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✓ Loaded dataset with {len(self.df)} destinations")
            print(f"✓ Columns: {list(self.df.columns)}")
            self._preprocess_data()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. "
                "Please download the Kaggle dataset and place it in the data/ directory."
            )

    def _preprocess_data(self):
        """Preprocess and clean the dataset"""
        # Convert column names to lowercase and replace spaces with underscores
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')

        # Handle missing values
        self.df = self.df.fillna('')

        # Standardize text columns (strip whitespace)
        text_columns = self.df.select_dtypes(include=['object']).columns
        for col in text_columns:
            self.df[col] = self.df[col].str.strip()

    def get_column_names(self) -> List[str]:
        """Return list of column names in the dataset"""
        return list(self.df.columns)

    def filter_by_type(self, destination_type: str) -> pd.DataFrame:
        """
        Filter destinations by type (beach, mountain, heritage, wildlife)

        Args:
            destination_type: Type of destination to filter by

        Returns:
            Filtered DataFrame
        """
        # Flexible matching - checks common column names
        type_columns = ['type', 'category', 'destination_type', 'place_type']

        for col in type_columns:
            if col in self.df.columns:
                mask = self.df[col].str.lower().str.contains(
                    destination_type.lower(),
                    case=False,
                    na=False
                )
                return self.df[mask]

        # If no type column found, search in description or name
        if 'description' in self.df.columns:
            mask = self.df['description'].str.lower().str.contains(
                destination_type.lower(),
                case=False,
                na=False
            )
            return self.df[mask]

        return self.df

    def filter_by_budget(self,
                        max_budget: float,
                        budget_column: str = None) -> pd.DataFrame:
        """
        Filter destinations by budget

        Args:
            max_budget: Maximum budget in rupees
            budget_column: Name of budget column (auto-detected if None)

        Returns:
            Filtered DataFrame
        """
        if budget_column is None:
            # Try to find budget-related columns
            possible_columns = ['budget', 'cost', 'price', 'estimated_cost',
                              'average_cost', 'budget_per_person']

            for col in possible_columns:
                if col in self.df.columns:
                    budget_column = col
                    break

        if budget_column and budget_column in self.df.columns:
            # Convert to numeric, handling any non-numeric values
            budget_values = pd.to_numeric(self.df[budget_column], errors='coerce')
            mask = (budget_values <= max_budget) | budget_values.isna()
            return self.df[mask]

        # If no budget column found, return all
        return self.df

    def filter_by_duration(self,
                          max_hours: float,
                          duration_column: str = None) -> pd.DataFrame:
        """
        Filter destinations by recommended visit duration

        Args:
            max_hours: Maximum duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)
            duration_column: Name of duration column (auto-detected if None)

        Returns:
            Filtered DataFrame
        """
        if duration_column is None:
            # Try to find duration-related columns
            possible_columns = ['duration', 'recommended_duration', 'visit_duration',
                              'time_required', 'hours', 'days']

            for col in possible_columns:
                if col in self.df.columns:
                    duration_column = col
                    break

        if duration_column and duration_column in self.df.columns:
            # Convert to numeric (assuming hours, convert days if needed)
            duration_values = pd.to_numeric(self.df[duration_column], errors='coerce')

            # If column name suggests days, convert to hours
            if 'day' in duration_column.lower():
                duration_values = duration_values * 24

            mask = (duration_values <= max_hours) | duration_values.isna()
            return self.df[mask]

        return self.df

    def search_quantitative(self,
                          destination_type: Optional[str] = None,
                          max_budget: Optional[float] = None,
                          max_hours: Optional[float] = None) -> pd.DataFrame:
        """
        Perform combined quantitative search

        Args:
            destination_type: Type of destination
            max_budget: Maximum budget
            max_hours: Maximum duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)

        Returns:
            Filtered DataFrame matching all criteria
        """
        result = self.df.copy()

        if destination_type:
            result = self.filter_by_type(destination_type)

        if max_budget:
            result = pd.DataFrame(result)  # Ensure it's a DataFrame
            handler_temp = DatasetHandler.__new__(DatasetHandler)
            handler_temp.df = result
            result = handler_temp.filter_by_budget(max_budget)

        if max_hours:
            result = pd.DataFrame(result)
            handler_temp = DatasetHandler.__new__(DatasetHandler)
            handler_temp.df = result
            result = handler_temp.filter_by_duration(max_hours)

        return result

    def get_destination_details(self, destination_name: str) -> Dict[str, Any]:
        """
        Get full details for a specific destination

        Args:
            destination_name: Name of the destination

        Returns:
            Dictionary with destination details
        """
        # Try to find by name
        name_columns = ['name', 'place', 'destination', 'location', 'place_name']

        for col in name_columns:
            if col in self.df.columns:
                mask = self.df[col].str.lower() == destination_name.lower()
                matches = self.df[mask]

                if not matches.empty:
                    return matches.iloc[0].to_dict()

        return {}

    def get_random_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get random sample of destinations

        Args:
            n: Number of samples to return

        Returns:
            DataFrame with random samples
        """
        return self.df.sample(n=min(n, len(self.df)))

    def to_dict_list(self, df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to list of dictionaries

        Args:
            df: DataFrame to convert (uses self.df if None)

        Returns:
            List of dictionaries
        """
        if df is None:
            df = self.df

        return df.to_dict('records')

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset"""
        stats = {
            'total_destinations': len(self.df),
            'columns': list(self.df.columns),
        }

        # Count by type if available
        type_columns = ['type', 'category', 'destination_type']
        for col in type_columns:
            if col in self.df.columns:
                stats['destinations_by_type'] = self.df[col].value_counts().to_dict()
                break

        return stats

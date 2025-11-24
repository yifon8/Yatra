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
            possible_columns = ['entrance_fee_in_inr', 'budget', 'cost', 'price',
                              'estimated_cost', 'average_cost', 'budget_per_person']

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
                              'time_required', 'hours', 'days', 'time_needed_to_visit_in_hrs']

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

    def filter_by_rating(self, min_rating: float = 4.0) -> pd.DataFrame:
        """
        Filter destinations by rating - keeps destinations with null rating OR rating > min_rating

        Args:
            min_rating: Minimum rating threshold (default 4.0 out of 5.0)

        Returns:
            Filtered DataFrame
        """
        # Try to find rating-related columns
        possible_columns = ['rating', 'google_review_rating', 'review_rating',
                          'user_rating', 'average_rating']

        rating_column = None
        for col in possible_columns:
            if col in self.df.columns:
                rating_column = col
                break

        if rating_column:
            # Convert to numeric
            rating_values = pd.to_numeric(self.df[rating_column], errors='coerce')

            # Keep destinations with null rating OR rating > min_rating
            mask = rating_values.isna() | (rating_values > min_rating)
            return self.df[mask]

        # If no rating column found, return all destinations
        return self.df

    def filter_by_family_friendly(self, has_small_child: bool = False) -> pd.DataFrame:
        """
        Filter destinations for family-friendliness based on type and significance

        Args:
            has_small_child: If True, apply stricter filtering for small children

        Returns:
            Filtered DataFrame with family-friendly destinations
        """
        # Family-friendly destination types
        family_friendly_types = [
            'park', 'botanical garden', 'botanical', 'garden', 'zoo',
            'museum', 'science', 'aquarium', 'amusement park',
            'theme park', 'lake', 'beach', 'palace', 'fort',
            'temple', 'church', 'gurudwara', 'monument', 'memorial',
            'waterfall', 'valley', 'viewpoint', 'scenic', 'nature',
            'island', 'mall', 'shopping'
        ]

        # Less suitable for small children (stricter filter)
        if has_small_child:
            exclude_for_small_children = [
                'trekking', 'trek', 'adventure sport', 'paragliding',
                'ski resort', 'cricket ground', 'race track',
                'national park', 'wildlife sanctuary', 'hill'
            ]
        else:
            exclude_for_small_children = []

        # Check if type column exists
        type_column = None
        possible_type_columns = ['type', 'category', 'destination_type', 'place_type']

        for col in possible_type_columns:
            if col in self.df.columns:
                type_column = col
                break

        if type_column:
            # Convert type to lowercase for matching
            type_lower = self.df[type_column].str.lower()

            # Include family-friendly types
            family_mask = type_lower.str.contains('|'.join(family_friendly_types),
                                                  case=False, na=False, regex=True)

            # Exclude types not suitable for small children if specified
            if exclude_for_small_children:
                exclude_mask = type_lower.str.contains('|'.join(exclude_for_small_children),
                                                       case=False, na=False, regex=True)
                family_mask = family_mask & ~exclude_mask

            return self.df[family_mask]

        # If no type column found, return all destinations
        return self.df

    def search_quantitative(self,
                          destination_type: Optional[str] = None,
                          max_budget: Optional[float] = None,
                          max_hours: Optional[float] = None,
                          has_small_child: bool = False) -> pd.DataFrame:
        """
        Perform combined quantitative search with system-level filters

        IMPORTANT: This method ALWAYS applies system-level filters:
        - Rating must be null or > 4.0
        - Destinations must be family-friendly

        Args:
            destination_type: Type of destination
            max_budget: Maximum budget
            max_hours: Maximum duration in hours (can be decimal, e.g., 0.5, 1.5, 8.25)
            has_small_child: If True, apply stricter family-friendly filtering

        Returns:
            Filtered DataFrame matching all criteria
        """
        result = self.df.copy()

        # SYSTEM-LEVEL FILTERS (always applied)
        # 1. Filter by rating (null or > 4.0)
        result = self._apply_rating_filter_on_df(result, min_rating=4.0)

        # 2. Filter for family-friendly destinations
        result = self._apply_family_filter_on_df(result, has_small_child=has_small_child)

        # USER-SPECIFIED FILTERS (optional)
        # 3. Filter by destination type if specified
        if destination_type:
            result = self._apply_type_filter_on_df(result, destination_type)

        # 4. Filter by budget if specified
        if max_budget is not None:
            result = self._apply_budget_filter_on_df(result, max_budget)

        # 5. Filter by duration if specified
        if max_hours is not None:
            result = self._apply_duration_filter_on_df(result, max_hours)

        return result

    def _apply_rating_filter_on_df(self, df: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
        """Apply rating filter directly on a DataFrame"""
        possible_columns = ['rating', 'google_review_rating', 'review_rating',
                          'user_rating', 'average_rating']

        rating_column = None
        for col in possible_columns:
            if col in df.columns:
                rating_column = col
                break

        if rating_column:
            rating_values = pd.to_numeric(df[rating_column], errors='coerce')
            mask = rating_values.isna() | (rating_values > min_rating)
            return df[mask].copy()

        return df.copy()

    def _apply_family_filter_on_df(self, df: pd.DataFrame, has_small_child: bool = False) -> pd.DataFrame:
        """Apply family-friendly filter directly on a DataFrame"""
        family_friendly_types = [
            'park', 'botanical garden', 'botanical', 'garden', 'zoo',
            'museum', 'science', 'aquarium', 'amusement park',
            'theme park', 'lake', 'beach', 'palace', 'fort',
            'temple', 'church', 'gurudwara', 'monument', 'memorial',
            'waterfall', 'valley', 'viewpoint', 'scenic', 'nature',
            'island', 'mall', 'shopping'
        ]

        if has_small_child:
            exclude_for_small_children = [
                'trekking', 'trek', 'adventure sport', 'paragliding',
                'ski resort', 'cricket ground', 'race track',
                'national park', 'wildlife sanctuary', 'hill'
            ]
        else:
            exclude_for_small_children = []

        type_column = None
        possible_type_columns = ['type', 'category', 'destination_type', 'place_type']

        for col in possible_type_columns:
            if col in df.columns:
                type_column = col
                break

        if type_column:
            type_lower = df[type_column].str.lower()
            family_mask = type_lower.str.contains('|'.join(family_friendly_types),
                                                  case=False, na=False, regex=True)

            if exclude_for_small_children:
                exclude_mask = type_lower.str.contains('|'.join(exclude_for_small_children),
                                                       case=False, na=False, regex=True)
                family_mask = family_mask & ~exclude_mask

            return df[family_mask].copy()

        return df.copy()

    def _apply_type_filter_on_df(self, df: pd.DataFrame, destination_type: str) -> pd.DataFrame:
        """Apply destination type filter directly on a DataFrame"""
        type_columns = ['type', 'category', 'destination_type', 'place_type']

        for col in type_columns:
            if col in df.columns:
                mask = df[col].str.lower().str.contains(
                    destination_type.lower(),
                    case=False,
                    na=False
                )
                return df[mask].copy()

        if 'description' in df.columns:
            mask = df['description'].str.lower().str.contains(
                destination_type.lower(),
                case=False,
                na=False
            )
            return df[mask].copy()

        return df.copy()

    def _apply_budget_filter_on_df(self, df: pd.DataFrame, max_budget: float) -> pd.DataFrame:
        """Apply budget filter directly on a DataFrame"""
        possible_columns = ['entrance_fee_in_inr', 'budget', 'cost', 'price',
                          'estimated_cost', 'average_cost', 'budget_per_person']

        budget_column = None
        for col in possible_columns:
            if col in df.columns:
                budget_column = col
                break

        if budget_column:
            budget_values = pd.to_numeric(df[budget_column], errors='coerce')
            # Only include destinations where entrance fee is known AND within budget
            mask = budget_values <= max_budget
            return df[mask].copy()

        return df.copy()

    def _apply_duration_filter_on_df(self, df: pd.DataFrame, max_hours: float) -> pd.DataFrame:
        """Apply duration filter directly on a DataFrame"""
        possible_columns = ['duration', 'recommended_duration', 'visit_duration',
                          'time_required', 'hours', 'days', 'time_needed_to_visit_in_hrs']

        duration_column = None
        for col in possible_columns:
            if col in df.columns:
                duration_column = col
                break

        if duration_column:
            duration_values = pd.to_numeric(df[duration_column], errors='coerce')

            if 'day' in duration_column.lower():
                duration_values = duration_values * 24

            mask = (duration_values <= max_hours) | duration_values.isna()
            return df[mask].copy()

        return df.copy()

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

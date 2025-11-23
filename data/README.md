# Dataset Directory

This directory should contain the Kaggle travel dataset.

## Required File

Place the downloaded dataset here as:
```
destinations.csv
```

## How to Download

1. Visit: https://www.kaggle.com/datasets/saketk511/travel-dataset-guide-to-indias-must-see-places
2. Click "Download" (requires Kaggle account)
3. Extract the CSV file
4. Rename it to `destinations.csv` if needed
5. Place it in this directory

## Expected CSV Structure

The agent expects columns related to Indian tourist destinations. Common columns might include:

- **name** / **place**: Name of the destination
- **type** / **category**: Type (beach, mountain, heritage, wildlife)
- **description**: Detailed description of the place
- **budget** / **cost** / **estimated_cost**: Budget information
- **duration** / **hours** / **recommended_duration**: Visit duration
- **state** / **location**: Geographic location
- **activities**: Things to do
- **amenities**: Facilities available
- **best_time**: Best time to visit
- **rating**: User ratings

The `DatasetHandler` class will automatically adapt to different column naming conventions.

## Custom Datasets

You can use your own dataset as long as it contains destination information.

Minimum recommended columns:
- Name/Place identifier
- Some description text (for qualitative LLM analysis)
- Optional: budget, duration, type/category fields

To use a custom dataset:
```bash
python run_agent.py -i --csv path/to/your/dataset.csv
```

## File Not Found?

If you see "Dataset not found" error:
1. Make sure the file is named `destinations.csv`
2. Make sure it's in the `data/` directory
3. Check the file path is correct

Example:
```
Yatra/
└── data/
    └── destinations.csv  ← File should be here
```

"""
Logging utilities for AA operational processing chain.

This module provides comprehensive debug logging for array operations,
dimensions tracking, and data flow analysis throughout the processing pipeline.
"""

import logging
import os
import numpy as np
import pandas as pd
import xarray as xr


def setup_aa_logging():
    """Setup AA logging with environment-controlled debug level"""
    # Set level based on environment variable
    debug_enabled = os.getenv('AA_DEBUG') == '1'
    
    # Configure the root logger at INFO level to suppress third-party debug messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        force=True
    )
    
    # Set DEBUG level only for AA modules if debug is enabled
    if debug_enabled:
        # Set debug level for all AA modules
        aa_logger = logging.getLogger('aa_operational')
        aa_logger.setLevel(logging.DEBUG)
        
        # Also set debug for any other AA-related loggers you might use
        logging.getLogger('AA').setLevel(logging.DEBUG)
        
        # Explicitly suppress debug messages from common third-party libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('fsspec').setLevel(logging.WARNING)
        logging.getLogger('s3fs').setLevel(logging.WARNING)
        logging.getLogger('dask').setLevel(logging.WARNING)
        logging.getLogger('xarray').setLevel(logging.WARNING)
        logging.getLogger('pandas').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('rasterio').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        
    # Return the AA-specific logger
    return logging.getLogger('aa_operational')


def log_array_info(logger, array_name, data, sample_values=True):
    """
    Log comprehensive array information including dimensions and sample values.
    
    Args:
        logger: Logger instance to use for output
        array_name: Descriptive name for the array being logged
        data: Array-like data (xarray Dataset/DataArray, numpy array, or pandas DataFrame)
        sample_values: Whether to include sample values in the log output
    """
    if isinstance(data, xr.Dataset):
        logger.debug(f"{array_name} Dataset - Variables: {list(data.data_vars)}")
        logger.debug(f"{array_name} Dataset - Coordinates: {list(data.coords)}")
        logger.debug(f"{array_name} Dataset - Dimensions: {dict(data.dims)}")
        
        # Log each data variable
        for var_name, var_data in data.data_vars.items():
            log_array_info(logger, f"{array_name}.{var_name}", var_data, sample_values)
            
    elif isinstance(data, xr.DataArray):
        logger.debug(f"{array_name} DataArray - Shape: {data.shape}, Dims: {data.dims}")
        logger.debug(f"{array_name} - Coordinates: {list(data.coords)}")
        logger.debug(f"{array_name} - Data type: {data.dtype}")
        
        # Memory usage
        memory_mb = data.nbytes / (1024**2)
        logger.debug(f"{array_name} - Memory usage: {memory_mb:.2f} MB")
        
        # Coordinate ranges
        for coord_name, coord_data in data.coords.items():
            if coord_data.size > 1:
                if np.issubdtype(coord_data.dtype, np.datetime64):
                    logger.debug(f"{array_name} - {coord_name} range: {coord_data.min().values} to {coord_data.max().values}")
                elif np.issubdtype(coord_data.dtype, np.number):
                    logger.debug(f"{array_name} - {coord_name} range: {float(coord_data.min()):.4f} to {float(coord_data.max()):.4f}")
        
        # Statistical summary - only compute for materialized arrays (not Dask arrays)
        if data.size > 0:
            # Check if data is a Dask array (lazy) or materialized
            if hasattr(data.data, 'chunks'):
                # This is a Dask array - don't compute statistics to avoid triggering computation
                logger.debug(f"{array_name} - Dask array detected - skipping statistical computation")
            else:
                # This is a materialized array - safe to compute statistics
                valid_data = data.where(~np.isnan(data), drop=True)
                if valid_data.size > 0:
                    logger.debug(f"{array_name} - Stats: min={float(valid_data.min()):.4f}, "
                               f"max={float(valid_data.max()):.4f}, "
                               f"mean={float(valid_data.mean()):.4f}, "
                               f"std={float(valid_data.std()):.4f}")
                    logger.debug(f"{array_name} - Valid values: {valid_data.size}/{data.size} "
                               f"({100*valid_data.size/data.size:.1f}%)")
                    
                    # Sample values from different parts of the array
                    if sample_values and valid_data.size > 10:
                        sample_size = min(25, valid_data.size)
                        # Sample along the first dimension
                        first_dim = valid_data.dims[0]
                        indices = np.linspace(0, valid_data.sizes[first_dim]-1, sample_size, dtype=int)
                        samples = valid_data.isel({first_dim: indices})
                        logger.debug(f"{array_name} - Sample values: {samples.values}")
                else:
                    logger.warning(f"{array_name} - All values are NaN!")
        else:
            logger.warning(f"{array_name} - Array is empty!")
    
    elif isinstance(data, np.ndarray):
        logger.debug(f"{array_name} Array - Shape: {data.shape}, dtype: {data.dtype}")
        memory_mb = data.nbytes / (1024**2)
        logger.debug(f"{array_name} - Memory usage: {memory_mb:.2f} MB")
        
        if data.size > 0:
            if np.issubdtype(data.dtype, np.floating):
                valid_mask = ~np.isnan(data)
                valid_data = data[valid_mask]
                logger.debug(f"{array_name} - Valid values: {valid_data.size}/{data.size} "
                           f"({100*valid_data.size/data.size:.1f}%)")
            else:
                valid_data = data.flatten()
                
            if valid_data.size > 0:
                logger.debug(f"{array_name} - Stats: min={np.min(valid_data):.4f}, "
                           f"max={np.max(valid_data):.4f}, mean={np.mean(valid_data):.4f}")
                
                if sample_values and valid_data.size > 10:
                    sample_size = min(25, valid_data.size)
                    indices = np.linspace(0, valid_data.size-1, sample_size, dtype=int)
                    samples = valid_data[indices]
                    logger.debug(f"{array_name} - Sample values: {samples}")
            else:
                logger.warning(f"{array_name} - No valid values!")
    
    elif isinstance(data, pd.DataFrame):
        logger.debug(f"{array_name} DataFrame - Shape: {data.shape}")
        logger.debug(f"{array_name} - Columns: {list(data.columns)}")
        logger.debug(f"{array_name} - Index: {data.index.name if data.index.name else 'unnamed'}")
        logger.debug(f"{array_name} - Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Log numeric column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.debug(f"{array_name} - Numeric columns: {list(numeric_cols)}")
            for col in numeric_cols:
                valid_count = data[col].notna().sum()
                if valid_count > 0:
                    logger.debug(f"{array_name}.{col} - Valid: {valid_count}/{len(data)}, "
                               f"range: {data[col].min():.4f} to {data[col].max():.4f}")
        
        # Log categorical column info
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            logger.debug(f"{array_name} - Categorical columns: {list(categorical_cols)}")
            for col in categorical_cols:
                unique_count = data[col].nunique()
                logger.debug(f"{array_name}.{col} - Unique values: {unique_count}")
                if unique_count <= 10:  # Log unique values if not too many
                    logger.debug(f"{array_name}.{col} - Values: {list(data[col].unique())}")
    
    else:
        logger.debug(f"{array_name} - Type: {type(data)}, Value: {data}")


def log_processing_step(logger, step_name, input_data=None, output_data=None, **kwargs):
    """
    Log a processing step with before/after array information.
    
    Args:
        logger: Logger instance
        step_name: Name of the processing step
        input_data: Input data to log (optional)
        output_data: Output data to log (optional)
        **kwargs: Additional parameters to log
    """
    logger.info(f"=== {step_name} ===")
    
    # Log additional parameters
    if kwargs:
        for key, value in kwargs.items():
            logger.debug(f"{step_name} - {key}: {value}")
    
    # Log input data
    if input_data is not None:
        if isinstance(input_data, dict):
            for name, data in input_data.items():
                log_array_info(logger, f"Input_{name}", data)
        else:
            log_array_info(logger, "Input", input_data)
    
    # Log output data
    if output_data is not None:
        if isinstance(output_data, dict):
            for name, data in output_data.items():
                log_array_info(logger, f"Output_{name}", data)
        else:
            log_array_info(logger, "Output", output_data)


def log_data_loading_step(logger, data_source, data_path, data, load_method="unknown"):
    """
    Log data loading step with source and result information.
    
    Args:
        logger: Logger instance
        data_source: Description of data source (e.g., "ECMWF forecasts", "CHIRPS observations")
        data_path: Path or identifier for the data
        data: Loaded data
        load_method: Method used for loading (e.g., "zarr cache", "source API")
    """
    logger.info(f"=== Loading {data_source} ===")
    logger.debug(f"Data path: {data_path}")
    logger.debug(f"Load method: {load_method}")
    
    log_array_info(logger, data_source.replace(" ", "_"), data)
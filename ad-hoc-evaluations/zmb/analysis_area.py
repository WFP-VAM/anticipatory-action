import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from odc.geo import SomeResolution
from odc.geo.crs import CRS, norm_crs
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, box
from odc.geo.types import Unset

from hip.analysis.aoi.constants import AREA_CONFIG
from hip.analysis.aoi.utils import normalize_geobox
from hip.analysis.data._datasources import (
    _inspect_hashes,
    datasources,
    get_admin_boundaries,
    load_dataset,
)
from hip.analysis.ops._dataframe import join_gdf
from _vector import rasterize_gdf
from hip.analysis.ops._zonal import zonal_stats

logging.basicConfig(level="INFO")


@dataclass
class AnalysisArea:
    datetime_range: str
    bbox: tuple | None = None
    hi_res: bool = False
    crs: CRS = Unset()
    resolution: Optional[SomeResolution] = None
    geobox: Optional[GeoBox] = None
    datasets: dict = field(default_factory=lambda: defaultdict(dict))
    satellite_config: dict | None = None
    BASE_AREA_DATASET: str = "base_area_dataset"
    area_config: dict | None = None

    """
    Main object defining the user's relevant analysis context
    """

    def __post_init__(self):
        # Set area config from defaults if not provided
        if self.area_config is None:
            self.area_config = AREA_CONFIG

        if self.geobox is not None and any(
            [self.bbox, not isinstance(self.crs, Unset), self.resolution]
        ):
            raise ValueError(
                "Cannot provide geobox with any of bbox, crs or resolution"
            )

        if self.geobox is None and self.bbox is None:
            raise ValueError("Must provide either geobox or bbox")

        if self.geobox is None:
            if isinstance(self.crs, Unset):
                # Switch between relevant UTM or EPSG:4326 depending on hi_res flag
                self.crs = (
                    norm_crs("utm", box(*self.bbox, crs="epsg:4326"))
                    if self.hi_res
                    else CRS("epsg:4326")
                )

            if self.resolution is None:
                # Assign reasonable default based on projected CRSs typically using m or ft
                # while geographic CRSs use typically degrees
                self.resolution = 30 if self.crs.projected else 0.05

            if self.crs != CRS("epsg:4326"):
                # Transform EPSG:4326 bbox to user crs
                output_bbox = (
                    BoundingBox(*self.bbox, crs="epsg:4326").to_crs(self.crs).bbox
                )
            else:
                output_bbox = self.bbox

            # Determine GeoBox from populated parameters
            self.geobox = GeoBox.from_bbox(
                output_bbox, crs=self.crs, resolution=self.resolution
            )

        else:
            self.crs = self.geobox.crs
            self.resolution = self.geobox.resolution

        # Normalize the GeoBox to ensure consistent pickle representation
        self.geobox = normalize_geobox(self.geobox)

    @classmethod
    def from_admin_boundaries(
        cls,
        iso3: str,
        admin_level: int = 1,
        datetime_range=None,
        **kwargs,
    ):
        adm_geo = get_admin_boundaries(iso3=iso3, admin_level=admin_level)

        if "hi_res" in kwargs and kwargs["hi_res"] == True:
            logging.warning(
                "You are setting up a country-level AnalysisArea with the hi_res flag set to True. "
                "This will lead to handling very large gridded datasets, and will very likely lead to computational issues. "
                "We recommend setting hi_res to False."
            )

        return cls.from_geodataframe(
            areas=adm_geo,
            index_col="Name",
            datetime_range=datetime_range,
            **kwargs,
        )

    @classmethod
    def from_geodataframe(
        cls,
        areas: gpd.GeoDataFrame,
        index_col: str | None = None,
        datetime_range=None,
        dataset_name=None,
        **kwargs,
    ):
        # Get bbox from the areas geodataframe
        # TODO this assumes CRS 4326. Add validation check?
        bbox = tuple(areas.total_bounds)

        if index_col is not None:
            areas = areas.set_index(index_col, drop=False)

        area = cls(bbox=bbox, datetime_range=datetime_range, **kwargs)

        if dataset_name is None:
            dataset_name = AnalysisArea.BASE_AREA_DATASET
        area.add_dataset(areas, [dataset_name])

        return area

    def load_dataset(self, datasource, bbox, geobox, dates, load_config, extra_config):
        # Resolve bbox or geobox input --> geobox
        if bbox is None and geobox is None:
            # Use area geobox and derive consistent bbox
            geobox = self.geobox
        elif bbox is not None and geobox is None:
            # Transform provided bbox in Geobox with area crs and resolution
            target_crs_bbox = BoundingBox(*bbox, crs="epsg:4326").to_crs(self.crs)
            geobox = GeoBox.from_bbox(target_crs_bbox, resolution=self.resolution)
        else:
            logging.warning(
                "Loading dataset with both geobox and bbox provided. Geobox will take precedence!"
            )

        # Normalize the GeoBox to ensure consistent pickle representation and caching
        geobox = normalize_geobox(geobox)

        if dates is None:
            dates = self.datetime_range

        _inspect_hashes(datasource, geobox, dates, self.area_config, load_config)

        return load_dataset(
            datasource, geobox, dates, self.area_config, load_config, extra_config
        )

    def get_dataset(
        self,
        key: list,
        bbox=None,
        geobox=None,
        dates=None,
        load_config: dict = None,
        extra_config: dict = None,
        persist: bool = None,
    ):
        """
        Load or retrieve a dataset for the analysis area.

        This method first checks if the dataset is already loaded in the analysis area's
        dataset cache. If not found, it loads the dataset from the configured datasource
        using the specified parameters.

        Args:
            key: List identifying the dataset (e.g., ['chirps', 'rainfall']).
            bbox: Bounding box to load data for. If None, uses the analysis area's bounding box.
            geobox: GeoBox to load data for. Takes precedence over bbox if both provided.
            dates: Date range string (e.g., '2020-01-01/2020-12-31'). If None, uses
                   the analysis area's datetime_range.
            load_config: Dictionary with loading configuration options. Key options include:
                        - 'gridded_load_kwargs' (dict): Parameters for data loading with keys:
                          * 'chunks' (dict): Dask chunking config (e.g., {'time': -1, 'x': 100})
                          * 'resampling' (str): Resampling method (all options from rasterio.enums.Resampling are valid)
                          * 'preprocess' (callable): Function to preprocess datasets before concatenation
            extra_config: Dictionary with additional configuration options that are ignored when caching the function
                          call. As such, these options should not affect the output dataset. Key options include:
                        - 'persist_lineage' (bool): Whether to persist intermediate datasets
            persist: Whether to persist the dataset. If None, uses the datasource default.

        Returns:
            xarray.Dataset or xarray.DataArray: The loaded dataset.
        """

        dataset = self.lookup_dataset(key, bbox, geobox, dates)

        if dataset is None:
            # Dataset is not found among those added by user
            # Look up connector from connector configs
            datasource = datasources.get_datasource(key)
            # Ensure config exists
            if load_config is None:
                load_config = {}
            if extra_config is None:
                extra_config = {}
            # Enforce user persist flag if provided, or datasource default, or otherwise False
            load_config["persist"] = (
                persist
                if persist is not None
                else getattr(datasource, "persist", False)
            )
            delattr(datasource, "description")
            delattr(datasource, "persist")
            # Then use load dataset
            dataset = self.load_dataset(
                datasource, bbox, geobox, dates, load_config, extra_config
            )

        return dataset

    def add_dataset(self, data_object, key, bbox=None, geobox=None, dates=None):
        extended_key = key.copy()
        extended_key.extend([bbox, geobox, dates])
        # Traverse the .datasets dict to key and add data_object as value
        # Creates dictionary key path if it does not exist.
        d = self.datasets
        for k in extended_key[:-1]:
            if d.get(k) is None:
                d[k] = {}
            d = d[k]
        d[extended_key[-1]] = data_object

    def lookup_dataset(self, key, bbox=None, geobox=None, dates=None):
        extended_key = key.copy()
        extended_key.extend([bbox, geobox, dates])
        # Traverse the .datasets dict to key
        d = self.datasets
        for k in extended_key[:-1]:
            d = d.get(k)
            if d is None:
                return
        return d.get(extended_key[-1])

    def get_admin_boundaries(
        self, iso3: str, admin_level: int = 1, rasterize: bool = False
    ):
        admin_bounds = get_admin_boundaries(
            geobox=self.geobox, iso3=iso3, admin_level=admin_level, rasterize=rasterize
        )

        return admin_bounds

    def get_data_by_area(
        self,
        data,
        area_key: str = None,
        selected_area_name: str = None,
        all_touched: bool = True,
    ):
        """
        # TODO Check if this function used by any downstream code. If not, delete.
        Function to clip data to a specific area.

        Args:
            data:               The xarray object to clip
            area_key:           The key of the main area dataset to load
            selected_area_name: Sub-area of the main area to restrain the clipping to.
                                It should be one of the main area indices.
            all_touched:        Wether to keep all pixels touching the area's geometry or not.
                                This will be passed in to rioxarray's rio.clip()

        Returns:
            data_clipped: data clipped to the specified area
        """

        if area_key is None:
            area_key = [self.BASE_AREA_DATASET]

        area = self.get_dataset(area_key)

        if selected_area_name is not None:
            if selected_area_name not in area.index:
                raise ValueError(
                    f"selected_area_name should be a sub-area of your area {area_key}, ie. one of {area.index}."
                )
            area = area[area.index == selected_area_name]

        data_clipped = data.rio.clip(
            area.geometry.values,
            area.crs,
            all_touched=all_touched,
        )

        return data_clipped

    def zonal_stats(
        self,
        data: list | xr.DataArray,
        zones: list | xr.DataArray | gpd.GeoDataFrame = None,
        zone_ids: list | np.ndarray = None,
        stats: list = None,
        return_xarray: bool = False,
        all_touched: bool = False,
    ) -> pd.DataFrame | xr.DataArray:
        """
        Calculate zonal statistics over raster data for a set of zones.

        This function supports zones provided as:
        - an ``xarray.DataArray`` of zone labels (already rasterized), or
        - a ``geopandas.GeoDataFrame`` of vector geometries, which will be
            rasterized internally to match the grid of ``data``.

        Rasterization behavior
        ----------------------
        When `zones` is a GeoDataFrame, it is rasterized to the pixel grid of `data`
        using `rasterio.features.rasterize` semantics.

        - all_touched=False (default):
            Only pixels whose *center* falls inside a polygon are assigned that polygon’s ID.

        - all_touched=True:
            Any pixel that is touched by a polygon (edges or corners) is considered inside
            and will be assigned to that polygon.
            This produces a more inclusive (larger) set of zone pixels.

        Overlapping zones
        -----------------
        If multiple polygons touch or overlap the same pixel (which is more common when
        `all_touched=True`), the pixel is assigned according to rasterio’s rule:

            **the last geometry in the GeoDataFrame wins**.

        This reflects the standard behavior of `rasterio.features.rasterize`, where later
        shapes overwrite earlier ones when they map to the same pixel. No blending or
        multi-label assignment is performed—each pixel receives exactly one zone ID.

        Args:
            data:          Dataset key or ``xarray.DataArray`` containing the raster data on which
                           zonal statistics will be computed.
            zones:         The dataset key or xr.DataArray, or geopandas dataframe containing zones
                           if xr.DataArray the zones should be numbered linearly starting with 0.
                           Defaults to the geometry used when creating the AnalysisArea object, if available.
            zone_ids:      Unique zone identifiers corresponding to the zone raster labels
                           (0..n-1). Defaults to the index of the AnalysisArea geometry.
            stats:         List of statistics to calculate:
                           Options include 'mean', 'max', 'min', 'sum', 'std', 'var', 'count'. Defaults to 'mean'.
            return_xarray: Boolean flag to determine return type. 
                           If True, returns an xarray.DataArray. If False, returns a pandas.DataFrame.
            all_touched:   Controls rasterization inclusiveness when ``zones`` is a GeoDataFrame.

        Returns:
            zonal_stats_out: Zonal statistics grouped by zone ID, in the selected output format.
        """

        # Ensure data is xr.DataArray (TODO what about xr.Dataset?)
        if isinstance(data, list):
            data = self.get_dataset(data)

        # Ensure zones is xr.DataArray
        zone_ids, zones = self._resolve_zones(data, zone_ids, zones, all_touched)

        if stats is None:
            stats = ["mean"]

        # Determine return type based on return_xarray flag
        return_type = "xarray.DataArray" if return_xarray else "pandas.DataFrame"

        # Run zonal stats
        zonal_stats_out = zonal_stats(data, zones, zone_ids, stats, return_type)

        return zonal_stats_out

    def join_zonal_stats(self, zonal_stats_df):
        """
        Joins pandas Series with AnalysisArea.BASE_AREA_DATASET

        Args:
            zonal_stats_df: pd.Series to join, assumes multiindex of zones and time, with zones matching the index of AnalysisArea.BASE_AREA_DATASET

        Returns:
            joined_gdf: output geodataframe containing data from zonal_stats_df, with zones as index and time as columns.
        """
        # TODO Allow user choice of which zonal dataset to use
        gdf = self.get_dataset([self.BASE_AREA_DATASET])
        # Drop Name column as this is already in the index
        gdf = gdf.drop(columns="Name")

        joined_gdf = join_gdf(zonal_stats_df, gdf)

        return joined_gdf

    def _resolve_zones(
        self,
        data: xr.DataArray,
        zone_ids: list | np.ndarray = None,
        zones: list | xr.DataArray | gpd.GeoDataFrame = None,
        all_touched: bool = False,
    ):
        """
        Resolves zone_ids as list | np.ndarray and zones as xr.DataArray from user input
        """
        # Use base geometry as default
        if zones is None:
            zones = self.get_dataset([self.BASE_AREA_DATASET])

        if isinstance(zones, list):
            zones = self.get_dataset(zones)

        # Rasterize geodataframe
        if isinstance(zones, gpd.geodataframe.GeoDataFrame):
            # TODO: add check for information loss due to spatial resolution
            if zone_ids is None:
                unique_ids = zones.index.values
            zones = rasterize_gdf(
                zones, 
                data.odc.geobox,
                all_touched=all_touched,
            )

        # When zone_ids is provided, raise a ValueError if it doesn't match zones
        if zone_ids is not None:
            if len(zone_ids) != len(np.unique(zones)[np.unique(zones) >= 0]):
                raise ValueError(
                    "Mismatch between zone ids and unique zones provided or detected. "
                )
        else:
            zone_ids = unique_ids if "unique_ids" in locals() else None

        # Ensure nodata value is set
        if "nodata" not in zones.attrs:
            zones = zones.where(zones.notnull(), -1)
            zones.attrs["nodata"] = -1

        # Ensure zone_ids are present and aligned with values
        if zone_ids is None:
            zone_ids = np.unique(zones)[np.unique(zones) >= 0]
        if len(zone_ids) != len(np.unique(zones)[np.unique(zones) >= 0]):
            logging.warning(
                "Mismatch between zone ids and unique zones detected in the data. "
                "zone ids length: %d, unique zones detected: %d. "
                "zone_ids will be adjusted to match unique zone values.",
                len(zone_ids),
                len(np.unique(zones)[np.unique(zones) >= 0]),
            )
            zone_ids = zone_ids[np.unique(zones)[np.unique(zones) >= 0]]

        return zone_ids, zones

    def check_dataset_includes_date_range(
        self, dataset_key: list, begin: str = None, end: str = None
    ) -> bool:
        """
        Function to check data completeness for a specific time range.

        Args:
            dataset_key: The dataset key that will be passed in `get_dataset` to fetch the data.
            begin:       Start date of the date range to check completeness on, string of the form 'YYYY-mm-dd'.
            end:         End date of the date range to check completeness on, string of the form 'YYYY-mm-dd'.

        Returns:
            is_complete: tuple of 3 booleans indicating completeness of the dataset over the time range
                         (`begin` is in the dataset, `end` is in the dataset, there is data in range).
        """
        # Defaults to area.datetime_range if time range is not provided
        if begin is None:
            begin = self.datetime_range[:10]
        if end is None:
            end = self.datetime_range[11:]

        # Get data for the selected range without persisting it
        data = self.get_dataset(
            key=dataset_key,
            geobox=self.geobox,
            dates="/".join([begin, end]),
            persist=False,
        )

        # Check time range completeness
        if data is None or len(data.time) == 0:
            return (False, False, False)
        else:
            return (
                pd.Timestamp(begin).date() == pd.Timestamp(data.time[0].values).date(),
                pd.Timestamp(end).date() == pd.Timestamp(data.time[-1].values).date(),
                True,
            )
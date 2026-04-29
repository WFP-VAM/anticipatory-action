import geopandas as gpd
import numpy as np
import rasterio as rio
import shapely.geometry
import xarray as xr
from odc.geo.geobox import GeoBox
from rasterio.features import rasterize


def mask_to_polygons_layer(mask, transform, crs):
    """
    Convert a raster mask to a MultiPolygon gpd layer
    """

    all_polygons = []
    for shape, value in rio.features.shapes(
        mask.astype(np.int16), mask=(mask > 0), transform=rio.Affine(*transform)
    ):
        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])

    all_polygons = gpd.GeoDataFrame({"id": [1], "geometry": [all_polygons]}, crs=crs)
    all_polygons = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([all_polygons.unary_union]), crs=all_polygons.crs
    )

    return all_polygons


def get_vertices_from_polygon(geometry):
    if isinstance(geometry, shapely.geometry.MultiPolygon):
        vertices = []
        for geom in geometry.geoms:
            vertices.append([x for x in geom.exterior.coords])

    else:
        vertices = [[x for x in geometry.exterior.coords]]

    return vertices


def rasterize_gdf(
    zones: gpd.GeoDataFrame, geobox: GeoBox | xr.DataArray, all_touched: bool = False,
) -> xr.DataArray:
    """
    Function to convert a geopandas geodataframe into a raster

    Args:
        zones:     geopandas geodataframe containing zones
        geobox:   odc.geo.Geobox defining output coordinates/resolution. If xr.DataArray can be provided, its GeoBox will be used.

    Returns:
        zone_rast: xr.DataArray with geometries burned in, labelled [0,n-1] and -1 for missing.
    """

    if isinstance(geobox, xr.DataArray):
        # Get geobox from xr.DataArray provided
        geobox = geobox.odc.geobox

    geom = (
        zones.reset_index(drop=True)
        .reset_index()[["geometry", "index"]]
        .values.tolist()
    )
    zone_rast = rasterize(
        geom,
        out_shape=geobox.shape.yx,
        fill=-1,
        transform=geobox.affine,
        all_touched=all_touched,
    )

    zone_rast = xr.DataArray(
        data=zone_rast,
        coords={dim: coord.values for dim, coord in geobox.coordinates.items()},
        dims=geobox.dims,
    )
    zone_rast = zone_rast.where(zone_rast.notnull(), -1)
    zone_rast.attrs["nodata"] = -1

    return zone_rast